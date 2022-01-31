import os
import os.path as osp

import torch
import torch.nn.functional as F
from distar.ctools.utils import get_rank, deep_merge_dicts, read_config

from distar.ctools.utils import try_import_link, get_rank, allreduce, get_world_size
from distar.ctools.torch_utils import sequence_mask

link = try_import_link()

default_config = read_config(osp.join(osp.dirname(__file__), "default_supervised_loss.yaml"))


class LabelSmoothingCrossEntropy(torch.nn.Module):
    """ NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor, reduce=False) -> torch.Tensor:
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        if reduce:
            return loss.mean()
        else:
            return loss


class SupervisedLoss:
    def __init__(self, cfg: dict) -> None:
        cfg = deep_merge_dicts(default_config, cfg)  # here, cfg is entire_cfg.learner
        self.whole_cfg = cfg
        self.cfg = cfg.learner
        # multiple loss to be calculate
        self.loss_func = {
            'action_type': self._action_type_loss,
            'delay': self._delay_loss,
            'queued': self._queued_loss,
            'selected_units': self._selected_units_loss,
            'target_unit': self._target_unit_loss,
            'target_location': self._target_location_loss,
        }

        self.loss_weight = self.cfg.loss_weight
        
        if self.cfg.get('label_smooth', False):
            self.criterion = LabelSmoothingCrossEntropy()
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.su_criterion = torch.nn.CrossEntropyLoss(reduction='none')  # su can not use label smooth cause of mask
        self.use_cuda = self.cfg.get('use_cuda', False) and torch.cuda.is_available()
        if self.use_cuda:
            self.device = torch.cuda.current_device()
        self.su_mask = self.cfg.su_mask
        self.cross_rank_loss = self.cfg.get('cross_rank_loss', False)
        self.total_batch_size = None
        self.rank = get_rank()
        self.world_size = get_world_size()

    def register_stats(self, record, tb_logger):
        record.register_var('total_loss')
        record.register_var('action_type_loss')
        record.register_var('delay_loss')
        record.register_var('queued_loss')
        record.register_var('selected_units_loss')
        record.register_var('selected_units_loss_norm')
        record.register_var('selected_units_end_flag_loss')
        record.register_var('target_unit_loss')
        record.register_var('target_location_loss')
        record.register_var('action_type_acc')
        record.register_var('delay_distance_L1')
        record.register_var('queued_acc')
        record.register_var('selected_units_iou')
        record.register_var('target_unit_acc')
        record.register_var('target_location_distance_L2')
        tb_logger.register_var('action_type_loss')
        tb_logger.register_var('action_type_acc')
        tb_logger.register_var('delay_loss')
        tb_logger.register_var('delay_distance_L1')
        tb_logger.register_var('queued_loss')
        tb_logger.register_var('queued_acc')
        tb_logger.register_var('selected_units_loss_norm')
        tb_logger.register_var('selected_units_end_flag_loss')
        tb_logger.register_var('selected_units_loss')
        tb_logger.register_var('selected_units_iou')
        tb_logger.register_var('target_unit_loss')
        tb_logger.register_var('target_unit_acc')
        tb_logger.register_var('target_location_loss')
        tb_logger.register_var('target_location_distance_L2')
        tb_logger.register_var('total_loss')

    def compute_loss(self, policy_logits, actions, actions_mask, selected_units_num, entity_num, infer_action_info):
        if self.cross_rank_loss:
            self.total_batch_size = torch.tensor(entity_num.shape[0], device=entity_num.device)
            allreduce(self.total_batch_size, reduce=False)
            print(self.rank, self.total_batch_size)
        loss_dict = {}
        for loss_item_name, loss_func in self.loss_func.items():
            if loss_item_name == 'selected_units':
                loss_dict.update(loss_func(
                    policy_logits[loss_item_name], actions[loss_item_name], actions_mask[loss_item_name],
                    selected_units_num, entity_num, infer_action_info['selected_units']))
            else:
                loss_dict.update(loss_func(policy_logits[loss_item_name], actions[loss_item_name],
                                          actions_mask[loss_item_name]))
        total_loss = 0.
        for name in self.loss_func.keys():
            total_loss += loss_dict[name + '_loss'] * self.loss_weight[name]
        loss_dict['total_loss'] = total_loss
        return loss_dict

    def _action_type_loss(self, logits, labels, mask):
        with torch.no_grad():
            acc_total = logits.argmax(dim=1) == labels
            acc_total = acc_total.float().sum() / len(labels)
        loss_tmp = self.criterion(logits, labels)
        loss_tmp *= mask

        if self.cross_rank_loss:
            loss = loss_tmp.mean()
            loss = (labels.shape[0] / self.total_batch_size * self.world_size) * loss
        else:
            valid_num = mask.sum()
            if valid_num > 0:
                loss = loss_tmp.sum() / valid_num
            else:
                loss = loss_tmp.sum() * 0
        return {'action_type_loss': loss, 'action_type_acc': acc_total}

    def _delay_loss(self, preds, labels, mask):
        loss_tmp = self.criterion(preds, labels)
        loss_tmp *= mask

        if self.cross_rank_loss:
            loss = loss_tmp.mean()
            loss = (labels.shape[0] / self.total_batch_size * self.world_size) * loss
        else:
            valid_num = mask.sum()
            if valid_num > 0:
                loss = loss_tmp.sum() / valid_num
            else:
                loss = loss_tmp.sum() * 0

        with torch.no_grad():
            preds = preds.argmax(dim=-1)
            delay_distance_l1 = ((preds - labels).abs() * mask).sum() / (mask.sum() + 1e-6)
        return {'delay_loss': loss, 'delay_distance_L1': delay_distance_l1}

    def _queued_loss(self, preds, labels, mask):
        loss_tmp = self.criterion(preds, labels)
        loss_tmp *= mask

        if self.cross_rank_loss:
            loss = loss_tmp.mean()
            loss = (labels.shape[0] / self.total_batch_size * self.world_size) * loss
        else:
            valid_num = mask.sum()
            if valid_num > 0:
                loss = loss_tmp.sum() / valid_num
            else:
                loss = loss_tmp.sum() * 0

        with torch.no_grad():
            preds = preds.argmax(dim=-1)
            acc = ((preds - labels).abs() * mask).sum() / (mask.sum() + 1e-6)
        return {'queued_loss': loss, 'queued_acc': acc}

    def _selected_units_loss(self, logits, labels, mask, lengths, entity_num, selected_units):
        b, s, n = logits.shape
        if self.su_mask:
            length_wo_end_flag = (lengths - 1).clamp_(min=0)
            length_mask = sequence_mask(length_wo_end_flag, max_len=labels.shape[1])
            new_labels = labels.clone()
            new_labels[~length_mask] = logits.shape[-1]
            new_labels = new_labels[:, :s]
            logits = torch.cat([logits, torch.zeros((*logits.shape[:2], 1), device=logits.device)], dim=-1)
            logits_mask = torch.ones_like(logits, device=logits.device)
            logits_mask = torch.scatter(logits_mask, dim=2,
                                        index=new_labels.unsqueeze(dim=1).repeat(1, logits.shape[1], 1), value=0.)

            logits_mask = torch.scatter(logits_mask, dim=2, index=new_labels.unsqueeze(dim=2), value=1.)
            # logits = logits_mask * logits
            logits = logits.masked_fill(~logits_mask.bool(), value=-1e9)
            logits = logits[:, :, :-1]
        select_mask = sequence_mask(lengths, max_len=logits.shape[1])
        logits_flat = logits.view(-1, n)
        labels = labels[:, :s]
        labels_flat = labels.contiguous().view(-1)
        loss_tmp = self.su_criterion(logits_flat, labels_flat)
        loss_tmp = loss_tmp.view(b, s)
        # mask invalid selected units
        loss_tmp = loss_tmp.masked_fill_(~select_mask, 0)
        # mask invalid actions
        loss_tmp *= mask.unsqueeze(dim=1)

        if self.cross_rank_loss:
            loss = loss_tmp.sum() / b
            loss = (labels.shape[0] / self.total_batch_size * self.world_size) * loss
        else:
            loss = loss_tmp.sum() / b
        loss_norm = loss_tmp.sum() / (sum(lengths) + 1e-6)
        end_flag_loss = loss_tmp[torch.arange(b), lengths - 1].mean()

        # acc
        with torch.no_grad():
            if selected_units is not None:
                preds = selected_units
                end_flag_index = (preds == entity_num.unsqueeze(dim=1)).long()
                end_flag_index = torch.sort(end_flag_index, dim=-1, descending=True)[1][:, 0]
                invalid_end_flag_index = end_flag_index == 0
                end_flag_index += 1
                end_flag_index[invalid_end_flag_index] += s
                preds = preds + 1
                labels = labels + 1
                preds_mask = sequence_mask(end_flag_index, max_len=s)
                labels = labels * select_mask
                preds = preds * preds_mask

                preds = torch.scatter(torch.zeros(b, n + 1, device=labels.device, dtype=torch.bool), dim=1,
                                      index=preds.long(), value=1)
                labels = torch.scatter(torch.zeros(b, n + 1, device=labels.device, dtype=torch.bool), dim=1,
                                       index=labels.long(), value=1)
                intersection = (preds & labels)[:, 1:].sum(dim=1)
                union = (preds | labels)[:, 1:].sum(dim=1)
                iou = (intersection / (union + 1e-6) * mask).sum() / (mask.sum() + 1e-6)
            else:
                iou = 0.

        return {'selected_units_loss': loss, 'selected_units_loss_norm': loss_norm,
                'selected_units_end_flag_loss': end_flag_loss, 'selected_units_iou': iou}

    def _target_unit_loss(self, logits, labels, mask):
        assert labels.max() <= logits.shape[1], '{}, {}'.format(labels.max(), logits.shape[1])
        loss_tmp = self.criterion(logits, labels)
        loss_tmp *= mask

        if self.cross_rank_loss:
            loss = loss_tmp.mean()
            loss = (labels.shape[0] / self.total_batch_size * self.world_size) * loss
        else:
            valid_num = mask.sum()
            if valid_num > 0:
                loss = loss_tmp.sum() / valid_num
            else:
                loss = loss_tmp.sum() * 0
        # acc
        with torch.no_grad():
            logits = logits.argmax(dim=-1)
            acc = ((logits == labels) * mask).sum() / (mask.sum() + 1e-6)
        return {'target_unit_loss': loss, 'target_unit_acc': acc}

    def _target_location_loss(self, logits, labels, mask):
        W = 160
        loss_tmp = self.criterion(logits, labels)
        loss_tmp *= mask

        if self.cross_rank_loss:
            loss = loss_tmp.mean()
            loss = (labels.shape[0] / self.total_batch_size * self.world_size) * loss
        else:
            valid_num = mask.sum()
            if valid_num > 0:
                loss = loss_tmp.sum() / valid_num
            else:
                loss = loss_tmp.sum() * 0
        # acc
        with torch.no_grad():
            preds = logits.argmax(dim=-1).unsqueeze(dim=1)
            y = preds // W
            x = preds % W
            preds = torch.cat([x, y], dim=1)
            labels = labels.unsqueeze(dim=1)
            y = labels // W
            x = labels % W
            labels = torch.cat([x, y], dim=1)
            target_location_distance_L2 = ((preds - labels).pow_(2).sum(dim=-1).float().sqrt_() * mask).sum() / (
                        mask.sum() + 1e-6)
        return {'target_location_loss': loss, 'target_location_distance_L2': target_location_distance_L2}
