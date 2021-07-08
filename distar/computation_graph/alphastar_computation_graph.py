"""
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. base class for supervised learning on linklink, including basic processes.
"""
import os.path as osp
from collections import namedtuple, OrderedDict

import torch
import torch.nn.functional as F
import numpy as np

from ctools.computation_graph import BaseCompGraph
from ctools.worker.agent import BaseAgent
from distar.computation_graph.as_rl_utils import td_lambda_loss
from ctools.utils import lists_to_dicts, get_rank, deep_merge_dicts, read_config
from distar.computation_graph.as_rl_utils import compute_neg_log_prob, vtrace_advantages, upgo_returns
from ctools.torch_utils import to_device

default_config = read_config(osp.join(osp.dirname(__file__), "alphastar_computation_graph_default_config.yaml"))

short = {'winloss': 'winloss', 'build_order': 'bo', 'built_unit': 'bu', 'effect': 'effect', 'upgrade': 'upgrade',
         'battle': 'battle', 'upgo': 'upgo', 'kl': 'kl', 'entropy': 'entropy',
         'action_type': 'at', 'delay': 'delay', 'queued': 'queued', 'selected_units': 'su',
         'target_units': 'tu', 'target_location': 'tl'}


filter_index = [
    16, 20, 24, 25, 32, 33, 34, 37, 41, 42, 48, 53, 54, 55, 56, 65, 71, 75, 80, 86,
    87, 90, 93, 94, 102, 103, 109, 112, 115, 125, 126, 127, 133, 134, 205, 208, 213,
    214, 216, 224, 227, 232, 234, 235, 238, 256, 258, 259, 260, 261, 262, 263, 264,
]
train_index = [
    269, 274, 278, 283, 285, 292, 295, 298, 301, 306, 309, 311, 322,
    162, 164, 167, 168, 171, 172, 176, 178, 181,
]

class AlphaStarCompGraph(BaseCompGraph):

    def __init__(self, cfg: dict) -> None:
        cfg = deep_merge_dicts(default_config.learner, cfg)
        self.cfg = cfg
        self.action_keys = ['action_type', 'delay', 'queued', 'selected_units', 'target_units', 'target_location']
        self.rollout_outputs = namedtuple(
            "rollout_outputs", [
                'target_outputs', 'behaviour_outputs', 'teacher_outputs', 'baselines', 'rewards', 'actions',
                'game_seconds', 'mask'
            ]
        )
        self.T = cfg.unroll_len
        self.action_type_kl_seconds = cfg.kl.action_type_kl_seconds

        self.loss_weights = cfg.loss_weights
        self.gammas = cfg.gammas

        self.rank = get_rank()
        self.device = 'cuda:{}'.format(self.rank % torch.cuda.device_count()) if cfg.use_cuda and torch.cuda.is_available() else 'cpu'
        # self.pad_value = -1e6
        # self.fake_learner = self.cfg.get('fake_learner', False)
        self.use_cuda = self.cfg.get('use_cuda', False) and torch.cuda.is_available()

    def forward(self, data: dict, agent: BaseAgent) -> dict:
        rollout_outputs = self.rollout(data, agent)
        # TODO(nyz) apply the importance sampling weight in gradient update
        target_outputs, behaviour_outputs, teacher_outputs, baselines = rollout_outputs[:4]
        rewards, actions, game_seconds, mask = rollout_outputs[4:]

        # td_lambda and v_trace
        actor_critic_loss = 0.
        loss_show = {}

        # compute negative log probability
        target_neg_logp = {}
        behaviour_neg_logp = {}
        for k, v in actions.items():
            if k == 'selected_units':
                target_neg_logp[k] = compute_neg_log_prob(target_outputs[k], actions[k], mask['selected_units_mask'])
                behaviour_neg_logp[k] = compute_neg_log_prob(behaviour_outputs[k], actions[k], mask['selected_units_mask'])
            else:
                target_neg_logp[k] = compute_neg_log_prob(target_outputs[k], actions[k])
                behaviour_neg_logp[k] = compute_neg_log_prob(behaviour_outputs[k], actions[k])

        for field, baseline in baselines.items():
            reward = rewards[field]
            # td_lambda_loss = self._td_lambda_loss(baseline, reward) * self.loss_weights.baseline[field]
            td_lambda_loss = self._td_lambda_loss(baseline, reward, gamma=self.gammas.baseline[field])
            loss_show[short[field] + '_td'] = td_lambda_loss.item()
            td_lambda_loss *= self.loss_weights.baseline[field]
            vtrace_loss, vtrace_loss_dict = self._vtrace_pg_loss(
                baseline, reward, target_neg_logp, behaviour_neg_logp, mask['actions_mask'], gamma=self.gammas.pg[field]
            )
            for k, v in vtrace_loss_dict.items():
                loss_show[short[field] + '_' + short[k]] = v

            loss_show[short[field] + '_total'] = vtrace_loss.item()
            vtrace_loss *= self.loss_weights.pg[field]
            actor_critic_loss += td_lambda_loss + vtrace_loss
        # upgo loss
        upgo_loss, upgo_loss_dict = self._upgo_loss(
            baselines['winloss'], rewards['winloss'], target_neg_logp, behaviour_neg_logp, mask['actions_mask'],
        )
        for k, v in upgo_loss_dict.items():
            loss_show['upgo' + '_' + short[k]] = v
        loss_show['upgo' + '_total'] = upgo_loss.item()
        upgo_loss *= self.loss_weights.upgo['winloss']
        # human kl loss
        kl_loss, action_type_kl_loss, filter_kl_loss, kl_loss_dict = self._human_kl_loss(target_outputs, teacher_outputs, game_seconds, mask)
        for k, v in kl_loss_dict.items():
            loss_show['kl' + '_' + short[k]] = v
        loss_show['kl' + '_total'] = kl_loss.item()
        loss_show['kl_reward'] = action_type_kl_loss.item()  # replace extra action type loss to kl_reward for log show
        loss_show['kl_td'] = filter_kl_loss.item()  # replace extra action type loss to kl_td for log show
        kl_loss *= self.loss_weights.kl
        action_type_kl_loss *= self.loss_weights.action_type_kl
        filter_kl_loss *= self.loss_weights.filter_kl

        # entropy loss
        ent_loss, ent_loss_dict = self._entropy_loss(target_outputs, mask)
        for k, v in ent_loss_dict.items():
            loss_show['entropy' + '_' + short[k]] = v
        loss_show['entropy' + '_total'] = ent_loss.item()
        ent_loss *= self.loss_weights.entropy

        total_loss = actor_critic_loss + kl_loss + action_type_kl_loss + ent_loss + upgo_loss + filter_kl_loss
        ret = {
            'total_loss': total_loss,
        }
        ret.update(loss_show)
        ret.update({short[k] + '_reward': v.mean() for k, v in rewards.items()})
        ret.update({short[k] + '_value': v.mean() for k, v in baselines.items()})
        return ret

    def rollout(self, data, agent):
        if self.use_cuda:
            data = to_device(data, 'cuda') # use this line when using fake actor
        # temperature = self.temperature_scheduler.step()
        outputs_dict = OrderedDict({k: [] for k in self.rollout_outputs._fields})
        outputs = agent.forward(data)
        outputs_dict['target_outputs'] = outputs['policy_outputs']
        outputs_dict['behaviour_outputs'] = data['behaviour_output']
        outputs_dict['teacher_outputs'] = data['teacher_output']
        outputs_dict['baselines'] = outputs['baselines']._asdict()
        outputs_dict['rewards'] = data['reward']
        outputs_dict['actions'] = data['actions']
        outputs_dict['game_seconds'] = data['game_second']
        outputs_dict['mask'] = data['mask']
        flag = outputs_dict['rewards']['winloss'][-1] == 0
        for k, v in outputs_dict['baselines'].items():
            outputs_dict['baselines'][k][-1] *= flag
        return self.rollout_outputs(*outputs_dict.values())

    def _td_lambda_loss(self, baseline, reward, gamma=1.0):
        """
            default: gamma=1.0, lamda=0.8
        """
        assert (isinstance(baseline, torch.Tensor) and baseline.shape[0] == self.T + 1)
        assert (isinstance(reward, torch.Tensor) and reward.shape[0] == self.T)
        return td_lambda_loss(baseline, reward, gamma=gamma)

    def _vtrace_pg_loss(self, baseline, reward, target_neg_logp, behaviour_neg_logp, mask, gamma=1.0):
        """
            seperated vtrace loss
        """
        loss = 0.
        loss_dict = {}
        for k in self.action_keys:
            with torch.no_grad():
                # rho = target_prob / behaviour_prob
                clipped_rhos = torch.exp((behaviour_neg_logp[k] - target_neg_logp[k])).clamp_(max=1)
                clipped_rhos = clipped_rhos.view(baseline.shape[0] - 1, baseline.shape[1])  # t - 1, b
                advantages = vtrace_advantages(clipped_rhos, clipped_rhos, reward, baseline, gamma, lambda_=0.8)
                advantages = advantages.view(-1)

            head_loss = advantages * target_neg_logp[k] * mask[k]
            head_loss = head_loss.mean()
            loss_dict[k] = head_loss.item()
            loss += head_loss
        return loss, loss_dict

    def _upgo_loss(self, baseline, reward, target_neg_logp, behaviour_neg_logp, mask):
        loss = 0.
        loss_dict = {}
        for k in self.action_keys:
            with torch.no_grad():
                # rho = target_prob / behaviour_prob
                clipped_rhos = torch.exp((behaviour_neg_logp[k] - target_neg_logp[k])).clamp_(max=1)
                advantages = upgo_returns(reward, baseline) - baseline[:-1]
                advantages = advantages.view(-1)

            head_loss = clipped_rhos * advantages * target_neg_logp[k] * mask[k]
            head_loss = head_loss.mean()
            loss_dict[k] = head_loss.item()
            loss += head_loss
        return loss, loss_dict

    def _human_kl_loss(self, target_outputs, teacher_outputs, game_seconds, mask):
        loss_dict = {}

        def kl(stu, tea, mask, k, flag=None):
            tea = F.log_softmax(tea, dim=-1)
            stu = F.log_softmax(stu, dim=-1)
            tea_probs = torch.exp(tea)

            kl = tea_probs * (tea - stu)
            if k == 'selected_units':
                kl *= mask['selected_units_mask'].unsqueeze(dim=2)
                kl = kl.sum(dim=1)
                kl = kl.sum(dim=-1)
            elif k == 'target_units':
                kl = kl.sum(dim=-1)
            else:
                kl = kl.sum(dim=-1)

            if flag is not None:
                kl *= flag
            kl = kl * mask['actions_mask'][k]
            kl = kl.mean()
            return kl

        def filter_kl(stu, tea, mask, k, build_fac, train_fac):
            tea = F.log_softmax(tea, dim=-1)
            stu = F.log_softmax(stu, dim=-1)
            tea_probs = torch.exp(tea)
            factor_filter = tea_probs[:, filter_index].max(dim=-1).values
            factor_train = tea_probs[:, train_index].max(dim=-1).values
            factor_mask = factor_filter > factor_train
            factor = factor_mask * factor_filter * build_fac + ~factor_mask * factor_train * train_fac

            kl = tea_probs * (tea - stu)
            kl = kl.sum(dim=-1)

            kl = kl * mask['actions_mask'][k] * factor
            kl = kl.mean()
            return kl


        kl_loss = torch.zeros(1).to(dtype=torch.float32, device=self.device)
        for k in self.action_keys:
            tmp_loss = kl(target_outputs[k], teacher_outputs[k], mask, k)
            loss_dict[k] = tmp_loss.item()
            kl_loss += tmp_loss

        flag = game_seconds < self.action_type_kl_seconds
        action_type_kl_loss = kl(
            target_outputs['action_type'], teacher_outputs['action_type'], mask, 'action_type', flag
        )

        filter_kl_loss = filter_kl(
            target_outputs['action_type'], teacher_outputs['action_type'], mask, 'action_type',
            self.loss_weights.filter_kl_build, self.loss_weights.filter_kl_train
        )

        return kl_loss, action_type_kl_loss, filter_kl_loss, loss_dict

    def _entropy_loss(self, target_outputs, mask):
        loss = torch.zeros(1).to(dtype=torch.float, device=self.device)
        loss_dict = {}
        for k in self.action_keys:
            policy = target_outputs[k]
            log_policy = F.log_softmax(policy, dim=-1)
            policy = torch.exp(log_policy)
            ent = -policy * log_policy

            if k == 'selected_units':
                ent *= mask['selected_units_mask'].unsqueeze(dim=2)
                ent = ent.sum(dim=1)
                ent = ent.sum(dim=-1) / torch.log(mask['selected_units_logits_mask'].sum(dim=-1).float())  # normalize
            elif k == 'target_units':
                ent = ent.sum(dim=-1) / torch.log(mask['target_units_logits_mask'].sum(dim=-1).float())  # normalize
            else:
                ent = ent.sum(dim=-1) / torch.log(torch.FloatTensor([ent.shape[-1]]).to(ent.device))

            ent = ent * mask['actions_mask'][k]
            tmp_loss = ent.mean()
            loss_dict[k] = tmp_loss.item()
            loss += tmp_loss
        return - loss, loss_dict

    def __repr__(self):
        return "AlphaStarCompGraph"
