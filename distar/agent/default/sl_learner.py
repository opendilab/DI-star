import torch

from easydict import EasyDict
from copy import deepcopy

from distar.ctools.worker.learner.base_learner import BaseLearner
from .model.model import Model
from .sl_training.sl_dataloader import SLDataloader, FakeDataloader
from .sl_training.sl_loss import SupervisedLoss
from distar.ctools.torch_utils.grad_clip import build_grad_clip


class SLLearner(BaseLearner):
    def _setup_model(self):
        self._model = Model(self._whole_cfg, temperature=1.0)
        self._grad_clip = build_grad_clip(self._whole_cfg.learner.grad_clip)
        self.num_layers = self._model.cfg.encoder.core_lstm.num_layers
        self.hidden_size = self._model.cfg.encoder.core_lstm.hidden_size

        zero_tensor = torch.zeros(self._whole_cfg.learner.data.batch_size, self.hidden_size)
        if self._whole_cfg.learner.use_cuda:
            zero_tensor = zero_tensor.cuda()
        self.hidden_state = [(zero_tensor, zero_tensor) for _ in range(self.num_layers)]
        self.ignore_step = 0
        self.debug = self._whole_cfg.learner.get('debug', False)
        if self.debug:
            print('[INFO] enable debug!!!!!1')
            self.debug_loss = {'action_type_loss': 0., 'delay_loss': 0., 'selected_units_loss_norm': 0., 
            'target_unit_loss': 0., 'target_location_loss': 0.}

    def reset_hidden_state(self, new_episodes):
        for l in range(self.num_layers):
            self.hidden_state[l] = (self.hidden_state[l][0].clone().detach(), self.hidden_state[l][1].clone().detach())
            self.hidden_state[l][0][new_episodes] *= 0
            self.hidden_state[l][1][new_episodes] *= 0

    def _setup_loss(self):
        self._loss = SupervisedLoss(self._whole_cfg)

    def _setup_dataloader(self):
        if self._whole_cfg.learner.job_type == 'train':
            self._dataloader = SLDataloader(self._whole_cfg)
        else:
            self._dataloader = FakeDataloader(self._whole_cfg)

    def _train(self, data):
        with self._timer:
            new_episodes = data.pop('new_episodes')
            self.reset_hidden_state(new_episodes)
            logits, infer_action_info, hidden_state = self._model.sl_train(**data, hidden_state=self.hidden_state)
            log_vars = self._loss.compute_loss(logits, data['action_info'], data['action_mask'],
                                               data['selected_units_num'], data['entity_num'], infer_action_info)
            loss = log_vars['total_loss']
        self._log_buffer['forward_time'] = self._timer.value
        if self.debug:
            for k, v in self.debug_loss.items():
                self.debug_loss[k] = v * 0.95 + log_vars[k].item() * 0.05
                if log_vars[k] > v * 10 and self.last_iter.val > 200:
                    self.save_checkpoint()
                    torch.save((data, self.hidden_state, log_vars, logits), '{}_iter_{}_rank_{}.pth'.format(k, self.last_iter.val, self._rank))
        with self._timer:
            if self.ignore_step > 5:
                self._optimizer.zero_grad()
                loss.backward()
                if self._use_distributed:
                    self._model.sync_gradients()
                gradient = self._grad_clip.apply(self._model.parameters())
                self._optimizer.step()
                self._lr_scheduler.step()
            else:
                gradient = 0.
            self.ignore_step += 1
        self.hidden_state = hidden_state
        self._log_buffer['gradient'] = gradient
        self._log_buffer['backward_time'] = self._timer.value
        self._log_buffer.update(log_vars)


if __name__ == '__main__':
    cfg = EasyDict({'learner': {'use_cuda': False,
        'data': {'train_data_file': r'C:\work\repo\artofwar\artofwar\data\1.txt',
                                         'cache_size': 1,
                                         'num_workers': 1}}})
    learner = SLLearner(cfg)
    learner.run()
