import os
import shutil
import socket
import subprocess
import time

import portpicker
import torch
from flask import Flask
from tensorboardX import SummaryWriter
from torch.optim.adam import Adam

from distar.ctools.utils import broadcast
from distar.ctools.utils.config_helper import read_config, deep_merge_dicts, save_config
from distar.ctools.worker.learner.base_learner import BaseLearner
from distar.ctools.worker.learner.learner_comm import LearnerComm
from distar.ctools.worker.learner.learner_hook import LearnerHook, add_learner_hook
from .rl_training.rl_dataloader import RLDataLoader
from .rl_training.rl_loss import ReinforcementLoss
from .model.model import Model


class RLLearner(BaseLearner):
    def __init__(self, cfg, *args):
        self._job_type = cfg.learner.job_type
        super(RLLearner, self).__init__(cfg, *args)
        self._player_id = cfg.learner.player_id
        if self._job_type == 'train':
            self.comm = LearnerComm(cfg)
            add_learner_hook(self._hooks, SendModelHook(position='after_iter'))
            add_learner_hook(self._hooks, SendModelHook(position='before_run'))
            add_learner_hook(self._hooks, SendTrainInfo(position='after_iter'))
            self._ip = os.environ.get('SLURMD_NODENAME') if 'SLURMD_NODENAME' in os.environ else '127.0.0.1'
            self._port = portpicker.pick_unused_port()
            self._save_grad = cfg.learner.get('save_grad') and self.rank == 0
            if self._save_grad:
                self.grad_tb_path = os.path.join(os.getcwd(), 'experiments', self._whole_cfg.common.experiment_name,
                                                 self.comm.player_id, 'grad')
                self.grad_tb_logger = SummaryWriter(self.grad_tb_path)
                self.clip_grad_tb_path = os.path.join(os.getcwd(), 'experiments',
                                                      self._whole_cfg.common.experiment_name,
                                                      self.comm.player_id, 'clip_grad')
                self.clip_grad_tb_logger = SummaryWriter(self.clip_grad_tb_path)
                self.model_tb_path = os.path.join(os.getcwd(), 'experiments', self._whole_cfg.common.experiment_name,
                                                  self.comm.player_id, 'model')
                self.model_tb_logger = SummaryWriter(self.model_tb_path)
                self.save_log_freq = self._whole_cfg.learner.get('save_log_freq', 400)
            self._dataloader = RLDataLoader(cfg=self._whole_cfg)
            model_ref = Model(self._whole_cfg, use_value_network=False).state_dict()
            self.comm.model_ref = {k: val.cpu().share_memory_() for k, val in model_ref.items()}
            self.comm._register_learner(self, self._ip, self._port, self._rank, self.world_size)
            self.comm.start_send_model()
            self._reset_value_flag = False
            self._update_config_flag = False
            self._reset_comm_setting_flag = False
            self._address_dir = os.path.join(os.getcwd(), 'experiments', self._whole_cfg.common.experiment_name,
                                             self._player_id, 'address')
            self._config_dir = os.path.join(os.getcwd(), 'experiments', self._whole_cfg.common.experiment_name,
                                            self.comm.player_id,'config')
            os.makedirs(self._address_dir, exist_ok=True)
            with open(os.path.join(self._address_dir, f'{self._ip}:{self._port}'), 'w') as f:
                f.write(f'rank:{self.rank}, ip:{self._ip}, port:{self._port},'
                        f' world_size:{self.world_size}'
                        f' player_id:{self._player_id}')
        self._remain_value_pretrain_iters = self._whole_cfg.learner.get('value_pretrain_iters', -1)

    def _setup_model(self):
        self._model = Model(self._whole_cfg, use_value_network=True)

    def _setup_loss(self):
        self._loss = ReinforcementLoss(self._whole_cfg.learner,self._whole_cfg.learner.player_id)

    def _setup_optimizer(self):
        self._optimizer = Adam(
            self.model.parameters(),
            lr=self._whole_cfg.learner.learning_rate,
            betas=(0, 0.99),
            eps=1e-5,
        )
        self._lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self._optimizer, milestones=[], gamma=1)

    def _train(self, data):
        with self._timer:
            self.step_value_pretrain()
            if self._remain_value_pretrain_iters > 0:
                staleness = 0
                staleness_std = 0
                staleness_max = 0
            else:
                model_last_iter = data.pop('model_last_iter')
                model_curr_iter = self.last_iter.val
                iter_diff = model_curr_iter - model_last_iter
                if iter_diff.shape[0] == 1:
                    staleness = iter_diff.item()
                    staleness_std = 0
                    staleness_max = iter_diff.item()
                else:
                    staleness_std, staleness, = torch.std_mean(iter_diff)
                    staleness_std = staleness_std.item()
                    staleness = staleness.item()
                    staleness_max = torch.max(iter_diff).item()
            model_output = self._model.rl_learner_forward(**data)
            if self._whole_cfg.learner.use_dapo:
                model_output['successive_logit'] = data['successive_logit']
            log_vars = self._loss.compute_loss(model_output)
            log_vars['entropy/reward'] = staleness
            log_vars['entropy/value'] = staleness_std
            log_vars['entropy/td'] = staleness_max

            loss = log_vars['total_loss']
        self._log_buffer['forward_time'] = self._timer.value

        with self._timer:
            self._optimizer.zero_grad()
            loss.backward()
            if self._use_distributed:
                self._model.sync_gradients()
            if self._save_grad and self._last_iter.val % self.save_log_freq == 0:
                for k, param in self._model.named_parameters():
                    if param.grad is not None:
                        self.grad_tb_logger.add_scalar(k, (torch.norm(param.grad)).item(),
                                                       global_step=self._last_iter.val)
                        self.model_tb_logger.add_scalar(k, (torch.norm(param.data)).item(),
                                                        global_step=self._last_iter.val)
            gradient = self._grad_clip.apply(self._model.parameters())
            if self._save_grad and self._last_iter.val % self.save_log_freq == 0:
                for k, param in self._model.named_parameters():
                    if param.grad is not None:
                        self.clip_grad_tb_logger.add_scalar(k, (torch.norm(param.grad)).item(),
                                                       global_step=self._last_iter.val)

            self._optimizer.step()
            # self._lr_scheduler.step()
        self._log_buffer['gradient'] = gradient
        self._log_buffer['backward_time'] = self._timer.value
        self._log_buffer.update(log_vars)
        if self._update_config_flag:
            self.update_config()
            self._update_config_flag = False
        if self._reset_value_flag:
            self.reset_value()
            self._reset_value_flag = False
        if self._reset_comm_setting_flag:
            self.reset_comm_setting()
            self._reset_comm_setting_flag = False

    def step_value_pretrain(self):
        if self._remain_value_pretrain_iters > 0:
            self._loss.only_update_value = True
            self._remain_value_pretrain_iters -= 1
            if self._use_distributed:
                self._model.module.only_update_baseline = True
                if self._rank == 0:
                    self._logger.info(f'only update baseline: {self._model.module.only_update_baseline}')
            else:
                self._model.only_update_baseline = True
                if self._rank == 0:
                    self._logger.info(f'only update baseline: {self._model.only_update_baseline}')

        elif self._remain_value_pretrain_iters == 0:
            self._loss.only_update_value = False
            self._remain_value_pretrain_iters -= 1
            if self._rank == 0:
                self._logger.info('value pretrain iter is 0')
            if self._use_distributed:
                self._model.module.only_update_baseline = False
                if self._rank == 0:
                    self._logger.info(f'only update baseline: {self._model.module.only_update_baseline}')
            else:
                self._model.only_update_baseline = False
                if self._rank == 0:
                    self._logger.info(f'only update baseline: {self._model.only_update_baseline}')

    def register_stats(self) -> None:
        """
        Overview:
            register some basic attributes to record & tb_logger(e.g.: cur_lr, data_time, train_time),
            register the attributes related to computation_graph to record & tb_logger.
        """
        super(RLLearner, self).register_stats()
        for k in ['total_loss', 'kl/extra_at', 'gradient']:
            self._record.register_var(k)
            self._tb_logger.register_var(k)

        for k1 in ['winloss', 'build_order', 'built_unit', 'effect', 'upgrade', 'battle', 'upgo', 'kl', 'entropy']:
            for k2 in ['reward', 'value', 'td', 'action_type', 'delay', 'queued', 'selected_units',
                       'target_unit', 'target_location', 'total']:
                k = k1 + '/' + k2
                self._record.register_var(k)
                self._tb_logger.register_var(k)

    def _setup_dataloader(self):
        if self._job_type == 'train':
            pass
        else:
            self._dataloader = FakeDataloader(unroll_len=self._whole_cfg.actor.traj_len,
                                              batch_size=self._whole_cfg.learner.data.batch_size)

    @property
    def cfg(self):
        return self._whole_cfg.learner

    def update_config(self):
        load_config_path = os.path.join(os.getcwd(), 'experiments', self._whole_cfg.common.experiment_name,
                                        f'user_config.yaml')
        load_config = read_config(load_config_path)
        player_id = self._whole_cfg.learner.player_id
        self._whole_cfg = deep_merge_dicts(self._whole_cfg, load_config)
        self._whole_cfg.learner.player_id = player_id
        self._setup_loss()
        self._remain_value_pretrain_iters = self._whole_cfg.learner.get('value_pretrain_iters', -1)
        if self.use_distributed:
            self.model.module.lstm_traj_infer = self._whole_cfg.learner.get('lstm_traj_infer', False)
        else:
            self.model.lstm_traj_infer = self._whole_cfg.learner.get('lstm_traj_infer', False)
        for g in self._optimizer.param_groups:
            g['lr'] = self._whole_cfg.learner.learning_rate
        print(f'update config from config_path:{load_config_path}')
        if self.rank == 0:
            time_label = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
            config_path = os.path.join(self._config_dir, f'user_config_{time_label}.yaml')
            save_config(self._whole_cfg, config_path)
            print(f'save_config to config_path:{config_path}')

    def _reset_value(self):
        ref_model = Model(self._whole_cfg, use_value_network=True)
        value_state_dict = {k: val for k, val in ref_model.state_dict().items() if 'value' in k or 'auxiliary' in k}
        self.model.load_state_dict(value_state_dict, strict=False)

    def reset_value(self):
        flag = torch.tensor([0])
        if self.rank == 0:
            flag = torch.tensor([1])
            self._reset_value()
        if self.world_size > 1:
            broadcast(flag, 0)
            if flag:
                self._setup_optimizer()
                self.model.broadcast_params()
        elif self.world_size == 1:
            self._setup_optimizer()
        print(f'reset_value')

    def reset_comm_setting(self):
        self.comm.close()
        del self.comm
        self._dataloader.close()
        del self._dataloader
        self._reset_comm()
        self._reset_dataloader()

    def _reset_comm(self):
        self.comm = LearnerComm(self._whole_cfg)
        self.comm._register_learner(self, self._ip, self._port, self._rank, self.world_size)
        model_ref = Model(self._whole_cfg, use_value_network=False).state_dict()
        self.comm.model_ref = {k: val.cpu().share_memory_() for k, val in model_ref.items()}
        self.comm.start_send_model()

    def _reset_dataloader(self):

        self._dataloader = RLDataLoader(data_source=self.comm.ask_for_metadata, cfg=self._whole_cfg)

    @staticmethod
    def create_rl_learner_app(learner):
        app = Flask(__name__)

        def build_ret(code, info=''):
            return {'code': code, 'info': info}

        # ************************** debug use *********************************
        @app.route('/rl_learner/update_config', methods=['GET'])
        def learner_update_config():
            learner._update_config_flag = True
            return {"done": "successfuly update config"}

        @app.route('/rl_learner/reset_comm_setting', methods=['GET'])
        def learner_reset_comm_setting():
            learner._update_config_flag = True
            learner._reset_comm_setting_flag = True
            return {"done": "successfuly reset_comm_setting"}

        @app.route('/rl_learner/reset_value', methods=['GET'])
        def learner_reset_value():
            learner._reset_value_flag = True
            learner._update_config_flag = True
            return {"done": "successfuly reset_value and update config"}
        return app


class SendModelHook(LearnerHook):
    def __init__(self, name='send_model_hook', position='after_iter', priority=40):
        super(SendModelHook, self).__init__(name=name, position=position, priority=priority)

    def __call__(self, engine):
        if self.position == 'before_run':
            engine.comm.send_model(engine, ignore_freq=True)
        elif self.position == 'after_iter':
            engine.comm.send_model(engine)


class SendTrainInfo(LearnerHook):
    def __init__(self, name='send_train_info_hook', position='after_iter', priority=60):
        super(SendTrainInfo, self).__init__(name=name, position=position, priority=priority)

    def __call__(self, engine):
        engine.comm.send_train_info(engine)

