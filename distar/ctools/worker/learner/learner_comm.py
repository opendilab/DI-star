import logging
import os
import sys
import time
import traceback
import platform
import torch.multiprocessing as tm

from distar.ctools.torch_utils.data_helper import to_device
from functools import partial

import requests
import torch
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from distar.ctools.utils.file_helper import redis,dumps
from distar.ctools.utils import broadcast
from distar.ctools.worker.coordinator.adapter import Adapter

class LearnerComm:
    def __init__(self, cfg):
        self._whole_cfg = cfg
        self.player_id = self._whole_cfg.learner.player_id
        self._send_model_freq = self._whole_cfg.communication.learner_send_model_freq
        self._send_model_count = 0
        self._send_train_info_freq = self._whole_cfg.communication.learner_send_train_info_freq
        self._send_train_info_count = 0
        self._logger = logging.getLogger('default_logger')
        self._send_model_worker_num = self._whole_cfg.communication.learner_send_model_worker_num

        # communication related
        self._league_url_prefix = 'http://{}:{}/'.format(
            self._whole_cfg.communication.coordinator_ip, self._whole_cfg.communication.league_port)
        self._requests_session = requests.session()
        retries = Retry(total=20, backoff_factor=1)
        self._requests_session.mount('http://', HTTPAdapter(max_retries=retries))

        self._model_fs_type = self._whole_cfg.communication.get('model_fs_type', 'torch')


    def _register_learner(self, learner,ip,port,rank,world_size) -> None:
        request_info = {'player_id': self.player_id,'ip':ip,'port':port,'rank':rank,'world_size':world_size}
        while True:
            result = self._flask_send(request_info, 'league/register_learner',to_league=True)
            if result is not None and result['code'] == 0:
                if not (learner._load_path and os.path.exists(learner._load_path)):
                    learner.load_path = result['info']['ckpt_path']
                # print(f'learner load path is :{learner._load_path}')
                return
            else:
                time.sleep(1)

    def start_send_model(self):
        context_str = 'spawn' if platform.system().lower() == 'windows' else 'fork'
        self.mp_fork_context = tm.get_context(context_str)
        self._model_parent_conn, self._model_child_conn = [], []
        self._send_model_processes = []
        for _ in range(self._send_model_worker_num):
            p, c = self.mp_fork_context.Pipe()
            self._model_parent_conn.append(p)
            process = self.mp_fork_context.Process(target=self._send_model_loop,
                                                      args=(c,
                                                            self._model_ref,
                                                            self._whole_cfg,
                                                            self.player_id,
                                                            self._model_fs_type),
                                                      daemon=True)
            self._send_model_processes.append(process)
            process.start()
        torch.set_num_threads(max(torch.get_num_threads() // 8, 1))

    def send_model(self, learner, ignore_freq=False, reset_flag=False) -> None:
        if ignore_freq or (self._send_model_count % self._send_model_freq == 0 and learner._remain_value_pretrain_iters< 0) :
            state_dict = to_device({k: v for k, v in learner.model.state_dict().items() if 'value_networks' not in k and 'value_encoder' not in k},device='cpu')
            for k,val in state_dict.items():
                self._model_ref[k].copy_(val)
            for i in range(self._send_model_worker_num):
                self._model_parent_conn[i].send((learner.last_iter.val, reset_flag))
            
        if not ignore_freq:
            self._send_model_count += 1

    @staticmethod
    def _send_model_loop(model_child_conn,model_ref, cfg, player_id, model_fs_type):
        torch.set_num_threads(1)
        adapter = Adapter(cfg=cfg, maxlen=1)
        worker_num = cfg.communication.adapter_model_worker_num
        model_last_iter, reset_flag = model_child_conn.recv()
        state_dict = {'model':  model_ref,'model_last_iter': 0, 'reset_flag': False}
        data = dumps(state_dict, fs_type=model_fs_type, compress=True)
        while True:
            if model_child_conn.poll():
                model_last_iter, reset_flag = model_child_conn.recv()
                state_dict = {'model':  model_ref,'model_last_iter': model_last_iter, 'reset_flag': reset_flag}
                data = dumps(state_dict, fs_type=model_fs_type, compress=True)

                adapter.push(data, token=player_id + 'model', worker_num=worker_num)
            if not adapter.full(player_id + 'model'):
                adapter.push(data, player_id + 'model', worker_num=worker_num)

    def send_train_info(self, learner):
        flag = torch.tensor([0])
        self._send_train_info_count += 1
        reset_checkpoint_path = 'none'
        if learner.rank == 0 and self._send_train_info_count % self._send_train_info_freq == 0:
            frames = int(
                self._send_train_info_freq * learner._world_size * self._whole_cfg.learner.data.batch_size * self._whole_cfg.actor.traj_len)
            request_info = {'player_id': self.player_id, 'train_steps': frames,
                            'checkpoint_path': os.path.abspath(learner.last_checkpoint_path)}
            for try_times in range(10):
                result = self._flask_send(request_info, 'league/learner_send_train_info',to_league=True)
                if result is not None and result['code'] == 0:
                    reset_checkpoint_path = result['info']['reset_checkpoint_path']
                    break
                else:
                    time.sleep(1)
            if reset_checkpoint_path != 'none':
                flag = torch.tensor([1])
                learner.checkpoint_manager.load(
                    reset_checkpoint_path,
                    model=learner.model,
                    logger_prefix='({})'.format(learner.name),
                    strict=False,
                    info_print=learner.rank == 0,
                )
                learner._reset_value()
                learner._remain_value_pretrain_iters = self._whole_cfg.learner.get('value_pretrain_iters', -1)
                self._logger.info(
                    '{} reset checkpoint in {}!!!!!!!!!!!!!!!!!'.format(learner.comm.player_id, reset_checkpoint_path))
                learner.comm.send_model(learner, ignore_freq=True, reset_flag=True)
        if learner.world_size > 1:
            broadcast(flag, 0) 
            if flag:
                learner._remain_value_pretrain_iters = self._whole_cfg.learner.get('value_pretrain_iters', -1)
                learner._setup_optimizer()
                learner.model.broadcast_params()
                learner.comm.send_model(learner, ignore_freq=True, reset_flag=True)

    def _flask_send(self, data: dict, api: str, to_league=False) -> dict:
        response = None
        t = time.time()
        try:
            if to_league:
                response = requests.post(self._league_url_prefix + api, json=data).json()
            else:
                response = requests.post(self._coordinator_url_prefix + api, json=data).json()
            if response['code'] == 0:
                pass
                # self._logger.info(
                #     "{} succeed sending result: {}, cost time: {:.4f}".format(api, self.player_id, time.time() - t))
            else:
                self._logger.error(
                    "{} failed to send result: {}, cost time: {:.4f}".format(api, self.player_id, time.time() - t))
        except Exception as e:
            self._logger.error(''.join(traceback.format_tb(e.__traceback__)))
            self._logger.error("api({}): {}".format(api, sys.exc_info()))
        return response

    @property
    def model_ref(self):
        return self._model_ref

    @model_ref.setter
    def model_ref(self,model_ref):
        self._model_ref = model_ref

    def close(self):
        if hasattr(self,'_send_model_process'):
            self._send_model_process.terminate()
        time.sleep(1)
        print('close subprocess in learner_comm')
