import os
import platform
import sys
import time
import traceback
import uuid
import threading
from copy import deepcopy
import requests
import torch
import torch.multiprocessing as tm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import numpy as np
from distar.agent.import_helper import import_module
from distar.ctools.utils.file_helper import dumps
from distar.ctools.utils.file_helper import save_traj_file, loads
from collections import deque
from distar.ctools.worker.coordinator.adapter import Adapter


def update_model_loop(cfg, model_ref, signal_queue, update_interval, model_last_iter_dict, avg_update_model_time):
    adapter = Adapter(cfg)
    last_update_time = time.time()
    update_model_time = deque(maxlen=100)
    last_reset_flag = {k: False for k in model_ref.keys()}
    while True:
        if time.time() - last_update_time > update_interval:
            reset_flag = False
            for player_id, model in model_ref.items():
                start_time = time.time()
                state_dict = adapter.pull(token=player_id + 'model', sleep_time=0.5, worker_num=1)
                model_last_iter = torch.Tensor([state_dict['model_last_iter']])
                model_last_iter_dict[player_id].copy_(model_last_iter)
                for k, v in state_dict['model'].items():
                    model[k].copy_(v)
                update_model_time.append(time.time() - start_time)
                avg_time = torch.Tensor([np.mean(update_model_time)])
                avg_update_model_time.copy_(avg_time)
                if not last_reset_flag[player_id] and state_dict['reset_flag']:
                    reset_flag = True
                last_reset_flag[player_id] = state_dict['reset_flag']
            last_update_time = time.time()
            signal_queue.put(reset_flag)


class ActorComm:
    def __init__(self, cfg, actor_uid, logger):
        self._whole_cfg = cfg
        self._actor_uid = actor_uid
        self._last_update_time = -1
        self._model_update_interval = self._whole_cfg.communication.actor_model_update_interval
        self._logger = logger
        self._fs_type = self._whole_cfg.learner.data.get('fs_type', 'pickle')
        self._model_fs_type = self._whole_cfg.communication.get('model_fs_type', 'pyarrow')
        self._league_url_prefix = 'http://{}:{}/'.format(
            self._whole_cfg.communication.coordinator_ip, self._whole_cfg.communication.league_port)
        self._requests_session = requests.session()
        retries = Retry(total=20, backoff_factor=1)
        self._requests_session.mount('http://', HTTPAdapter(max_retries=retries))
        self._send_result_num_workers = self._whole_cfg.communication.get("send_result_num_workers", 1)

        self._update_model_time = deque(maxlen=100)
        self._avg_update_model_time = torch.Tensor([0]).share_memory_()
        self.model_last_iter_dict = {}
        self._adapter = Adapter(cfg=self._whole_cfg, maxlen=3)
        self.worker_num = self._whole_cfg.communication.adapter_traj_worker_num

    def ask_for_job(self, actor):
        request_info = {'job_type':actor._league_job_type}
        while True:
            result = self._flask_send(request_info, 'league/actor_ask_for_job', to_league=True)
            if result is not None and result['code'] == 0:
                job = result['info']
                print(job)
                self.job = job
                self.tmp_model_path = {}
                self.tmp_traj_path = {}
                actor.agents = []
                actor.models = {}
                actor.successive_models = {}
                teacher_models = {}
                for idx, (player_id, pipeline, z_path ,z_prob, side_id) in enumerate(zip(self.job['player_ids'],
                                                                    self.job['pipelines'], self.job['z_path'], self.job['z_prob'], self.job['side_ids'])):

                    if 'bot' in pipeline:
                        continue
                    Agent = import_module(pipeline, 'Agent')
                    merged_agent_cfg = deepcopy(self._whole_cfg)
                    merged_agent_cfg.agent.z_path = z_path
                    agent = Agent(merged_agent_cfg)
                    agent.player_id = player_id
                    agent.side_id = side_id
                    agent._fake_reward_prob = z_prob
                    if job.get('bot_id',None) is not None:
                        agent.opponent_id = job['bot_id']
                    else:
                        agent.opponent_id = self.job['env_info']['player_ids'][1-idx]
                    actor.agents.append(agent)
                    if player_id in self.job['update_players']:
                        self.tmp_model_path[player_id] = os.path.join(os.getcwd(), 'experiments',
                                                                      self._whole_cfg.common.experiment_name,
                                                                      'tmp/{}/model'.format(player_id))
                    if player_id in self.job['send_data_players']:
                        self.tmp_traj_path[player_id] = os.path.join(os.getcwd(), 'experiments',
                                                                     self._whole_cfg.common.experiment_name,
                                                                     'tmp/{}/traj'.format(player_id))
                    if agent.HAS_MODEL:
                        if player_id not in actor.models.keys():
                            agent.model = agent.model.eval().share_memory()
                            agent.player_id = player_id
                            if self._whole_cfg.actor.use_cuda:
                                agent.model = agent.model.cuda()
                            if not self._whole_cfg.actor.fake_model:
                                state_dict = torch.load(self.job['checkpoint_paths'][idx], map_location='cpu')
                                model_state_dict = {k: v for k, v in state_dict['model'].items() if
                                                    'value_networks' not in k}
                                agent.model.load_state_dict(model_state_dict,strict=False)
                            actor.models[player_id] = agent.model
                            if 'last_iter' in state_dict.keys():
                                last_iter = state_dict['last_iter']
                            else:
                                last_iter = 0
                            self.model_last_iter_dict[player_id] = torch.Tensor([last_iter]).share_memory_()
                        else:
                            agent.model = actor.models[player_id]
                for idx, successive_player_id in enumerate(self.job['successive_ids']):
                    if successive_player_id == 'none':
                        continue
                    agent = actor.agents[idx]
                    agent.successive_player_id = successive_player_id
                    if agent.HAS_SUCCESSIVE_MODEL:
                        if successive_player_id not in actor.successive_models.keys():
                            agent.successive_model = agent.successive_model.eval().share_memory()
                            if self._whole_cfg.actor.use_cuda:
                                agent.successive_model = agent.successive_model.cuda()
                            if not self._whole_cfg.actor.fake_model:
                                tmp_dir = os.path.abspath(os.path.join(os.getcwd(), 'experiments', self._whole_cfg.common.experiment_name,
                                'successive_model/{}/'.format(successive_player_id)))
                                tmp_files = os.listdir(tmp_dir)
                                tmp_path = os.path.join(tmp_dir, tmp_files[-1])
                                state_dict = torch.load(tmp_path, map_location='cpu')
                                model_state_dict = {k: v for k, v in state_dict['model'].items() if
                                                    'value_networks' not in k}
                                agent.successive_model.load_state_dict(model_state_dict,strict=False)
                        else:
                            agent.successive_model = agent.successive_models[successive_player_id]

                for idx, teacher_player_id in enumerate(self.job['teacher_player_ids']):
                    if teacher_player_id == 'none':
                        continue
                    agent = actor.agents[idx]
                    agent.teacher_player_id = teacher_player_id
                    if agent.HAS_TEACHER_MODEL:
                        if teacher_player_id not in teacher_models.keys():
                            agent.teacher_model = agent.teacher_model.eval().share_memory()
                            if self._whole_cfg.actor.use_cuda:
                                agent.teacher_model = agent.teacher_model.cuda()
                            if not self._whole_cfg.actor.fake_model:
                                state_dict = torch.load(self.job['teacher_checkpoint_paths'][idx],
                                                        map_location='cpu')
                                model_state_dict = {k: v for k, v in state_dict['model'].items() if
                                                    'value_networks' not in k}
                                agent.teacher_model.load_state_dict(model_state_dict,strict=False)
                            teacher_models[teacher_player_id] = agent.teacher_model
                        else:
                            agent.teacher_model = teacher_models[teacher_player_id]
                return
            else:
                time.sleep(3)

    def update_model(self, actor):
        torch.set_num_threads(1)
        last_reset_flag = {k: False for k in actor.models.keys()}
        if len(self.job['update_players']) == 0:
            return
        if self._last_update_time == -1 or time.time() - self._last_update_time > self._model_update_interval:
            reset_flag = False
            for player_id, model in actor.models.items():
                if player_id in self.job['update_players']:
                    start = time.time()

                    state_dict = self._adapter.pull(token=player_id + 'model', sleep_time=0.5, worker_num=1)

                    model_last_iter = torch.Tensor([state_dict['model_last_iter']])
                    self.model_last_iter_dict[player_id].copy_(model_last_iter)
                    model.load_state_dict(state_dict['model'])
                    self._update_model_time.append(time.time() - start)
                    avg_time = torch.Tensor([np.mean(self._update_model_time)])
                    self._avg_update_model_time.copy_(avg_time)
                    if not last_reset_flag[player_id]  and state_dict['reset_flag']:
                        reset_flag = True
                    last_reset_flag[player_id] = state_dict['reset_flag']
                self._last_update_time = time.time()
            if reset_flag:
                actor.reset_env()

    def async_update_model(self, actor):
        torch.set_num_threads(1)
        if not hasattr(self, '_update_model_loop'):
            self._model_ref = {}
            for player_id, model in actor.models.items():
                if player_id in self.job['update_players']:
                    self._model_ref[player_id] = {k: v.cpu().clone().share_memory_() for k, v in model.state_dict().items()}
            mp_context = tm.get_context('fork')
            self._model_signal_queue = mp_context.Queue()
            self._update_model_loop = mp_context.Process(target=update_model_loop, args=(
            self._whole_cfg, self._model_ref, self._model_signal_queue, self._model_update_interval,
            self.model_last_iter_dict, self._avg_update_model_time), daemon=True)
            self._update_model_loop.start()
        if self._model_signal_queue.qsize():
            reset_flag = self._model_signal_queue.get()
            for player_id, model in self._model_ref.items():
                actor.models[player_id].load_state_dict(model)
            if reset_flag:
                actor.reset_env()

    def send_data(self, data, player_id):
        if player_id not in self.job['send_data_players']:
            return
        length = self._adapter.length(player_id + 'traj')
        if length:
            print(f'actor stored data length: {length}, if this keeps showing, reduce actor number')

        self._adapter.push(data, token=player_id + 'traj', fs_type='nppickle', worker_num=self.worker_num)
        
    def send_result(self, result_info):
        while True:
            try:
                self._flask_send(result_info, 'league/actor_send_result', to_league=True)
                return
            except Exception as e:
                time.sleep(0.01)
                print('ERROR raised, not send result!!!!!!!!!!!!!!!!')

    def _flask_send(self, data, api, to_league=False):
        response = None
        t = time.time()
        try:
            if to_league:
                response = self._requests_session.post(self._league_url_prefix + api, json=data).json()
            else:
                response = self._requests_session.post(self._coordinator_url_prefix + api, json=data).json()
            name = self._actor_uid
            if response['code'] == 0:
                pass
                # self._logger.info("{} succeed sending result: {}, cost_time: {}".format(api, name, time.time() - t))
            else:
                self._logger.error("{} failed to send result: {}, cost_time: {}".format(api, name, time.time() - t))
        except Exception as e:
            self._logger.error(''.join(traceback.format_tb(e.__traceback__)))
            self._logger.error("[error] api({}): {}".format(api, sys.exc_info()))
        return response

    def close(self):
        if hasattr(self, '_update_model_loop'):
            self._update_model_loop.terminate()
            self._update_model_loop.join()
