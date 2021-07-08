import os
import time
import copy
import requests
import traceback
import sys
import uuid
import random
import traceback
import argparse
from easydict import EasyDict
from absl import app
from collections import namedtuple
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ctools.worker.actor import BaseEnvManager, SubprocessEnvManager
from ctools.torch_utils import to_device, tensor_to_list
from distar.model import AlphaStarActorCritic
from distar.worker.agent.alphastar_agent import create_as_actor_agent
from ctools.utils import get_data_compressor, lists_to_dicts, get_task_uid
from distar.envs import AlphaStarEnv, FakeAlphaStarEnv, EvalEnv
from distar.data.collate_fn import as_eval_collate_fn
from collections import OrderedDict
from ctools.utils import deep_merge_dicts, read_config, read_file
import torch


short = {'winloss': 'winloss', 'build_order': 'bo', 'built_unit': 'bu', 'effect': 'effect', 'upgrade': 'upgrade',
         'battle': 'battle', 'value': 'v', 'reward': 'r'}

default_config = read_config(os.path.join(os.path.dirname(__file__), "eval.yaml"))


class ASEvalActor:

    def __init__(self, model1=None, model2=None, cuda=None, game_type=None):

        self._result_map = {1: 'wins', 0: 'draws', -1: 'losses'}
        self._job_result = []
        self._units_num = []
        self._cfg = default_config.actor
        if model1:
            self._cfg.model_path[0] = model1
            self._cfg.model_path[1] = model2
            self._cfg.use_cuda = cuda
            self._cfg.env.game_type = game_type
        if game_type == 'human_vs_agent':
            self._cfg.env.player2.name = 'humanPlayer'
            self._cfg.env.player1.name = os.path.basename(model1.split('_')[0].upper()) + 'Agent'
        elif game_type == 'agent_vs_agent':
            self._cfg.env.player1.name = os.path.basename(model1.split('_')[0].upper()) + 'Agent'
            self._cfg.env.player2.name = os.path.basename(model2.split('_')[0].upper()) + 'Agent'
        elif game_type == 'agent_vs_bot':
            self._cfg.env.player1.name = os.path.basename(model1.split('_')[0].upper()) + 'Agent'
            self._cfg.env.player2.name = ''
        else:
            raise NotImplementedError('game type must choose from human_vs_agent agent_vs_agent agent_vs_bot, not: {}'.format(game_type))
        self._setup_agents()
        self._setup_env_manager()


    def _setup_env_manager(self) -> None:
        if self._cfg.env.map_name == 'random':
            self._cfg.env.map_name = random.choice(['KingsCove', 'NewRepugnancy', 'CyberForest', 'KairosJunction'])
        env_cfg = self._cfg.env
        self.env_num = 1
        env_manager = BaseEnvManager
        env_cfgs = []
        for _ in range(self.env_num):
            cfg = copy.deepcopy(env_cfg)
            cfg.random_seed = random.randint(1, 1000034)
            env_cfgs.append(cfg)
        env_type = self._cfg.env.get('env_type', 'AlphaStar')
        Env = EvalEnv
        self._env_manager = env_manager(
            env_fn=Env, env_cfg=env_cfgs, env_num=self.env_num,
            episode_num=1, player_num=self._agent_num, map_name=self._cfg.env.map_name
        )
        self._env_manager.launch()

    def _setup_agents(self):
        if self._cfg.env.game_type == 'agent_vs_agent':
            self._agent_num = 2
        else:
            self._agent_num = 1
        self._agent = []
        for i in range(self._agent_num):
            model = AlphaStarActorCritic()
            if self._cfg.use_cuda:
                model.cuda()
            agent = create_as_actor_agent(model, 1, use_teacher=False)
            agent.mode(False)
            agent.reset()
            # load model
            agent = self._load_model(agent, self._cfg.model_path[i])
            self._agent.append(agent)
        self._valid_obs_flag = {k: [True for _ in range(self._agent_num)] for k in range(1)}

    def _load_model(self, agent, model_path):
        state_dict = torch.load(model_path, map_location='cpu')
        actor_state_dict = state_dict['model']
        actor_state_dict = OrderedDict({k: v for k, v in actor_state_dict.items() if 'value_networks' not in k})
        agent._model.load_state_dict(actor_state_dict, strict=False)
        agent.reset()
        return agent

    # override
    def _agent_inference(self, obs):
        data = [None for _ in range(self._agent_num)]
        state_id = obs.keys()
        obs = as_eval_collate_fn(list(obs.values()))
        valid_id = [[env_idx for env_idx in range(self.env_num) if self._valid_obs_flag[env_idx][agent_idx]] for agent_idx in range(self._agent_num)]

        if self._cfg.use_cuda:
            obs = [to_device(o, 'cuda') for o in obs]
        for agent_obs_idx, agent in enumerate(self._agent):
            data[agent_obs_idx] = agent.forward(obs[agent_obs_idx], state_id=state_id, valid_id=valid_id[agent_obs_idx])
        data = lists_to_dicts(data)
        tmp_action = list(zip(*data['action']))
        if self._cfg.use_cuda:
            tmp_action = to_device(tmp_action, 'cpu')
        action = {}
        for i, a in zip(state_id, tmp_action):
            action[i] = a
        return action, data

    def run(self) -> None:
        step_ = 1
        while True:
            obs = self._env_manager.next_obs
            t = time.time()
            action, data = self._agent_inference(obs)
            inference_time = time.time() - t
            timestep = self._env_manager.step(action)
            self._process_timestep(timestep)
            print('step: {}, inference_time: {}'.format(step_, inference_time))
            step_ += 1
            if self._env_manager.done:
                break

    # override
    def _process_timestep(self, timestep: namedtuple) -> None:
        for env_id, t in timestep.items():
            for agent_idx in range(self._agent_num):
                self._valid_obs_flag[env_id][agent_idx] = timestep[env_id].due[agent_idx]
                if timestep[env_id].done:
                    self._agent[agent_idx].reset(state_id=[env_id])
                    self._valid_obs_flag[env_id] = [True for _ in range(self._agent_num)]
