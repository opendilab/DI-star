import copy
import os
import pickle
import random
from absl import logging
from collections import namedtuple
from typing import List, Any
import logging

import numpy as np
import time
import ctools.pysc2.env.sc2_env as sc2_env
from ctools.pysc2.env.sc2_env import SC2Env
from ctools.pysc2.env.sc2_eval_env import SC2EVALEnv
from ctools.pysc2.lib.action_dict import GENERAL_ACTION_INFO_MASK
from ctools.pysc2.lib.actions import FunctionCall
from .other.alphastar_map import get_map_size
from .action.alphastar_action_runner import AlphaStarRawActionRunner
from .reward.alphastar_reward_runner import AlphaStarRewardRunner
from .obs.alphastar_obs_runner import AlphaStarObsRunner
from .other.alphastar_statistics import RealTimeStatistics, GameLoopStatistics
from ctools.envs.env.base_env import BaseEnv
from ctools.utils import deep_merge_dicts, read_config, read_file

default_config = read_config(os.path.join(os.path.dirname(__file__), 'alphastar_env_default_config.yaml'))


class EvalEnv(BaseEnv, SC2EVALEnv):
    timestep = namedtuple('AlphaStarTimestep', ['obs', 'reward', 'battle_value', 'done',
                                                'info', 'episode_steps', 'due',
                                                'dists', 'units_num'])
    info_template = namedtuple('BaseEnvInfo', ['agent_num', 'obs_space', 'act_space', 'rew_space'])

    def __init__(self, cfg: dict) -> None:
        cfg = deep_merge_dicts(default_config.env, cfg)
        self._map_size = get_map_size(cfg.map_name, cropped=cfg.crop_map_to_playable_area)
        cfg.map_size = self._map_size
        cfg.obs_spatial.spatial_resolution = self._map_size
        cfg.action.map_size = self._map_size
        self._agent_num = 2 if cfg.game_type == 'agent_vs_agent' else 1
        cfg.agent_num = self._agent_num
        self._cfg = cfg
        self._local = self._cfg.local

        self._obs_helper = AlphaStarObsRunner(cfg)
        self._begin_num = self._obs_helper._obs_scalar.begin_num
        self._action_helper = AlphaStarRawActionRunner(cfg)
        self._reward_helper = AlphaStarRewardRunner(self._agent_num, cfg.pseudo_reward_type, cfg.pseudo_reward_prob)

        self._launch_env_flag = False
        if os.path.exists('./api-log'):
            logging.basicConfig(format='%(process)d - %(asctime)s - %(levelname)s: %(message)s',
                                filename='./api-log/actor_error.log',
                                filemode='a',
                                level=logging.ERROR)
        self.logger = logging.getLogger()


    def _get_players(self, cfg):
        if cfg.game_type == 'agent_vs_bot':
            players = [
                sc2_env.Agent(sc2_env.Race[cfg.player1.race], cfg.player1.get('name', 'unknown')),
                sc2_env.Bot(
                    sc2_env.Race[cfg.player2.race], sc2_env.Difficulty[cfg.player2.difficulty],
                    sc2_env.BotBuild[cfg.player2.build]
                )
            ]
        elif cfg.game_type == 'agent_vs_agent' or cfg.game_type == 'human_vs_agent':
            players = [sc2_env.Agent(sc2_env.Race[cfg.player1.race], cfg.player1.get('name', 'unknown')), sc2_env.Agent(sc2_env.Race[cfg.player2.race], cfg.player2.get('name', 'unknown'))]
        else:
            raise KeyError("invalid game_type: {}".format(cfg.game_type))
        return players

    def _launch_env(self) -> None:
        cfg = self._cfg
        self._players = self._get_players(cfg)
        agent_interface_format = sc2_env.parse_agent_interface_format(
            feature_screen=cfg.screen_resolution,
            feature_minimap=self._map_size,  # x, y
            crop_to_playable_area=cfg.crop_map_to_playable_area,
            raw_crop_to_playable_area=cfg.crop_map_to_playable_area,
            action_delays=cfg.action_delays
        )

        SC2EVALEnv.__init__(
            self,
            map_name=cfg.map_name,
            random_seed=cfg.random_seed,
            step_mul=cfg.default_step_mul,
            players=self._players,
            game_steps_per_episode=cfg.game_steps_per_episode,
            agent_interface_format=agent_interface_format,
            save_replay_episodes=cfg.get('save_replay_episodes', 0),
            replay_dir=cfg.get('replay_dir', None),
            disable_fog=cfg.disable_fog,
            score_index=-1,  # use win/loss reward rather than score
            ensure_available_actions=False,
            realtime=cfg.realtime,
            game_type=cfg.game_type,
        )

    def _raw_env_reset(self, agent_names=None):
        players = None
        if agent_names:
            players = [sc2_env.Agent(sc2_env.Race[self._cfg.player1.race], agent_names[0]),
                       sc2_env.Agent(sc2_env.Race[self._cfg.player2.race], agent_names[1])]
        return SC2EVALEnv.reset(self, players)

    def _raw_env_step(self, raw_action, step_mul):
        return SC2EVALEnv.step(self, raw_action, step_mul=step_mul)

    def load_stat(self, timesteps):
        self._loaded_eval_stat = []
        for agent_no in range(self._agent_num):
            if self._local:
                path = os.path.join(os.path.dirname(__file__), '../data/Z/', 'stat_hand_filter.local')
                f = open(path, 'rb')
                all_stats = pickle.load(f)
                f.close()
                opponent_born_location = timesteps[agent_no].game_info.start_raw.start_locations
                assert len(opponent_born_location) == 1, 'only one opponent born location!'
                for idx, stats in enumerate(all_stats[self._cfg.map_name]):
                    if stats[0][0] == opponent_born_location[0].x and stats[0][1] == opponent_born_location[0].y:
                        p = stats[random.randint(1, len(stats) - 1)]
                        f = open(os.path.join(os.path.dirname(__file__), '../data/Z/', p), 'rb')
                        stat = pickle.load(f)
                        # stat_info = '*************STAT INFO*************\n'
                        # stat_info += 'Map: {} \n'.format(self._cfg.map_name)
                        # stat_info += 'born location: {}\n'.format(stat['born_location'])
                        # units = list(stat['cumulative_stat'][-1].keys())
                        # units.remove('game_loop')
                        # stat_info += 'Built units: \n'
                        # for i in units:
                        #     stat_info += '   {}\n'.format(GENERAL_ACTION_INFO_MASK[i]['name'])
                        # stat_info += 'Building Order: \n'
                        # for i in stat['beginning_build_order']:
                        #     stat_info += '   {}\n'.format(GENERAL_ACTION_INFO_MASK[i['action_type']]['name'])
                        # print(stat_info)
                        f.close()
                        break
            else:
                path = os.path.join(os.path.dirname(__file__), '../data/Z/', 'stat_hand_filter')
                f = open(path, 'rb')
                all_stats = pickle.load(f)
                f.close()
                opponent_born_location = timesteps[agent_no].game_info.start_raw.start_locations
                assert len(opponent_born_location) == 1, 'only one opponent born location!'
                for idx, stats in enumerate(all_stats[self._cfg.map_name]):
                    if stats[0][0] == opponent_born_location[0].x and stats[0][1] == opponent_born_location[0].y:
                        p = stats[random.randint(1, len(stats) - 1)] + '.z'
                        stat = read_file(p, fs_type='ceph')
                        break
            self._loaded_eval_stat.append(GameLoopStatistics(stat, self._begin_num))

    def reset(self, agent_names=None):
        self._launch_env()
        self._reward_helper.reset()
        self._obs_helper.reset()
        self._action_helper.reset()
        self._episode_stat = [RealTimeStatistics(self._begin_num) for _ in range(self._agent_num)]
        self._next_obs_step = [0 for _ in range(self._agent_num)]
        self.due = [True for _ in range(self._agent_num)]
        self.action = [None] * self._agent_num
        timestep = self._raw_env_reset(agent_names)
        self.load_stat(timestep)
        self._raw_obs = [timestep[n].observation for n in range(self._agent_num)]
        obs = self._obs_helper.get(self)
        self._last_obs = obs
        info = [t.game_info for t in timestep]
        env_provided_map_size = info[0].start_raw.map_size
        env_provided_map_size = [env_provided_map_size.x, env_provided_map_size.y]
        assert tuple(env_provided_map_size) == tuple(self._map_size), \
            "Environment uses a different map size {} compared to config " \
            "{}.".format(env_provided_map_size, self._map_size)
        # Note: self._episode_steps is updated in SC2Env
        self._episode_steps = 0
        self._launch_env_flag = True
        return obs

    def step(self, action: list) -> 'EvalEnv.timestep':
        """
        Note:
            delay: delay is the relative steps between two observations of a agent
            step_mul: step_mul is the relative steps that the env executes in the next step operation
            episode_steps: episode_steps is the current absolute steps that the env has finished
            _next_obs_step: _next_obs_step is the absolute steps what a agent gets its observation
        """
        action_data = copy.deepcopy(action)
        assert self._launch_env_flag
        # get transformed action and delay
        self.agent_action = action_data
        # save original locat
        locations = [action_data[i]['action']['target_location'] for i in range(self._agent_num)]

        raw_action, delay, action = self._action_helper.get(self)
        raw_action = list(raw_action)
        for i, r_act in enumerate(raw_action):
            if r_act.function == 1:
                location = r_act.arguments[-1]
                raw_action[i] = [r_act, FunctionCall(168, [location])]
        prev_due = copy.deepcopy(self.due)
        # get step_mul
        for n in range(self._agent_num):
            if self.due[n]:
                self._next_obs_step[n] = self.episode_steps + delay[n]
                self.action[n] = action[n]
        self._obs_helper.update_last_action(self)
        step_mul = min(self._next_obs_step) - self.episode_steps
        # TODO(nyz) deal with step == 0 case for stat and reward
        if step_mul <= 0:
            step_mul = 1
        self.due = [s <= self.episode_steps + step_mul for s in self._next_obs_step]
        assert any(self.due), 'at least one of the agents must finish its delay'

        # env step
        last_episode_steps = self.episode_steps
        timestep, results = self._raw_env_step(raw_action, step_mul)  # update episode_steps

        for n in range(self._agent_num):
            if prev_due[n]:
                if results[n]:
                    self._episode_stat[n].update_stat(action[n], None, self.episode_steps, locations[n])

        # transform obs, reward and record statistics
        self.raw_obs = [timestep[n].observation for n in range(self._agent_num)]
        obs = self._obs_helper.get(self)
        self.reward = [timestep[n].reward for n in range(self._agent_num)]
        info = [timestep[n].game_info for n in range(self._agent_num)]
        done = any([timestep[n].last() for n in range(self._agent_num)])
        # Note: pseudo reward must be derived after statistics update
        if self._agent_num > 1:
            self.reward, self.dists = self._reward_helper.get(self)
            units_num = []
            for n in range(self._agent_num):
                ut_num = self._episode_stat[n].get_norm_units_num()
                units_num.append(ut_num)
            self.units_num = units_num
        else:
            self.dists = None
            self.units_num = None
        # update last state variable
        self._last_obs = obs


        return EvalEnv.timestep(
            obs=obs,
            reward=self.reward,
            battle_value=self._get_battle_value(self.raw_obs),
            done=done,
            info=info,
            episode_steps=[int(last_episode_steps) for _ in range(self._agent_num)],
            due=copy.deepcopy(self.due),
            dists=self.dists,
            units_num=self.units_num
        )

    def seed(self, seed: int) -> None:
        """Note: because SC2Env sets up the random seed in input args, we don't implement this method"""
        raise NotImplementedError()

    def info(self) -> 'EvalEnv.info':
        info_data = {
            'agent_num': self._agent_num,
            'obs_space': self._obs_helper.info,
            'act_space': self._action_helper.info,
            'rew_space': self._reward_helper.info,
        }
        return EvalEnv.info_template(**info_data)

    def __repr__(self) -> str:
        return 'AlphaStarEnv:\n\
                \tobservation[{}]\n\
                \taction[{}]\n\
                \treward[{}]\n'.format(repr(self._obs_helper), repr(self._action_helper), repr(self._reward_helper))

    def close(self) -> None:
        SC2EVALEnv.close(self)

    def _get_battle_value(self, raw_obs):
        minerals_ratio = 1.
        vespene_ratio = 1.
        return [
            int(
                np.sum(obs['score_by_category']['killed_minerals']) * minerals_ratio +
                np.sum(obs['score_by_category']['killed_vespene'] * vespene_ratio)
            ) for obs in raw_obs
        ]

    @property
    def episode_stat(self) -> RealTimeStatistics:
        return self._episode_stat

    @property
    def episode_steps(self) -> int:
        return self._episode_steps

    @property
    def loaded_eval_stat(self) -> GameLoopStatistics:
        return self._loaded_eval_stat

    @property
    def action(self) -> namedtuple:
        return self._action

    @action.setter
    def action(self, _action: namedtuple) -> None:
        self._action = _action

    @property
    def dists(self) -> list:
        return self._dists

    @dists.setter
    def dists(self, _dists: list) -> None:
        self._dists = _dists

    @property
    def units_num(self) -> list:
        return self._units_num

    @units_num.setter
    def units_num(self, _units_num: list) -> None:
        self._units_num = _units_num

    @property
    def reward(self) -> list:
        return self._reward

    @reward.setter
    def reward(self, _reward: list) -> None:
        self._reward = _reward

    @property
    def raw_obs(self) -> list:
        return self._raw_obs

    @raw_obs.setter
    def raw_obs(self, _raw_obs) -> None:
        self._raw_obs = _raw_obs

    @property
    def agent_action(self) -> list:
        return self._agent_action

    @agent_action.setter
    def agent_action(self, _agent_action) -> None:
        self._agent_action = _agent_action


AlphaStarTimestep = EvalEnv.timestep


class FakeAlphaStarEnv(object):
    timestep = namedtuple('FakeAlphaStarTimestep', ['obs', 'reward', 'done', 'info', 'episode_steps', 'due'])

    def __init__(self, *args, **kwargs):
        self.fake_data = np.load(os.path.join(os.path.dirname(__file__), 'fake_raw_env_data.npy'), allow_pickle=True)
        self.count = 0

    def reset(self):
        idx = np.random.choice(range(len(self.fake_data)))
        self.count = 0
        return self.fake_data[idx][0]

    def step(self, action):
        assert isinstance(action, list)
        idx = np.random.choice(range(len(self.fake_data)))
        data = copy.deepcopy(self.fake_data[idx])
        episode_steps = data[4]
        data[4] = [episode_steps for _ in range(2)]
        if self.count < 16:
            data[2] = False
        self.count += 1
        return FakeAlphaStarEnv.timestep(*data)

    def close(self):
        pass
