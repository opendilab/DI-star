import copy
import json
import os
import random
import time

import torch

import copy
import json
import os
import random
import torch

from copy import deepcopy
from collections import deque, defaultdict
from functools import partial
from torch.utils.data._utils.collate import default_collate

from .model.model import Model
from .lib.actions import NUM_CUMULATIVE_STAT_ACTIONS, ACTIONS, BEGINNING_ORDER_ACTIONS, CUMULATIVE_STAT_ACTIONS, UNIT_ABILITY_TO_ACTION, QUEUE_ACTIONS, UNIT_TO_CUM, UPGRADE_TO_CUM
from .lib.features import Features, SPATIAL_SIZE, BEGINNING_ORDER_ACTIONS, CUMULATIVE_STAT_ACTIONS, BEGINNING_ORDER_LENGTH, ScoreCategories, compute_battle_score, fake_step_data, fake_model_output
from .lib.stat import Stat, cum_dict
from distar.ctools.torch_utils.metric import levenshtein_distance, hamming_distance, l2_distance
from distar.pysc2.lib.units import get_unit_type
from distar.pysc2.lib.static_data import UNIT_TYPES, NUM_UNIT_TYPES
from distar.ctools.torch_utils import to_device

RACE_DICT = {
    1: 'terran',
    2: 'zerg',
    3: 'protoss',
    4: 'random',
}


def copy_input_data(shared_step_data, step_data, data_idx):
    entity_num = step_data['entity_num']
    if 'selected_units_num' in step_data.keys():
        selected_units_num = step_data['selected_units_num']
    else:
        selected_units_num = 0
    for k, v in step_data.items():
        if k == 'hidden_state':
            for i in range(len(v)):
                shared_step_data['hidden_state'][i][0][data_idx].copy_(v[i][0])
                shared_step_data['hidden_state'][i][1][data_idx].copy_(v[i][1])
        elif k == 'value_feature':
            pass
        elif isinstance(v, torch.Tensor):
            shared_step_data[k][data_idx].copy_(step_data[k])
        elif isinstance(v, dict):
            for _k, _v in v.items():
                if k == 'action_info' and _k == 'selected_units':
                    if selected_units_num > 0:
                        shared_step_data[k][_k][data_idx, :selected_units_num].copy_(step_data[k][_k])
                elif k == 'entity_info':
                    shared_step_data[k][_k][data_idx, :entity_num].copy_(step_data[k][_k])
                elif k == 'spatial_info':
                    if 'effect' in _k:
                        shared_step_data[k][_k][data_idx].copy_(step_data[k][_k])
                    else:
                        h, w = step_data[k][_k].shape
                        shared_step_data[k][_k][data_idx] *= 0
                        shared_step_data[k][_k][data_idx, :h, :w].copy_(step_data[k][_k])
                else:
                    shared_step_data[k][_k][data_idx].copy_(step_data[k][_k])


def copy_output_data(shared_step_data, step_data, data_indexes):
    data_indexes = data_indexes.nonzero().squeeze(dim=1)
    for k, v in step_data.items():
        if k == 'hidden_state':
            for i in range(len(v)):
                shared_step_data['hidden_state'][i][0].index_copy_(0, data_indexes, v[i][0][data_indexes].cpu())
                shared_step_data['hidden_state'][i][1].index_copy_(0, data_indexes, v[i][1][data_indexes].cpu())
        elif isinstance(v, dict):
            for _k, _v in v.items():
                if len(_v.shape) == 3:
                    _, s1, s2 = _v.shape
                    #shared_step_data[k][_k][:, :s1, :s2][data_indexes].copy_(_v[data_indexes])
                    shared_step_data[k][_k][:, :s1, :s2].index_copy_(0, data_indexes, _v[data_indexes].cpu())
                elif len(_v.shape) == 2:
                    _, s1 = _v.shape
                    shared_step_data[k][_k][:, :s1].index_copy_(0, data_indexes, _v[data_indexes].cpu())
                elif len(_v.shape) == 1:
                    shared_step_data[k][_k].index_copy_(0, data_indexes, _v[data_indexes].cpu())
        elif isinstance(v, torch.Tensor):
            shared_step_data[k].index_copy_(0, data_indexes, v[data_indexes].cpu())


class Agent:
    HAS_MODEL = True
    HAS_TEACHER_MODEL = True
    HAS_SUCCESSIVE_MODEL = False

    def __init__(self, cfg=None, env_id=0):
        self._whole_cfg = cfg
        self._job_type = cfg.actor.job_type
        self._only_cum_action_kl = self._whole_cfg.get('learner', {}).get('only_cum_action_kl',False)
        self._z_path = self._whole_cfg.agent.z_path
        self._bo_norm = self._whole_cfg.get('learner', {}).get('bo_norm',20)
        self._cum_norm = self._whole_cfg.get('learner', {}).get('cum_norm',30)
        self._battle_norm = self._whole_cfg.get('learner', {}).get('battle_norm',30)
        self.model = Model(cfg)
        self._player_id = None
        self._num_layers = self.model.cfg.encoder.core_lstm.num_layers
        self._hidden_size = self.model.cfg.encoder.core_lstm.hidden_size
        self._zero_z_value = self._whole_cfg.get('feature', {}).get('zero_z_value', 1.)
        self._zero_z_exceed_loop = self._whole_cfg.agent.get('zero_z_exceed_loop', False)
        self._extra_units = self._whole_cfg.agent.get('extra_units', False)
        self._bo_zergling_num = self._whole_cfg.agent.get('bo_zergling_num', 8)
        self._fake_reward_prob = self._whole_cfg.agent.get('fake_reward_prob', 1.)
        self._use_value_feature = self._whole_cfg.get('learner', {}).get('use_value_feature',False)
        self._clip_bo = self._whole_cfg.agent.get('clip_bo', True)
        self._cum_type = self._whole_cfg.agent.get('cum_type', 'action')  # observation or action
        self._env_id = env_id
        self._gpu_batch_inference = self._whole_cfg.actor.get('gpu_batch_inference', False)
        self.z_idx = None
        if self._whole_cfg.env.realtime:
            data = fake_step_data(share_memory=True, batch_size=1, hidden_size=self._hidden_size,
                                               hidden_layer=self._num_layers, train=False)
            if self._whole_cfg.actor.use_cuda:
                data = to_device(data, torch.cuda.current_device())
                self.model = self.model.cuda()
            with torch.no_grad():
                _ = self.model.compute_logp_action(**data)
        if self._gpu_batch_inference:
            batch_size = self._whole_cfg.actor.env_num
            self._shared_input = fake_step_data(share_memory=True, batch_size=batch_size, hidden_size=self._hidden_size,
                                               hidden_layer=self._num_layers, train=False)
            self._shared_output = fake_model_output(batch_size=batch_size, hidden_size=self._hidden_size,
                                             hidden_layer=self._num_layers, teacher=False)
            self._signals = torch.zeros(batch_size).share_memory_()
            if 'train' in self._job_type:
                self._teacher_shared_input = fake_step_data(share_memory=True, batch_size=batch_size,
                                                           hidden_size=self._hidden_size,
                                                           hidden_layer=self._num_layers, train=True)
                self._teacher_shared_output = fake_model_output(batch_size=batch_size, hidden_size=self._hidden_size,
                                                 hidden_layer=self._num_layers, teacher=True)
                self._teacher_signals = torch.zeros(batch_size).share_memory_()
        if 'train' in self._job_type:
            self.teacher_model = Model(cfg)

    def reset(self, map_name, race, game_info, obs):
        self._stat_api = Stat(race)
        self._race = race
        self.model.policy.action_type_head.race = race
        self._map_name = map_name
        self._hidden_state = [(torch.zeros(self._hidden_size), torch.zeros(self._hidden_size)) for _ in range(self._num_layers)]
        self._last_action_type = torch.tensor(0, dtype=torch.long)
        self._last_delay = torch.tensor(0, dtype=torch.long)
        self._last_queued = torch.tensor(0, dtype=torch.long)
        self._last_selected_unit_tags = None
        self._last_target_unit_tag = None
        self._last_location = None  # [x, y]
        self._enemy_unit_type_bool = torch.zeros(NUM_UNIT_TYPES, dtype=torch.uint8)
        self._observation = None
        self._output = None
        self._iter_count = 0
        self._model_last_iter = 0
        self._game_step = 0  # step * 10 is game duration time
        self._behaviour_building_order = [] # idx in BEGINNING_ORDER_ACTIONS
        self._behaviour_bo_location = []
        self._bo_zergling_count = 0
        self._behaviour_cumulative_stat = [0] * NUM_CUMULATIVE_STAT_ACTIONS
        self._feature = Features(game_info, obs['raw_obs'], self._whole_cfg)
        self._exceed_flag = True

        if 'train' in self._job_type:
            self._hidden_state_backup = [(torch.zeros(self._hidden_size), torch.zeros(self._hidden_size)) for _ in range(self._num_layers)]
            self._teacher_hidden_state = [(torch.zeros(self._hidden_size), torch.zeros(self._hidden_size)) for _ in range(self._num_layers)]
            self._data_buffer = deque(maxlen=self._whole_cfg.actor.traj_len)
            self._push_count = 0

        # init Z
        raw_ob = obs['raw_obs']
        location = []
        for i in raw_ob.observation.raw_data.units:
            if i.unit_type == 59 or i.unit_type == 18 or i.unit_type == 86:
                location.append([i.pos.x, i.pos.y])
        assert len(location) == 1, 'no fog of war, check game version!'
        self._born_location = deepcopy(location[0])
        born_location = location[0]
        born_location[0] = int(born_location[0])
        born_location[1] = int(self._feature.map_size.y - born_location[1])
        born_location_str = str(born_location[0] + born_location[1] * 160)
        self._z_path = os.path.join(os.path.dirname(__file__), 'lib', self._z_path)
        with open(self._z_path, 'r') as f:
            self._z_data = json.load(f)
            z_data = self._z_data
        z_type = None
        idx = None
        raw_ob = obs['raw_obs']
        race = RACE_DICT[self._feature.requested_races[raw_ob.observation.player_common.player_id]]
        opponent_id = 1 if raw_ob.observation.player_common.player_id == 2 else 2
        opponent_race = RACE_DICT[self._feature.requested_races[opponent_id]]
        if race == opponent_race:
            mix_race = race
        else:
            mix_race = race + opponent_race
        if self.z_idx is not None:
            idx, z_type = random.choice(self.z_idx[self._map_name][mix_race][born_location_str])
            z = z_data[self._map_name][mix_race][born_location_str][idx]
        else:
            z = random.choice(z_data[self._map_name][mix_race][born_location_str])
        if len(z) == 5:
            self._target_building_order, target_cumulative_stat, bo_location, self._target_z_loop, z_type = z
        else:
            self._target_building_order, target_cumulative_stat, bo_location, self._target_z_loop = z
        self.use_cum_reward = True
        self.use_bo_reward = True
        if z_type is not None:
            if z_type == 2 or z_type == 3:
                self.use_cum_reward = False
            if z_type == 1 or z_type == 3:
                self.use_bo_reward = False
        if random.random() > self._fake_reward_prob:
            self.use_cum_reward = False
        if random.random() > self._fake_reward_prob:
            self.use_bo_reward = False
        print('z_type', z_type, 'cum', self.use_cum_reward, 'bo', self.use_bo_reward)

        if self._whole_cfg.agent.get('show_Z', False):
            s = 'Map: {} Race: {}, Born location: ({}, {}), loop: {}, idx: {}\n'.format(map_name, mix_race, born_location[0], born_location[1], self._target_z_loop, idx)
            s += 'Building order:\n'
            for idx in range(len(self._target_building_order)):
                a = self._target_building_order[idx]
                if a != 0:
                    action_type = BEGINNING_ORDER_ACTIONS[a]
                    x, y = bo_location[idx] % 160, bo_location[idx] // 160
                    s += '  {}, ({}, {})\n'.format(ACTIONS[action_type]['name'], x, y)
            s += 'Cumulative stat:\n'
            for i in target_cumulative_stat:
                action_type = CUMULATIVE_STAT_ACTIONS[i]
                s += '  {}\n'.format(ACTIONS[action_type]['name'])
            print(s)
        self._bo_norm = len(self._target_building_order)
        self._cum_norm = len(target_cumulative_stat)
        self._target_bo_location = torch.tensor(bo_location, dtype=torch.long)
        self._target_building_order = torch.tensor(self._target_building_order, dtype=torch.long)
        self._target_cumulative_stat = torch.zeros(NUM_CUMULATIVE_STAT_ACTIONS, dtype=torch.float)
        self._target_cumulative_stat.scatter_(index=torch.tensor(target_cumulative_stat, dtype=torch.long), dim=0, value=1.)
        if not self._whole_cfg.env.realtime:
            if not self._clip_bo:
                self._old_bo_reward = -levenshtein_distance(
                                torch.as_tensor(self._behaviour_building_order, dtype=torch.long),
                                self._target_building_order) /self._bo_norm
            else:
                self._old_bo_reward = torch.tensor(0.)
            self._old_cum_reward = -hamming_distance(torch.as_tensor(self._behaviour_cumulative_stat, dtype=torch.float),
                                                       self._target_cumulative_stat)/ self._cum_norm
            self._total_bo_reward = torch.zeros(size=(), dtype=torch.float)
            self._total_cum_reward = torch.zeros(size=(), dtype=torch.float)


    def _pre_process(self, obs):
        if self._use_value_feature:
            agent_obs = self._feature.transform_obs(obs['raw_obs'], padding_spatial=True, opponent_obs=obs['opponent_obs'])
        else:
            agent_obs = self._feature.transform_obs(obs['raw_obs'], padding_spatial=True)
        self._game_info = agent_obs.pop('game_info')
        self._game_step = self._game_info['game_loop']
        if self._zero_z_exceed_loop and self._game_step > self._target_z_loop:
            self._exceed_flag = False
            self._target_z_loop = 99999999

        last_selected_units = torch.zeros(agent_obs['entity_num'], dtype=torch.int8)
        last_targeted_unit = torch.zeros(agent_obs['entity_num'], dtype=torch.int8)
        tags = self._game_info['tags']
        if self._last_selected_unit_tags is not None:
            for t in self._last_selected_unit_tags:
                if t in tags:
                    last_selected_units[tags.index(t)] = 1
        if self._last_target_unit_tag is not None:
            if self._last_target_unit_tag in tags:
                last_targeted_unit[tags.index(self._last_target_unit_tag)] = 1
        agent_obs['entity_info']['last_selected_units'] = last_selected_units
        agent_obs['entity_info']['last_targeted_unit'] = last_targeted_unit

        agent_obs['hidden_state'] = self._hidden_state
        agent_obs['scalar_info']['last_delay'] = self._last_delay
        agent_obs['scalar_info']['last_action_type'] = self._last_action_type
        agent_obs['scalar_info']['last_queued'] = self._last_queued

        agent_obs['scalar_info']['enemy_unit_type_bool'] = (self._enemy_unit_type_bool | agent_obs['scalar_info']['enemy_unit_type_bool']).to(torch.uint8)

        agent_obs['scalar_info']['beginning_order'] = self._target_building_order * (self.use_bo_reward & self._exceed_flag)
        agent_obs['scalar_info']['bo_location'] = self._target_bo_location * (self.use_bo_reward & self._exceed_flag)
        if self.use_cum_reward and self._exceed_flag:
            agent_obs['scalar_info']['cumulative_stat'] = self._target_cumulative_stat
        else:
            agent_obs['scalar_info']['cumulative_stat'] = self._target_cumulative_stat * 0 + self._zero_z_value

        self._observation = agent_obs
        if self._whole_cfg.actor.use_cuda:
            agent_obs = to_device(agent_obs, 'cuda:0')
        if self._gpu_batch_inference:
            copy_input_data(self._shared_input, agent_obs, data_idx=self._env_id)
            self._signals[self._env_id] += 1
            model_input = None
        else:
            model_input = default_collate([agent_obs])
        return model_input

    def step(self, observation):
        if 'eval' in self._job_type and self._iter_count > 0 and not self._whole_cfg.env.realtime:
            self._update_fake_reward(self._last_action_type, self._last_location, observation)
        model_input = self._pre_process(observation)
        self._stat_api.update(self._last_action_type, observation['action_result'][0], self._observation, self._game_step)
        if not self._gpu_batch_inference:
            model_output = self.model.compute_logp_action(**model_input)
        else:
            while True:
                if self._signals[self._env_id] == 0:
                    model_output = self._shared_output
                    break
                else:
                    time.sleep(0.01)
        action = self._post_process(model_output)
        self._iter_count += 1
        return action

    def decollate_output(self, output, k=None, batch_idx=None):
        if isinstance(output, torch.Tensor):
            if batch_idx is None:
                return output.squeeze(dim=0)
            else:
                return output[batch_idx].clone().cpu()
        elif k == 'hidden_state':
            if batch_idx is None:
                return [(output[l][0].squeeze(dim=0), output[l][1].squeeze(dim=0)) for l in range(len(output))]
            else:
                return [(output[l][0][batch_idx].clone().cpu(), output[l][1][batch_idx].clone().cpu()) for l in range(len(output))]
        elif isinstance(output, dict):
            data = {k: self.decollate_output(v, k, batch_idx) for k, v in output.items()}
            if batch_idx is not None and k is None:
                entity_num = data['entity_num']
                selected_units_num = data['selected_units_num']
                data['logit']['selected_units'] = data['logit']['selected_units'][:selected_units_num, :entity_num + 1]
                data['logit']['target_unit'] = data['logit']['target_unit'][:entity_num]
                if 'action_info' in data.keys():
                    data['action_info']['selected_units'] = data['action_info']['selected_units'][:selected_units_num]
                    data['action_logp']['selected_units'] = data['action_logp']['selected_units'][:selected_units_num]
            return data

    def _post_process(self, output):
        if self._gpu_batch_inference:
            output = self.decollate_output(output, batch_idx=self._env_id)
        else:
            output = self.decollate_output(output)

        self._hidden_state = output['hidden_state']
        self._last_queued = output['action_info']['queued']
        self._last_action_type = output['action_info']['action_type']
        self._last_delay = output['action_info']['delay']
        self._last_location = output['action_info']['target_location']
        self._output = output

        # action_info = {'func_id': 0, 'skip_steps': 0, 'queued': 0, 'unit_tags': [0, 1], 'target_unit_tag': 0,
        #                'location': [0, 0]}
        action_info = {}
        action_info['func_id'] = ACTIONS[output['action_info']['action_type'].item()]['func_id']
        action_info['skip_steps'] = output['action_info']['delay'].item()
        action_info['queued'] = output['action_info']['queued'].item()
        action_info['unit_tags'] = []
        for i in range(output['selected_units_num'] - 1):
            try:
                action_info['unit_tags'].append(self._game_info['tags'][output['action_info']['selected_units'][i].item()])
            except:
                print()
        if self._extra_units:
            extra_units = torch.nonzero(output['extra_units']).squeeze(dim=1).tolist()
            for unit_index in extra_units:
                action_info['unit_tags'].append(
                    self._game_info['tags'][unit_index])

        if ACTIONS[output['action_info']['action_type'].item()]['selected_units']:
            self._last_selected_unit_tags = action_info['unit_tags']
        else:
            self._last_selected_unit_tags = None
        action_info['target_unit_tag'] = self._game_info['tags'][output['action_info']['target_unit'].item()]
        if ACTIONS[output['action_info']['action_type'].item()]['target_unit']:
            self._last_target_unit_tag = action_info['target_unit_tag']
        else:
            self._last_target_unit_tag = None
        x = output['action_info']['target_location'].item() % SPATIAL_SIZE[1]
        y = output['action_info']['target_location'].item() // SPATIAL_SIZE[1]
        inverse_y = max(self._feature.map_size.y - y, 0)
        action_info['location'] = (x, inverse_y)
        if 'test' in self._job_type:
            self._print_action(output['action_info'], [x, y], output['action_logp'])
        return [action_info]

    def get_unit_num_info(self):
        return {'unit_num': self._stat_api.unit_num}

    def _print_action(self, action_info, location, logp):
        action_type = action_info['action_type'].item()
        action_name = ACTIONS[action_type]['name']
        selected_units = ''
        su_len = len(action_info['selected_units'])
        if ACTIONS[action_type]['selected_units']:
            for i, u in enumerate(action_info['selected_units'][:-1].tolist()):
                selected_units += ' ' + str(get_unit_type(UNIT_TYPES[self._observation['entity_info']['unit_type'][u]])).split('.')[-1] + '({:.2f})'.format(torch.exp(logp['selected_units'][i]).item())
            selected_units += ' ' + 'end({:.2f})'.format(torch.exp(logp['selected_units'][-1]).item())
        unit_types = set(self._observation['entity_info']['unit_type'][action_info['selected_units'][:-1]].tolist())
        target_unit = None
        if ACTIONS[action_type]['target_unit']:
            target_unit = str(get_unit_type(UNIT_TYPES[self._observation['entity_info']['unit_type'][action_info['target_unit'].item()]])).split('.')[-1]
        delay = action_info['delay']
        at_logp = torch.exp(logp['action_type']).item()
        delay_logp = torch.exp(logp['delay']).item()
        tl_logp = torch.exp(logp['target_location']).item()
        tu_logp = torch.exp(logp['target_unit']).item()
        s = f'{self.player_id}, game_step:{self._game_step}, at:{action_name}({at_logp:.2f}), delay:{delay}({delay_logp:.2f}), su:({su_len}){selected_units}, tu:{target_unit}({tu_logp:.2f}), lo:{location}({tl_logp:.2f})'
        print(s)

    def get_stat_data(self):
        data = self._stat_api.get_stat_data()

        bo_distance = levenshtein_distance(torch.as_tensor(self._behaviour_building_order, dtype=torch.int),
                                           torch.as_tensor(self._target_building_order, dtype=torch.int)).item()
        bo_distance_with_location = levenshtein_distance(torch.as_tensor(self._behaviour_building_order, dtype=torch.int),
                                                         torch.as_tensor(self._target_building_order, dtype=torch.int),
                                                         torch.as_tensor(self._behaviour_bo_location, dtype=torch.int),
                                                         torch.as_tensor(self._target_bo_location, dtype=torch.int),
                                                         partial(l2_distance, spatial_x=SPATIAL_SIZE[1])
                                                         ).item()
        stat_data = {
            'race_id': self.race,
            'step': self._game_step,
            'dist/bo': bo_distance,
            'dist/bo_location': bo_distance_with_location - bo_distance,
            'dist/cum': hamming_distance(torch.as_tensor(self._behaviour_cumulative_stat, dtype=torch.bool),
                                         torch.as_tensor(self._target_cumulative_stat, dtype=torch.bool)).item(),
            'bo_reward': self._total_bo_reward.item(),
            'cum_reward': self._total_cum_reward.item(),
            'bo_len': len(self._behaviour_building_order)
        }
        z_type_0, z_type_1 = 0, 0
        if not self.use_bo_reward:
            stat_data['dist/bo']=None
            stat_data['bo_reward']=None
            stat_data['bo_len']=None
            stat_data['dist/bo_location']=None
            z_type_0 = 1
        if not self.use_cum_reward:
            stat_data['dist/cum']=None
            stat_data['cum_reward']=None
            z_type_1 = 1
        stat_data['z_type'] = 2*z_type_1 + z_type_0
        data.update(stat_data)

        cum_in = defaultdict(int)
        cum_out = defaultdict(int)
        for i in range(len(self._behaviour_cumulative_stat)):
            if self.race not in cum_dict[i]['race']:
                continue
            cum_name = cum_dict[i]['name']
            if self._target_cumulative_stat[i] < 1e-3:
                if self._behaviour_cumulative_stat[i] >= 1:
                    cum_out['cum_out/' + cum_name] = 1
                else:
                    cum_out['cum_out/' + cum_name] = 0
            if self._target_cumulative_stat[i] > 1e-3:
                if self._behaviour_cumulative_stat[i] >= 1:
                    cum_in['cum_in/' + cum_name] = 1
                else:
                    cum_in['cum_in/' + cum_name] = 0
        data.update(cum_in)
        data.update(cum_out)
        return data

    def collect_data(self, next_obs, reward, done,idx):
        action_result = False if next_obs is None else ('Success' in next_obs['action_result'])
        if action_result:
            self._success_iter_count += 1

        behavior_z = self.get_behavior_z()
        bo_reward, cum_reward, battle_reward = self.update_fake_reward(next_obs)
        agent_obs = self._observation
        # teacher model forward
        teacher_obs = {'spatial_info': agent_obs['spatial_info'], 'entity_info': agent_obs['entity_info'],
                       'scalar_info': agent_obs['scalar_info'],
                       'entity_num': agent_obs['entity_num'], 'hidden_state': self._teacher_hidden_state,
                       'selected_units_num': self._output['selected_units_num'],
                       'action_info': self._output['action_info']}
        if self._whole_cfg.actor.use_cuda:
            teacher_obs = to_device(teacher_obs, 'cuda:0')
        if self._gpu_batch_inference:
            copy_input_data(self._teacher_shared_input, teacher_obs, data_idx=self._env_id)
            self._teacher_signals[self._env_id] += 1
            while True:
                if self._teacher_signals[self._env_id] == 0:
                    teacher_output = self._teacher_shared_output
                    teacher_output = self.decollate_output(teacher_output, batch_idx=self._env_id)
                    break
                else:
                    time.sleep(0.01)
        else:
            teacher_model_input = default_collate([teacher_obs])
            teacher_output = self.teacher_model.compute_teacher_logit(**teacher_model_input)
            teacher_output = self.decollate_output(teacher_output)
        self._teacher_hidden_state = teacher_output['hidden_state']
        # successive model forward
        if self._whole_cfg.learner.use_dapo:
            successive_obs = deepcopy(self._observation)
            successive_obs['hidden_state'] = self._successive_hidden_state
            successive_obs['selected_units_num'] = self._output['selected_units_num']
            successive_obs['action_info'] = self._output['action_info']
            successive_model_input = default_collate([successive_obs])
            successive_output = self.successive_model.compute_teacher_logit(**successive_model_input)
            successive_output = self.decollate_output(successive_output)
            self._successive_hidden_state = successive_output['hidden_state']

        # gather step data
        action_info = deepcopy(self._output['action_info'])
        mask = dict()
        mask['actions_mask'] = copy.deepcopy(
            {k: val for k, val in ACTIONS[action_info['action_type'].item()].items() if k not in ['name', 'goal','func_id','general_ability_id', 'game_id']})
        if self._only_cum_action_kl:
            mask['cum_action_mask'] = torch.tensor(0.0,dtype=torch.float)
        else:
            mask['cum_action_mask'] = torch.tensor(1.0,dtype=torch.float)
        if self.use_bo_reward:
            mask['build_order_mask'] = torch.tensor(1.0,dtype=torch.float)
        else:
            mask['build_order_mask'] = torch.tensor(0.0,dtype=torch.float)
        if self.use_cum_reward:
            mask['built_unit_mask'] = torch.tensor(1.0,dtype=torch.float)
            mask['cum_action_mask'] = torch.tensor(1.0,dtype=torch.float)
        else:
            mask['built_unit_mask'] = torch.tensor(0.0,dtype=torch.float)
        selected_units_num = self._output['selected_units_num']
        for k, v in mask['actions_mask'].items():
            mask['actions_mask'][k] = torch.tensor(v, dtype=torch.long)
        step_data = {
            'map_name': self._map_name,
            'spatial_info': agent_obs['spatial_info'],
            'model_last_iter': torch.tensor(self._model_last_iter, dtype=torch.float),
            # 'spatial_info_ref': spatial_info_ref,
            'entity_info': agent_obs['entity_info'],
            'scalar_info': agent_obs['scalar_info'],
            'entity_num': agent_obs['entity_num'],
            'selected_units_num': selected_units_num,
            'hidden_state': self._hidden_state_backup,
            'action_info': action_info,
            'behaviour_logp': self._output['action_logp'],
            'teacher_logit': teacher_output['logit'],
            # 'successive_logit': deepcopy(teacher_output['logit']),
            'reward': {'winloss': torch.tensor(reward, dtype=torch.float),
                       'build_order': bo_reward,
                       'built_unit': cum_reward,
                       # 'upgrade': torch.randint(-1, 1, size=(), dtype=torch.float),
                       'battle': battle_reward,
                       },
            'step': torch.tensor(self._game_step, dtype=torch.float),
            'mask': mask,
        }
        ##TODO: add value feature
        if self._use_value_feature:
            step_data['value_feature'] = agent_obs['value_feature']
            step_data['value_feature'].update(behavior_z)
        if self._whole_cfg.learner.use_dapo:
            step_data['successive_logit'] = successive_output['logit']
        self._hidden_state_backup = self._hidden_state

        # push data
        self._data_buffer.append(step_data)
        self._push_count += 1
        if self._push_count == self._whole_cfg.actor.traj_len or done:
            if not done:
                # can not obtain next observation in environment when done is true, use last step data instead,
                # next observation is not used in learner when done is true anyway
                if not next_obs['raw_obs'].observation:
                    return None
                self._pre_process(next_obs)
                agent_obs = deepcopy(self._observation)
                last_step_data = {
                    'map_name': self._map_name,
                    'spatial_info': agent_obs['spatial_info'],
                    # 'spatial_info_ref':spatial_info_ref,
                    'entity_info': agent_obs['entity_info'],
                    'scalar_info': agent_obs['scalar_info'],
                    'entity_num': agent_obs['entity_num'],
                    'hidden_state': self._hidden_state,
                }
            else:
                last_step_data = deepcopy({
                    'map_name': self._map_name,
                    'spatial_info': agent_obs['spatial_info'],
                    # 'spatial_info_ref':spatial_info_ref,
                    'entity_info': agent_obs['entity_info'],
                    'scalar_info': agent_obs['scalar_info'],
                    'entity_num': agent_obs['entity_num'],
                    'hidden_state': self._hidden_state,
                })
            if self._use_value_feature:
                last_step_data['value_feature'] = agent_obs['value_feature']
                last_step_data['value_feature'].update(self.get_behavior_z())
            list_data = list(self._data_buffer)
            list_data.append(last_step_data)
            self._push_count = 0
            return list_data
        else:
            return None
    
    def get_behavior_z(self):
        bo = self._behaviour_building_order + [0] * (BEGINNING_ORDER_LENGTH - len(self._behaviour_building_order))
        bo_location = self._behaviour_bo_location + [0] * (BEGINNING_ORDER_LENGTH - len(self._behaviour_bo_location))
        return {'beginning_order': torch.as_tensor(bo, dtype=torch.long), 'bo_location': torch.as_tensor(bo_location, dtype=torch.long),
                'cumulative_stat': torch.as_tensor(self._behaviour_cumulative_stat, dtype=torch.bool).long()}

    def update_fake_reward(self, next_obs):
        bo_reward, cum_reward, battle_reward = self._update_fake_reward(self._last_action_type, self._last_location, next_obs)
        return bo_reward, cum_reward, battle_reward

    def _update_fake_reward(self, action_type, location, next_obs):
        bo_reward = torch.zeros(size=(), dtype=torch.float)
        cum_reward = torch.zeros(size=(), dtype=torch.float)

        battle_score = compute_battle_score(next_obs['raw_obs'])
        opponent_battle_score = compute_battle_score(next_obs['opponent_obs'])
        battle_reward = battle_score - self._game_info['battle_score'] - (opponent_battle_score - self._game_info['opponent_battle_score'])
        battle_reward = torch.tensor(battle_reward, dtype=torch.float) / self._battle_norm

        if not self._exceed_flag:
            return bo_reward, cum_reward, battle_reward

        if action_type in BEGINNING_ORDER_ACTIONS and next_obs['action_result'][0] == 1:
            if action_type == 322:
                self._bo_zergling_count += 1
                if self._bo_zergling_count > 8:
                    return bo_reward, cum_reward, battle_reward
            order_index = BEGINNING_ORDER_ACTIONS.index(action_type)
            if order_index == 39 and 39 not in self._target_building_order:  # ignore spinecrawler
                return bo_reward, cum_reward, battle_reward
            if len(self._behaviour_building_order) < len(self._target_building_order):
                # only consider bo_reward if behaviour size < target size
                self._behaviour_building_order.append(order_index)
                if ACTIONS[action_type]['target_location']:
                    self._behaviour_bo_location.append(location.item())
                else:
                    self._behaviour_bo_location.append(0)
                if self.use_bo_reward:
                    if self._clip_bo:
                        tz = self._target_building_order[:len(self._behaviour_building_order)]
                        tz_lo = self._target_bo_location[:len(self._behaviour_building_order)]
                    else:
                        tz = self._target_building_order
                        tz_lo = self._target_bo_location
                    new_bo_dist = - levenshtein_distance(torch.as_tensor(self._behaviour_building_order, dtype=torch.int),
                                                       torch.as_tensor(tz, dtype=torch.int),
                                                       torch.as_tensor(self._behaviour_bo_location, dtype=torch.int),
                                                       torch.as_tensor(tz_lo, dtype=torch.int),
                                                       partial(l2_distance, spatial_x=SPATIAL_SIZE[1])
                                                       ) / self._bo_norm
                    bo_reward = new_bo_dist - self._old_bo_reward
                    self._old_bo_reward = new_bo_dist

        if self._cum_type == 'observation':
            cum_flag = True
            for u in next_obs['raw_obs'].observation.raw_data.units:
                if u.alliance == 1 and u.unit_type in [59, 18, 86]:  # ignore first base
                    if u.pos.x == self._born_location[0] and u.pos.y == self._born_location[1]:
                        continue
                if u.alliance == 1 and u.build_progress == 1 and UNIT_TO_CUM[u.unit_type] != -1:
                    self._behaviour_cumulative_stat[UNIT_TO_CUM[u.unit_type]] = 1
            for u in next_obs['raw_obs'].observation.raw_data.player.upgrade_ids:
                if UPGRADE_TO_CUM[u] != -1:
                    self._behaviour_cumulative_stat[UPGRADE_TO_CUM[u]] = 1
                    from distar.pysc2.lib.upgrades import Upgrades
                    for up in Upgrades:
                        if up.value == u:
                            name = up.name
                            break
        elif self._cum_type == 'action':
            action_name = ACTIONS[action_type]['name']
            action_info = self._output['action_info']
            cum_flag = False
            if action_name == 'Cancel_quick' or action_name == 'Cancel_Last_quick':
                unit_index = action_info['selected_units'][0].item()
                order_len = self._observation['entity_info']['order_length'][unit_index]
                if order_len == 0:
                    action_index = 0
                elif order_len == 1:
                    action_index = UNIT_ABILITY_TO_ACTION[self._observation['entity_info']['order_id_0'][unit_index].item()]
                elif order_len > 1:
                    order_str = 'order_id_{}'.format(order_len - 1)
                    action_index = QUEUE_ACTIONS[self._observation['entity_info'][order_str][unit_index].item() - 1]
                print(self.player_id, action_name, order_len.item(), 'cancel action:', ACTIONS[action_index]['name'])
                if action_index in CUMULATIVE_STAT_ACTIONS:
                    cum_flag = True
                    cum_index = CUMULATIVE_STAT_ACTIONS.index(action_index)
                    self._behaviour_cumulative_stat[cum_index] = max(0, self._behaviour_cumulative_stat[cum_index] - 1)

            if action_type in CUMULATIVE_STAT_ACTIONS:
                cum_flag = True
                cum_index = CUMULATIVE_STAT_ACTIONS.index(action_type)
                self._behaviour_cumulative_stat[cum_index] += 1
        else:
            raise NotImplementedError

        if self.use_cum_reward and cum_flag and (self._cum_type == 'observation' or next_obs['action_result'][0] == 1):
            new_cum_reward = -hamming_distance(
                torch.as_tensor(self._behaviour_cumulative_stat, dtype=torch.bool),
                torch.as_tensor(self._target_cumulative_stat, dtype=torch.bool)) / self._cum_norm
            cum_reward = (new_cum_reward - self._old_cum_reward) * self._get_time_factor(self._game_step)
            self._old_cum_reward = new_cum_reward
        self._total_bo_reward += bo_reward
        self._total_cum_reward += cum_reward
        return bo_reward, cum_reward, battle_reward

    def gpu_batch_inference(self, teacher=False):
        if not teacher:
            inference_indexes = self._signals.clone().bool()
            batch_num = inference_indexes.sum()
            if batch_num > 0:
                #print(self.player_id, 'batch num:', batch_num, inference_indexes)
                pass
            else:
                return
            model_input = to_device(self._shared_input, torch.cuda.current_device())
            model_output = self.model.compute_logp_action(**model_input)
            copy_output_data(self._shared_output, model_output, inference_indexes)
            self._signals[inference_indexes] *= 0
        else:
            inference_indexes = self._teacher_signals.clone().bool()
            batch_num = inference_indexes.sum()
            if batch_num > 0:
                #print(self.player_id, 'teacher batch num:', batch_num)
                pass
            else:
                return
            model_input = to_device(self._teacher_shared_input, torch.cuda.current_device())
            model_output = self.teacher_model.compute_teacher_logit(**model_input)
            copy_output_data(self._teacher_shared_output, model_output, inference_indexes)
            self._teacher_signals[inference_indexes] *= 0

    @staticmethod
    def _get_time_factor(game_step):
        if game_step < 1 * 10000:
            return 1.0
        elif game_step < 2 * 10000:
            return 0.5
        elif game_step < 3 * 10000:
            return 0.25
        else:
            return 0

    @property
    def player_id(self):
        return self._player_id

    @player_id.setter
    def player_id(self, player_id):
        self._player_id = player_id
    
    @property
    def env_id(self):
        return self._env_id

    @env_id.setter
    def env_id(self, env_id):
        self._env_id = env_id

    @property
    def race(self):
        return self._race

    @property
    def iter_count(self):
        return self._iter_count

