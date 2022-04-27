import torch
import os
import time
import random
import traceback
import mpyq
import six
import json

from distar.pysc2 import run_configs
from distar.pysc2.lib import point
from distar.pysc2.lib.static_data import NUM_UNIT_TYPES

from .lib.features import Features, MAX_DELAY
from distar.envs.map_info import get_map_size, LOCALIZED_BNET_NAME_TO_NAME_LUT
from distar.pysc2.lib.actions import RAW_FUNCTIONS
from distar.pysc2.run_configs.lib import VERSIONS

from s2clientprotocol import sc2api_pb2 as sc_pb
from sc2reader.decoders import BitPackedDecoder


RACE_DICT = {
    1: 'terran',
    2: 'zerg',
    3: 'protoss',
    4: 'random',
}

RESULT_DICT = {
    1: 'W',
    2: 'L',
    3: 'D',
    4: 'U'
}

BUILD2VERSION = {
    80188: "4.12.1",
    81009: "5.0.0",
    81433: "5.0.3"
}


def find_missed_tag(obs, action, saved_tags):
    if action.action_raw.HasField('unit_command'):
        uc = action.action_raw.unit_command
        if uc.HasField('target_unit_tag'):
            target_tag = uc.target_unit_tag
            tags = set()
            for i in obs.observation.raw_data.units:
                tags.add(i.tag)
            if target_tag not in tags:
                # match missed tag
                if target_tag in saved_tags.keys():
                    for i in obs.observation.raw_data.units:
                        if i.pos.x == saved_tags[target_tag][0] and i.pos.y == saved_tags[target_tag][1]:
                            action.action_raw.unit_command.target_unit_tag = i.tag

    return action


def get_tags(obs):
    tags = dict()
    for i in obs.observation.raw_data.units:
        if i.unit_type in [665, 666, 341, 1961, 483, 884, 885, 796, 797, 146, 147, 608, 880, 344, 881, 342]:
            tags[i.tag] = [i.pos.x, i.pos.y]
    return tags


class FilterActions:
    def __init__(self, flag=False):
        zerg_morph_action_names = ['Train_Baneling_quick', 'Train_Corruptor_quick', 'Train_Drone_quick',
                                   'Train_Hydralisk_quick',
                                   'Train_Infestor_quick', 'Train_Mutalisk_quick', 'Train_Overlord_quick',
                                   'Train_Roach_quick',
                                   'Train_SwarmHost_quick', 'Train_Ultralisk_quick', 'Train_Zergling_quick']
        self.morph_abilities = [a.ability_id for a in RAW_FUNCTIONS if a.name in zerg_morph_action_names or 'Morph' in a.name]
        self.train_abilities = [a.ability_id for a in RAW_FUNCTIONS if 'Train' in a.name and a.name not in zerg_morph_action_names]
        self.research_abilities = [a.ability_id for a in RAW_FUNCTIONS if 'Research' in a.name]
        self.corrosivebile = [2338]
        self.target_abilities = self.train_abilities + self.research_abilities + self.corrosivebile + self.morph_abilities
        self.max_loop = 4
        self.filter_flag = flag
        if self.filter_flag:
            print('[INFO] use filter action!!!')

    def gen_ability_id(self, action):
        ar = action.action_raw
        if ar.HasField('unit_command'):
            return ar.unit_command.ability_id
        elif ar.HasField('toggle_autocast'):
            return ar.toggle_autocast.ability_id

    def gen_unit_tags(self, action):
        ar = action.action_raw
        if ar.HasField('unit_command'):
            return ar.unit_command.unit_tags
        elif ar.HasField('toggle_autocast'):
            return ar.toggle_autocast.unit_tags

    def filter(self, actions, a_id, last_last_ob, last_ob, ob):
        # morph = True
        if a_id not in self.target_abilities or len(actions) == 1:
            return actions
        else:
            if actions[0].game_loop >= last_ob.observation.game_loop:
                pre_obs = last_ob.observation.raw_data
            else:
                pre_obs = last_last_ob.observation.raw_data
            post_obs = ob.observation.raw_data
            new_actions = []
            action_count = 0
            if a_id in self.morph_abilities:
                unit_tags = self.gen_unit_tags(actions[0])
                pre_unit_types = [u.unit_type for u in pre_obs.units]
                pre_unit_tags = [u.tag for u in pre_obs.units]
                post_unit_types = [u.unit_type for u in post_obs.units]
                post_unit_tags = [u.tag for u in post_obs.units]
                for t in unit_tags:
                    try:
                        pre_unit_index = pre_unit_tags.index(t)
                    except ValueError:
                        # print('not found')
                        action_count += 1
                        continue
                    try:
                        post_unit_index = post_unit_tags.index(t)
                    except ValueError:
                        # print('not found')
                        action_count += 1
                        continue
                    if pre_unit_types[pre_unit_index] != post_unit_types[post_unit_index]:
                        action_count += 1
                # if action_count < len(actions):
                #     # print('----------------morph--------------------')
                #     morph = False

            elif a_id in self.research_abilities:
                # print('----------------research--------------------')
                # print([(actions[0].action_raw, actions[0].game_loop)])
                return [actions[0]]
            elif a_id in self.corrosivebile:
                unit_tags = self.gen_unit_tags(actions[0])
                pre_unit_types = [u.unit_type for u in pre_obs.units]
                pre_unit_tags = [u.tag for u in pre_obs.units]
                action_count = 0
                for t in unit_tags:
                    try:
                        pre_unit_index = pre_unit_tags.index(t)
                    except ValueError:
                        # print('not found')
                        action_count += 1
                        continue
                    if pre_unit_types[pre_unit_index] == 688:  # Ravager
                        action_count += 1
                # if action_count < len(actions):
                #     print('----------------bile--------------------')
            elif a_id in self.train_abilities:
                unit_tags = self.gen_unit_tags(actions[0])
                pre_unit_orders = [len(u.orders) for u in pre_obs.units]
                pre_unit_tags = [u.tag for u in pre_obs.units]
                post_unit_orders = [len(u.orders) for u in post_obs.units]
                post_unit_tags = [u.tag for u in post_obs.units]
                pre_order_len = 0
                post_order_len = 0
                for t in unit_tags:
                    try:
                        pre_unit_index = pre_unit_tags.index(t)
                    except ValueError:
                        # print('not found')
                        return actions
                    pre_order_len += pre_unit_orders[pre_unit_index]
                    try:
                        post_unit_index = post_unit_tags.index(t)
                    except ValueError:
                        # print('not found')
                        return actions
                    post_order_len += post_unit_orders[post_unit_index]
                action_count = post_order_len - pre_order_len
                # if action_count < len(actions):
                #     print('----------------train--------------------')

            action_count = min(action_count, len(actions))
            for i in range(action_count):
                if i == action_count - 1:
                    index = - 1
                else:
                    index = (len(actions) // action_count) * i
                new_actions.append(actions[index])
            # if action_count < len(actions) and morph:
            #     print(action_count, len(actions))
            #     print([(a.action_raw, a.game_loop) for a in actions])
            #     print('**********************************************')
            #     print([(a.action_raw, a.game_loop) for a in new_actions])
            return new_actions

    def run(self, last_last_ob, last_ob, ob, cached_actions):
        if not self.filter_flag or ob.observation.game_loop > 8000:  # ignore filter after 6 min
            return [], cached_actions
        if len(cached_actions) == 0:
            return [], []
        actions = []
        consecutive_actions = []
        for idx, a in enumerate(cached_actions[:-1]):
            consecutive_actions.append(a)
            a_id = self.gen_ability_id(a)
            a_id_next = self.gen_ability_id((cached_actions[idx + 1]))
            loop_gap = cached_actions[idx + 1].game_loop - a.game_loop
            if a_id != a_id_next or loop_gap > self.max_loop:
                actions += self.filter(consecutive_actions, a_id, last_last_ob, last_ob, ob)
                consecutive_actions = []
        cached_actions = consecutive_actions + [cached_actions[-1]]
        return cached_actions, actions


class ReplayDecoder:
    def __init__(self, cfg):
        self._version = None
        self._sc2_process = None
        self._run_config = None
        self._controller = None
        self._restart_count = 0
        self._interface = sc_pb.InterfaceOptions(
            raw=True,
            score=False,
            raw_crop_to_playable_area=True,
            feature_layer=sc_pb.SpatialCameraSetup(width=1, crop_to_playable_area=True)
        )
        self._interface.feature_layer.resolution.x, self._interface.feature_layer.resolution.y = 1, 1
        self._parse_race = cfg.learner.data.parse_race
        self._cfg = cfg
        self._minimum_action_length = self._cfg.get('minimum_action_length', 128)
        self._filter = FilterActions(cfg.learner.data.filter_action)

    def _parse_replay(self):
        player = self._player_index + 1
        game_loops = self._replay_info['game_steps']

        size = point.Point(1, 1)  # action doesn't need spatial information, for speeding up
        size.assign_to(self._interface.feature_layer.minimap_resolution)
        self._controller.start_replay(
            sc_pb.RequestStartReplay(
                replay_path=self._replay_path,
                options=self._interface,
                observed_player_id=player,
            )
        )
        raw_map_size = self._controller.game_info().start_raw.map_size
        assert [raw_map_size.x, raw_map_size.y] == self._map_size
        cur_loop = 0
        player_actions = []
        filtered_player_actions = []
        step = 50
        cached_actions = []
        last_last_ob = last_ob = self._controller.observe()
        location = []
        for i in last_ob.observation.raw_data.units:
            if i.unit_type == 59 or i.unit_type == 18 or i.unit_type == 86:
                location.append([i.pos.x, i.pos.y])
        assert len(location) == 1, 'No fog of war in this replay'
        while cur_loop < game_loops:
            next_loop = min(game_loops, cur_loop + step)
            self._controller.step(next_loop - cur_loop)
            cur_loop = next_loop
            ob = self._controller.observe()
            for a in ob.actions:
                if a.HasField('action_raw'):
                    if not a.action_raw.HasField('camera_move'):
                        assert a.HasField('game_loop')  # debug
                        cached_actions.append(a)
                        player_actions.append(a)
            cached_actions, new_actions = self._filter.run(last_last_ob, last_ob, ob, cached_actions)
            last_last_ob = last_ob
            last_ob = ob
            filtered_player_actions += new_actions
            if len(ob.player_result):
                filtered_player_actions += cached_actions
                break
        size = point.Point(self._map_size[0], self._map_size[1])
        size.assign_to(self._interface.feature_layer.minimap_resolution)
        self._controller.start_replay(
            sc_pb.RequestStartReplay(
                replay_path=self._replay_path, options=self._interface, observed_player_id=player, disable_fog=False
            )
        )
        raw_ob = self._controller.observe()
        saved_tags = get_tags(raw_ob)
        traj_data = []
        game_info = self._controller.game_info()
        feature = Features(game_info, raw_ob, self._cfg)
        last_selected_unit_tags, last_target_unit_tag, last_delay, last_action_type, last_queued, enemy_unit_type_bool \
            = None, None, torch.tensor(0, dtype=torch.long), torch.tensor(0, dtype=torch.long), torch.tensor(0, dtype=torch.long), torch.zeros(NUM_UNIT_TYPES, dtype=torch.uint8)
        self._controller.step(max(player_actions[0].game_loop - 2, 0))
        for idx, action in enumerate(player_actions):
            if idx == len(player_actions) - 1:
                delay = random.randint(0, MAX_DELAY)
            else:
                delay = player_actions[idx + 1].game_loop - action.game_loop
            
            raw_ob = self._controller.observe()
            if len(raw_ob.player_result):
                break
            if delay > 0:
                self._controller.step(delay)
            action = find_missed_tag(raw_ob, action, saved_tags)
            # obs
            step_data = feature.transform_obs(raw_ob)
            # add last step info
            entity_num = step_data['entity_num']
            tags = step_data['game_info']['tags']
            last_selected_units = torch.zeros(entity_num, dtype=torch.int8)
            last_targeted_unit = torch.zeros(entity_num, dtype=torch.int8)
            if last_selected_unit_tags is not None:
                for t in last_selected_unit_tags:
                    if t in tags:
                        last_selected_units[tags.index(t)] = 1
            if last_target_unit_tag is not None:
                if last_target_unit_tag in tags:
                    last_targeted_unit[tags.index(last_target_unit_tag)] = 1
            step_data['entity_info']['last_selected_units'] = last_selected_units
            step_data['entity_info']['last_targeted_unit'] = last_targeted_unit
            step_data['scalar_info']['last_delay'] = last_delay
            step_data['scalar_info']['last_action_type'] = last_action_type
            step_data['scalar_info']['last_queued'] = last_queued
            step_data['scalar_info']['enemy_unit_type_bool'] = (enemy_unit_type_bool | step_data['scalar_info']['enemy_unit_type_bool']).to(torch.uint8)
            # action
            action, action_mask, selected_units_num, last_selected_unit_tags, last_target_unit_tag, invalid_action_flag = feature.reverse_raw_action(action, step_data['game_info']['tags'])
            if invalid_action_flag:
                continue
            action['delay'] = torch.tensor(delay, dtype=torch.long).clamp_(max=MAX_DELAY - 1)
            last_action_type, last_delay, last_queued = action['action_type'], action['delay'], action['queued']
            enemy_unit_type_bool = step_data['scalar_info']['enemy_unit_type_bool']
            step_data.update({'action_info': action, 'action_mask': action_mask, 'selected_units_num': selected_units_num})
            step_data.pop('game_info')
            traj_data.append(step_data)

        # add Z
        filtered_traj_data = []
        for a in filtered_player_actions:
            action, _, _, _, _, _ = feature.reverse_raw_action(
                            a, [])
            filtered_traj_data.append({'action_info': action})
        beginning_order, cumulative_stat, bo_len, bo_location = feature.get_z(filtered_traj_data)
        for step_data in traj_data:
            step_data['scalar_info']['beginning_order'] = beginning_order
            step_data['scalar_info']['cumulative_stat'] = cumulative_stat
            step_data['scalar_info']['bo_location'] = bo_location
        return traj_data

    def _parse_replay_info(self):
        replay_info = self._controller.replay_info(replay_path=self._replay_path)
        ret = dict()
        ret['race'] = [RACE_DICT[p.player_info.race_actual] for p in replay_info.player_info]
        ret['result'] = [RESULT_DICT[p.player_result.result] for p in replay_info.player_info]
        ret['player_type'] = [p.player_info.type for p in replay_info.player_info]
        ret['mmr'] = [p.player_mmr for p in replay_info.player_info]
        ret['map_name'] = LOCALIZED_BNET_NAME_TO_NAME_LUT[replay_info.map_name]
        ret['game_steps'] = replay_info.game_duration_loops 
        return ret

    def run(self, path, player_index):
        try:
            replay_path = path.strip()
            with open(replay_path, 'rb') as f:
                replay_io = six.BytesIO()
                replay_io.write(f.read())
                replay_io.seek(0)
                archive = mpyq.MPQArchive(replay_io).extract()
                metadata = json.loads(archive[b"replay.gamemetadata.json"].decode("utf-8"))
                versions = metadata["GameVersion"].split(".")[:-1]
                build = int(metadata["BaseBuild"][4:])
                if build in BUILD2VERSION:
                    versions = BUILD2VERSION[build].split(".")
                version = "{0}.{1}.{2}".format(*versions)
                env_path = 'SC2PATH{0}_{1}_{2}'.format(*versions)
                if env_path in os.environ:
                    os.environ['SC2PATH'] = os.environ[env_path]
                if version not in VERSIONS:
                    print(f'Decode replay ERROR: {replay_path}, no corresponded game version: {version}, use latest in stead.')
                    version = 'latest'
            self._player_index = player_index
            print(f'Start decoding replay with player {player_index}, path: {replay_path}')
            if self._version != version or self._restart_count == 10:
                if self._version is not None:
                    self._sc2_process.close()
                self._version = version
                if not self._restart():
                    self._version = None
                    self._restart_count = 0
                    return None
            # self._replay_data = self._run_config.replay_data(replay_path)
            self._replay_path = replay_path
            self._replay_info = self._parse_replay_info()
            if self._replay_info['player_type'][player_index] == 2:  # Computer
                return None
            if self._replay_info['race'][self._player_index][0].upper() not in self._parse_race:
                return None
            self._map_size = get_map_size(self._replay_info['map_name'])
            self._restart_count += 1
            start_time = time.time()
            data = self._parse_replay()
            if len(data) < self._minimum_action_length:
                return None
            else:
                game_loops = self._replay_info['game_steps']
                print(f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} path: {replay_path}, game step: {game_loops}, action length: {len(data)}, cost time: {time.time() - start_time:.2f}, per step time: {(time.time() - start_time) / len(data)}')
                return data
        except Exception as e:
            print(f'{os.getpid()} [ERROR] parse replay error', e)
            print(''.join(traceback.format_tb(e.__traceback__)))
            self._close()
            return None

    def _restart(self):
        for i in range(10):
            try:
                if self._sc2_process is not None and self._sc2_process.running:
                    self._close()
                self._run_config = run_configs.get(self._version)
                self._sc2_process = self._run_config.start(want_rgb=False)
                self._controller = self._sc2_process.controller
                return True
            except Exception as e:
                print(f'{os.getpid()} [ERROR] start sc2 ERROR, retry times: {i}', e)
                self._close()
        return False

    def _close(self):
        self._version = None
        try:
            if self._sc2_process is not None:
                self._sc2_process.close()
        except Exception as e:
            print(f'{os.getpid()} [ERROR] close sc2 failed!!!!!', e)

