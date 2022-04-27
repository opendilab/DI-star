import torch
import os
import time
import random
import traceback
import json
import multiprocessing
import mpyq
import argparse
import six
import json

from collections import defaultdict

from distar.pysc2 import run_configs
from distar.pysc2.lib import point
from distar.agent.default.lib.features import Features

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


def result_loop(result_queue, path_queue, output_file):
    result = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    cnt = 0
    while True:
        if result_queue.empty():
            time.sleep(0.01)
            continue
        data = result_queue.get()
        if data is None:
            continue
        if data == 'done':
            print(f'done {cnt}, left: {path_queue.qsize() * 2}')
            with open(output_file, 'w') as f:
                json.dump(result, f)
            return
        map_name, beginning_order, cumulative_stat, bo_location, born_location, race, loop = data
        result[map_name][race][born_location].append([beginning_order, cumulative_stat, bo_location, loop])
        cnt += 1
        if cnt % 10 == 0:
            print(f'done {cnt}, left: {path_queue.qsize() * 2}')
            with open(output_file, 'w') as f:
                json.dump(result, f)


def worker_loop(path_queue, result_queue):
    decoder = ReplayDecoder({})
    while not path_queue.empty():
        path = path_queue.get()
        for player_index in range(2):
            result_queue.put(decoder.run(path, player_index))


class FilterActions:
    def __init__(self, flag=True):
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
            action_count = 1
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
        if not self.filter_flag:
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
        self._parse_race = ['Z', 'T', 'P']
        self._filter = FilterActions()

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
        step = 50
        game_info = self._controller.game_info()
        raw_ob = self._controller.observe()
        feature = Features(game_info, raw_ob)
        traj_data = []
        race = RACE_DICT[feature.requested_races[raw_ob.observation.player_common.player_id]]
        opponent_id = 1 if raw_ob.observation.player_common.player_id == 2 else 2
        opponent_race = RACE_DICT[feature.requested_races[opponent_id]]
        if race == opponent_race:
            mix_race = race
        else:
            mix_race = race + opponent_race
        cached_actions = []
        last_last_ob = last_ob = raw_ob
        while cur_loop < game_loops:
            next_loop = min(game_loops, cur_loop + step)
            self._controller.step(next_loop - cur_loop)
            cur_loop = next_loop
            ob = self._controller.observe()
            for a in ob.actions:
                if a.HasField('action_raw'):
                    if not a.action_raw.HasField('camera_move'):
                        cached_actions.append(a)
            cached_actions, new_actions = self._filter.run(last_last_ob, last_ob, ob, cached_actions)
            last_last_ob = last_ob
            last_ob = ob
            player_actions += new_actions
            if len(ob.player_result):
                player_actions += cached_actions
                break
        for a in player_actions:
            action, _, _, _, _, _ = feature.reverse_raw_action(
                            a, [])
            traj_data.append({'action_info': action})
        # add Z
        beginning_order, cumulative_stat, bo_len, bo_location = feature.get_z(traj_data)
        bo_location = bo_location.tolist()
        beginning_order = beginning_order.tolist()
        cumulative_stat = cumulative_stat.nonzero().squeeze(dim=1).tolist()
        loop = player_actions[-1].game_loop
        return beginning_order, cumulative_stat, bo_len, bo_location, feature.home_born_location, mix_race, loop

    def _parse_replay_info(self):
        replay_info = self._controller.replay_info(replay_path=self._replay_path)
        ret = dict()
        ret['race'] = [RACE_DICT[p.player_info.race_actual] for p in replay_info.player_info]
        ret['result'] = [RESULT_DICT[p.player_result.result] for p in replay_info.player_info]
        ret['mmr'] = [p.player_mmr for p in replay_info.player_info]
        ret['map_name'] = LOCALIZED_BNET_NAME_TO_NAME_LUT[replay_info.map_name]
        ret['game_steps'] = replay_info.game_duration_loops
        return ret

    def run(self, path, player_index):
        self._player_index = player_index
        try:
            replay_path = path
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

            if self._version != version or self._restart_count == 10:
                if self._version is not None:
                    self._sc2_process.close()
                self._version = version
                if not self._restart():
                    self._version = None
                    self._restart_count = 0
                    return None
            print(f'Start decoding replay with player {self._player_index}, path: {path}')
            self._replay_path = replay_path
            self._replay_info = self._parse_replay_info()
            if self._replay_info['result'][self._player_index - 1] != 'W':
                return None
            self._player_indices = [player_idx for player_idx in range(2) if
                                    self._replay_info['race'][player_idx] in self._parse_race]
            self._map_size = get_map_size(self._replay_info['map_name'])
            self._restart_count += 1
            beginning_order, cumulative_stat, bo_len, bo_location, born_location, race, loop = self._parse_replay()
            if bo_len < 10:
                return None
            return self._replay_info['map_name'], beginning_order, cumulative_stat, bo_location, born_location, race, loop
        except Exception as e:
            print(f'{os.getpid()} [ERROR {e}] parse replay error')
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
                print(f'{os.getpid()} [ERROR {repr(e)}] start sc2 ERROR, retry times: {i}')
                self._close()
        return False

    def _close(self):
        self._version = None
        try:
            self._sc2_process.close()
        except Exception as e:
            print(f'{os.getpid()} [ERROR {repr(e)}] close sc2 failed!!!!!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="gen_z")
    parser.add_argument("--data", required=True, help='replay directory or file with replay paths')
    parser.add_argument("--name", required=True, help='output file name')
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()

    run_config = run_configs.get('4.10.0')
    num_procs = args.num_workers
    replay_path = args.data
    output_file = os.path.join(os.path.dirname(__file__), '../agent/default/lib','{}.json'.format(args.name))
    print('Z file will be saved at {}\n'.format(os.path.abspath(output_file)))

    path_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()
    data_path = []
    if os.path.isfile(replay_path):
        with open(replay_path, 'r') as f:
            for l in f.readlines():
                path_queue.put(l.strip())
    elif os.path.isdir(replay_path):
        for p in os.listdir(replay_path):
            path_queue.put(os.path.join(replay_path, p))

    result_loop_process = multiprocessing.Process(target=result_loop, args=(result_queue, path_queue, output_file), daemon=True, )
    result_loop_process.start()
    procs = []
    for i in range(num_procs):
        p = multiprocessing.Process(target=worker_loop, args=(path_queue, result_queue), daemon=True)
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    result_queue.put('done')
    result_loop_process.join()


