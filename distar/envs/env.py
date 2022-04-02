import collections
from absl import logging
import random
import time

import os
import enum

from distar.pysc2 import run_configs, maps
from distar.pysc2.env import environment
from distar.pysc2.lib import features
from distar.pysc2.lib import actions as actions_lib
from distar.pysc2.lib import metrics, portspicker
from distar.pysc2.lib import run_parallel
from distar.pysc2.lib import stopwatch
from distar.pysc2.lib import point
from distar.envs.map_info import get_map_size

from s2clientprotocol import common_pb2 as sc_common
from s2clientprotocol import sc2api_pb2 as sc_pb

sw = stopwatch.sw

possible_results = {
    sc_pb.Victory: 1,
    sc_pb.Defeat: -1,
    sc_pb.Tie: 0,
    sc_pb.Undecided: 0,
}


class Race(enum.IntEnum):
    random = sc_common.Random
    protoss = sc_common.Protoss
    terran = sc_common.Terran
    zerg = sc_common.Zerg


class Difficulty(enum.IntEnum):
    """Bot difficulties. Range: [1, 10]."""
    very_easy = sc_pb.VeryEasy
    easy = sc_pb.Easy
    medium = sc_pb.Medium
    medium_hard = sc_pb.MediumHard
    hard = sc_pb.Hard
    harder = sc_pb.Harder
    very_hard = sc_pb.VeryHard
    cheat_vision = sc_pb.CheatVision
    cheat_money = sc_pb.CheatMoney
    cheat_insane = sc_pb.CheatInsane


class BotBuild(enum.IntEnum):
    """Bot build strategies."""
    random = sc_pb.RandomBuild
    rush = sc_pb.Rush
    timing = sc_pb.Timing
    power = sc_pb.Power
    macro = sc_pb.Macro
    air = sc_pb.Air


#MAPS = ['KairosJunction', 'CyberForest', 'Acropolis', 'KingsCove', 'NewRepugnancy', 'Thunderbird', 'PortAleksander']
MAPS = ['KairosJunction', 'KingsCove', 'NewRepugnancy']


def to_list(arg):
    return arg if isinstance(arg, list) else [arg]


def get_default(a, b):
    return b if a is None else a


class Agent(collections.namedtuple("Agent", ["race", "name"])):
    """Define an Agent. It can have a single race or a list of races."""

    def __new__(cls, race, name=None):
        return super(Agent, cls).__new__(cls, to_list(race), name or "<unknown>")


class Bot(collections.namedtuple("Bot", ["race", "difficulty", "build"])):
    """Define a Bot. It can have a single or list of races or builds."""

    def __new__(cls, race, difficulty, build=None):
        return super(Bot, cls).__new__(
            cls, to_list(race), difficulty, to_list(build or BotBuild.random))



REALTIME_GAME_LOOP_SECONDS = 1 / 22.4
MAX_STEP_COUNT = 524000  # The game fails above 2^19=524288 steps.
NUM_ACTION_DELAY_BUCKETS = 10


class SC2Env(object):
    def __init__(self, cfg):
        self._whole_cfg = cfg
        self._cfg = cfg.env
        players = []
        self._human_flag = False
        for idx in range(2):
            if 'bot' in self._cfg.player_ids[idx]:
                bot_level = int(self._cfg.player_ids[idx].split('bot')[1])
                players.append(Bot(Race[self._cfg.races[idx]], bot_level))
            else:
                players.append(Agent(Race[self._cfg.races[idx]], self._cfg.player_ids[idx]))
                if 'human' in self._cfg.player_ids[idx]:
                    self._human_flag = True

        num_players = len(players)
        self._num_agents = sum(1 for p in players if isinstance(p, Agent))
        self._players = players

        self._ori_map_name = self._cfg.get('map_name', 'KairosJunction')
        if len(self._ori_map_name.split('_')) == 2:
            self._ori_map_name, self._born_location = self._ori_map_name.split('_')
        else:
            self._born_location = None

        self._realtime = self._cfg.get('realtime', False)
        self._last_step_time = None
        self._save_replay_episodes = self._cfg.get('save_replay_episodes', 0)
        self._replay_dir = self._cfg.get('replay_dir', '.')
        self._result_dir = self._replay_dir.replace('replays', 'replay_results')
        self._random_seed = self._cfg.get('random_seed', None)
        if self._random_seed == 'none':
            self._random_seed = None
        self._default_episode_length = self._cfg.get('game_steps_per_episode', 100000)
        self._version = self._cfg.get('version', None)
        self._update_both_obs = self._cfg.get('update_both_obs', False)
        if self._human_flag:
            self._update_both_obs = False

        self._run_config = run_configs.get(version=self._version)
        self._parallel = run_parallel.RunParallel()  # Needed for multiplayer.

        self._last_score = None
        self._total_steps = 0
        self._episode_steps = 0
        self._episode_count = 0
        self._obs = [None] * self._num_agents
        self._agent_obs = [None] * self._num_agents
        self._state = environment.StepType.LAST  # Want to jump to `reset`.
        self._controllers = None
        self._sc2_procs = None
        self._ports = None
        self._random_delay_weights = self._cfg.get('random_delay_weights', [0, 0.7, 0.2, 0.1])

    def _setup_interface(self):
        self._interface = []
        map_size = get_map_size(self._map_name)
        if self._human_flag:
            raw_affects_selection = True
        else:
            raw_affects_selection = False
        for i in range(self._num_agents):
            interface = sc_pb.InterfaceOptions(
                raw=True,
                show_cloaked=False,
                show_burrowed_shadows=False,
                show_placeholders=False,
                raw_affects_selection=raw_affects_selection,
                raw_crop_to_playable_area=True,
                score=True)

            interface.feature_layer.width = 24
            interface.feature_layer.resolution.x = 1
            interface.feature_layer.resolution.y = 1
            if self._cfg.map_size_resolutions[i]:
                interface.feature_layer.minimap_resolution.x = map_size[0]
                interface.feature_layer.minimap_resolution.y = map_size[1]
            else:
                interface.feature_layer.minimap_resolution.x = self._cfg.minimap_resolutions[i][0]
                interface.feature_layer.minimap_resolution.y = self._cfg.minimap_resolutions[i][1]
            interface.feature_layer.crop_to_playable_area = True
            self._interface.append(interface)

    def _launch_game(self):
        # Reserve a whole bunch of ports for the weird multiplayer implementation.
        max_retry_times = 10
        for i in range(max_retry_times):
            try:
                if self._num_agents > 1:
                    self._ports = portspicker.pick_unused_ports(self._num_agents * 2)
                    # logging.info("Ports used for multiplayer: %s", self._ports)
                else:
                    self._ports = []

                # Actually launch the game processes.
                if self._human_flag:
                    self._sc2_procs = [
                        self._run_config.start(extra_ports=self._ports,
                                               want_rgb=False),
                        self._run_config.start(extra_ports=self._ports,
                                               want_rgb=False, full_screen=True)
                    ]
                else:
                    self._sc2_procs = [
                        self._run_config.start(extra_ports=self._ports,
                                            want_rgb=False)
                        for _ in range(self._num_agents)]
                self._controllers = [p.controller for p in self._sc2_procs]
                return
            except Exception as e:
                print('[ERROR {}] start SC2 failed, retry times: {}'.format(e, i))
                self.close()
                if i == max_retry_times - 1:
                    raise e

    def _create_join(self):
        """Create the game, and join it."""
        map_inst = random.choice(self._maps)
        self._map_name = map_inst.name

        self._episode_length = get_default(self._default_episode_length,
                                           map_inst.game_steps_per_episode)
        if self._episode_length <= 0 or self._episode_length > MAX_STEP_COUNT:
            self._episode_length = MAX_STEP_COUNT

        # Create the game. Set the first instance as the host.
        create = sc_pb.RequestCreateGame(
            disable_fog=False,
            realtime=self._realtime)
        if self._born_location is not None:
            map_inst.filename = map_inst.filename + '_' + self._born_location
        create.local_map.map_path = map_inst.path
        map_data = map_inst.data(self._run_config)
        if self._num_agents == 1:
            create.local_map.map_data = map_data
        else:
            # Save the maps so they can access it. Don't do it in parallel since SC2
            # doesn't respect tmpdir on windows, which leads to a race condition:
            # https://github.com/Blizzard/s2client-proto/issues/102
            for c in self._controllers:
                c.save_map(map_inst.path, map_data)

        if self._random_seed is not None:
            create.random_seed = self._random_seed
        for p in self._players:
            if isinstance(p, Agent):
                create.player_setup.add(type=sc_pb.Participant)
            else:
                create.player_setup.add(
                    type=sc_pb.Computer, race=random.choice(p.race),
                    difficulty=p.difficulty, ai_build=random.choice(p.build))
        if self._num_agents > 1:
            self._controllers[1].create_game(create)
        else:
            self._controllers[0].create_game(create)

        # Create the join requests.
        agent_players = [p for p in self._players if isinstance(p, Agent)]
        self.sanitized_names = crop_and_deduplicate_names(p.name for p in agent_players)
        join_reqs = []
        for p, name, interface in zip(agent_players, self.sanitized_names, self._interface):
            join = sc_pb.RequestJoinGame(options=interface)
            join.race = random.choice(p.race)
            join.player_name = name
            if self._ports:
                join.shared_port = 0  # unused
                join.server_ports.game_port = self._ports[0]
                join.server_ports.base_port = self._ports[1]
                for i in range(self._num_agents - 1):
                    join.client_ports.add(game_port=self._ports[i * 2 + 2],
                                          base_port=self._ports[i * 2 + 3])
            join_reqs.append(join)

        # Join the game. This must be run in parallel because Join is a blocking
        # call to the game that waits until all clients have joined.
        self._parallel.run((c.join_game, join)
                           for c, join in zip(self._controllers, join_reqs))

        self._game_info = self._parallel.run(c.game_info for c in self._controllers)

    @property
    def map_name(self):
        return self._map_name

    @property
    def game_info(self):
        """A list of ResponseGameInfo, one per agent."""
        return self._game_info

    def static_data(self):
        return self._controllers[0].data()

    def _restart(self):
        if (len(self._players) == 1 and len(self._players[0].race) == 1 and
                len(self._maps) == 1):
            # Need to support restart for fast-restart of mini-games.
            self._controllers[0].restart()
        else:
            if len(self._controllers) > 1 and self._episode_count:
                self._parallel.run(c.leave for c in self._controllers)
            self._create_join()

    def reset(self, players=None):
        """Start a new episode."""
        if self._ori_map_name == 'random':
            self._map_name = random.choice(MAPS)
        else:
            self._map_name = self._ori_map_name
        self._maps = [maps.get(name) for name in to_list(self._map_name)]
        self._setup_interface()
        if players:
            self._players = players
        self._episode_steps = 0
        if self._controllers is None or (self._episode_count + 1) % 10 == 0:  #  restart game for memory release
            self.close()
            self._launch_game()
        self._restart()

        self._next_obs_step = [0] * self._num_agents
        if self._human_flag:
            self._next_obs_step[1] = 9999999
        self._action_result = [[0]] * self._num_agents
        self._episode_count += 1
        races = self._cfg.races
        logging.info("Starting episode %s: [%s] on %s",
                     self._episode_count, ", ".join(races), self._map_name)

        self._last_score = [0] * self._num_agents
        self._state = environment.StepType.FIRST
        if self._realtime:
            self._last_step_time = time.time()
            self._action_delays = [[0] * NUM_ACTION_DELAY_BUCKETS] * self._num_agents
        observations, reward, done = self._observe(target_game_loop=0)
        game_info = {idx: v for idx, v in enumerate(self._game_info)}
        return observations, game_info, self._map_name

    # action_space {'func_id': 0, 'skip_steps': 0, 'queued': 0, 'unit_tags': [0, 1], 'target_unit_tag': 0, 'target_location': [0,0]}
    def step(self, actions):
        if self._state == environment.StepType.LAST:
            return self.reset()

        transformed_actions = []
        valid_op_idx = []
        max_skip_steps = 0
        for agent_idx in range(self._num_agents):
            if agent_idx in actions:
                action, skip_steps = self.transform_action(actions[agent_idx])
                transformed_actions.append(action)
                self._next_obs_step[agent_idx] = self._episode_steps + skip_steps
                max_skip_steps = max(max_skip_steps, skip_steps)
                valid_op_idx.append(agent_idx)
            else:
                transformed_actions.append([])

        random_step = 0
        if not self._realtime and max_skip_steps < 4:  # simulate inference and network latency in realtime mode
            random_step = random.choices(list(range(len(self._random_delay_weights))), weights=self._random_delay_weights, k=1)[0]
            if not self._controllers[0].status_ended:  # May already have ended.
                steps = self._parallel.run((c.step, random_step) for c in self._controllers)

        funcs_with_args = []
        for c, a in zip(self._controllers, transformed_actions):
            if a:
                item = (c.acts, a)
                funcs_with_args.append(item)
        if not self._controllers[0].status_ended:
            action_result = self._parallel.run(funcs_with_args)
            for idx in range(len(action_result)):
                if action_result[idx] is not None:
                    result = action_result[idx].result
                    self._action_result[valid_op_idx[idx]] = result # actions and results could be many

        if not self._realtime:
            self._episode_steps += random_step
        step_mul = min(self._next_obs_step) - self._episode_steps
        step_mul = max(0, step_mul)
        target_game_loop = self._episode_steps + step_mul
        if not self._controllers[0].status_ended:  # May already have ended.
            steps = self._parallel.run((c.step, step_mul) for c in self._controllers)
        return self._observe(target_game_loop)

    def _observe(self, target_game_loop):
        def parallel_observe(c):
            obs = c.observe(target_game_loop=target_game_loop)
            return obs

        agent_indices = [idx for idx, next_obs_step in enumerate(self._next_obs_step) if next_obs_step <= target_game_loop]
        observe_funcs = []
        if self._human_flag:
            agent_indices = [0]
        for agent_idx in range(self._num_agents):
            if self._update_both_obs or agent_idx in agent_indices:
                observe_funcs.append((parallel_observe, self._controllers[agent_idx]))
        observations = self._parallel.run(observe_funcs)
        game_loops = []
        if self._update_both_obs:
            self._obs = observations
            game_loops.append(self._obs[0].observation.game_loop)
        else:
            for idx, agent_idx in enumerate(agent_indices):
                self._obs[agent_idx] = observations[idx]
                game_loops.append(self._obs[agent_idx].observation.game_loop)
        if len(game_loops) == 2 and not self._realtime:
            assert game_loops[0] == game_loops[1], '2 agents step into different game loop'
        game_loop = game_loops[0]

        if (game_loop < target_game_loop and
                not any(o.player_result for o in self._obs)):
            raise ValueError(
                ("The game didn't advance to the expected game loop. "
                 "Expected: %s, got: %s") % (target_game_loop, game_loop))
        elif game_loop > target_game_loop and target_game_loop > 0:
            logging.warn("Received observation %d step(s) late: %d rather than %d.",
                         game_loop - target_game_loop, game_loop, target_game_loop)

        outcome = [0] * self._num_agents
        episode_complete = any(o.player_result for o in self._obs if o is not None)
        if episode_complete:
            self._state = environment.StepType.LAST
            for i, o in enumerate(self._obs):
                if o is None:
                    continue
                player_id = o.observation.player_common.player_id
                for result in o.player_result:
                    if result.player_id == player_id:
                        outcome[i] = possible_results.get(result.result, 0)
                    elif self._num_agents == 2:
                        outcome[1 - i] = possible_results.get(result.result, 0)

        reward = outcome
        self._total_steps += game_loop - self._episode_steps
        self._episode_steps = game_loop
        if self._episode_steps >= self._episode_length:
            self._state = environment.StepType.LAST
            episode_complete = True
            if self._episode_steps >= MAX_STEP_COUNT:
                logging.info("Cut short to avoid SC2's max step count of 2^19=524288.")

        if self._state == environment.StepType.LAST:
            if (self._save_replay_episodes > 0 and
                    self._episode_count % self._save_replay_episodes == 0):
                prefix = '_'.join([self._map_name, '_vs_'.join(self.sanitized_names), str(outcome)])
                self.save_replay(self._replay_dir, prefix)
            logging.info(("Episode %s finished after %s game steps. "
                          "Outcome: %s, reward: %s"),
                         self._episode_count, self._episode_steps, outcome, reward)

        ret = {}
        if episode_complete:
            agent_indices = list(range(self._num_agents))

        for agent_idx in agent_indices:
            if self._num_agents == 2:
                opponent_obs = self._obs[1 - agent_idx]
            else:
                opponent_obs = None
            ret[agent_idx] = {'raw_obs': self._obs[agent_idx],
                              'opponent_obs': opponent_obs,
                              'action_result': self._action_result[agent_idx]}
        return ret, reward, episode_complete

    def transform_action(self, actions):
        # action_space {'func_id': 0, 'skip_steps': 0, 'queued': 0, 'unit_tags': [0, 1], 'target_unit_tag': 0, 'location': [0,0]}
        sc2_actions = []
        skip_steps = MAX_STEP_COUNT
        for action in actions:
            skip_steps = min(skip_steps, action['skip_steps'])
            raw_action = actions_lib.RAW_FUNCTIONS[action['func_id']]
            ability_id = raw_action.ability_id
            if raw_action.function_type is actions_lib.raw_no_op:
                continue
            elif raw_action.function_type is actions_lib.raw_cmd:
                args = [ability_id, action['queued'], action['unit_tags']]
            elif raw_action.function_type is actions_lib.raw_cmd_pt:
                location = point.Point(*action['location'])
                args = [ability_id, action['queued'], action['unit_tags'], location]
            elif raw_action.function_type is actions_lib.raw_cmd_unit:
                args = [ability_id, action['queued'], action['unit_tags'], action['target_unit_tag']]
            elif raw_action.function_type is actions_lib.raw_move_camera:
                location = point.Point(*action['location'])
                args = [location]
            elif raw_action.function_type is actions_lib.raw_autocast:
                args = [ability_id, action['unit_tags']]
            sc2_action = sc_pb.Action()
            args.insert(0, sc2_action)
            raw_action.function_type(*args)
            sc2_actions.append(sc2_action)
        return sc2_actions, skip_steps

    def save_replay(self, replay_dir, prefix=None):
        if prefix is None:
            prefix = self._map_name
        if ('raceid' in replay_dir and 'repeat' in replay_dir) or self._whole_cfg.actor.job_type == 'eval_test':
            replay_dir = replay_dir
            replay_dir = os.path.abspath(replay_dir)
        else:
            time_label = '-'.join(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time())).split('-')[:-2])
            replay_dir = os.path.abspath(os.path.join(replay_dir, time_label))
        replay_path = self._run_config.save_replay(
            self._controllers[0].save_replay(), replay_dir, prefix)
        logging.info("Wrote replay to: %s", replay_path)
        print("Wrote replay to: {}".format(replay_path))
        if self._whole_cfg.env.get('save_replay',True) == False:
            try:
                os.remove(replay_path)
            except:
                pass
        return replay_path

    @property
    def state(self):
        return self._state

    def close(self):
        # Don't use parallel since it might be broken by an exception.
        if self._controllers:
            for c in self._controllers:
                c.quit()
            self._controllers = None
        if self._sc2_procs:
            for p in self._sc2_procs:
                p.close()
            self._sc2_procs = None

        if self._ports:
            portspicker.return_ports(self._ports)
            self._ports = None
        self._episode_count = 0
        self._game_info = None

    @property
    def obs_pb(self):
        return self._obs

    @property
    def controller(self):
        return self._controllers[0]


def crop_and_deduplicate_names(names):
    """Crops and de-duplicates the passed names.

    SC2 gets confused in a multi-agent game when agents have the same
    name. We check for name duplication to avoid this, but - SC2 also
    crops player names to a hard character limit, which can again lead
    to duplicate names. To avoid this we unique-ify names if they are
    equivalent after cropping. Ideally SC2 would handle duplicate names,
    making this unnecessary.

    TODO(b/121092563): Fix this in the SC2 binary.

    Args:
      names: List of names.

    Returns:
      De-duplicated names cropped to 32 characters.
    """
    max_name_length = 32

    # Crop.
    cropped = [n[:max_name_length] for n in names]

    # De-duplicate.
    deduplicated = []
    name_counts = collections.Counter(n for n in cropped)
    name_index = collections.defaultdict(lambda: 1)
    for n in cropped:
        if name_counts[n] == 1:
            deduplicated.append(n)
        else:
            deduplicated.append("({}) {}".format(name_index[n], n))
            name_index[n] += 1

    # Crop again.
    recropped = [n[:max_name_length] for n in deduplicated]
    if len(set(recropped)) != len(recropped):
        raise ValueError("Failed to de-duplicate names")

    return recropped
