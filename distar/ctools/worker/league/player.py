import pprint
import random
from abc import abstractmethod

import numpy as np

from distar.ctools.worker.league.algorithms import pfsp
from .cum_stat import CumStat
from .dist_stat import DistStat
from .payoff import Payoff
from .unit_num_stat import UnitNumStat
FRAC_ID = {0:  ['zerg', 'terran', 'protoss'], 1: ['zerg'], 2: ['terran'], 3: ['protoss']}


class Player:
    """
    Overview:
        Base player class, player is the basic member of a league
    Interfaces:
        __init__
    Property:
        race, payoff, checkpoint_path, player_id, total_agent_step
    """
    _name = "BasePlayer"  # override this variable for sub-class player
    _stat_keys = ['checkpoint_path', 'player_id', 'pipeline', 'frac_id', 'z_path', 'z_prob',
                  'teacher_id', 'teacher_checkpoint_path',
                  'total_agent_step', 'decay', 'warm_up_size', 'min_win_rate_games','total_game_count']
    _log_keys = ['payoff', 'teammate_payoff', 'opponent_payoff', 'dist_stat', 'cum_stat', 'unit_num_stat']

    def __init__(
            self,
            checkpoint_path: str,
            player_id: str,
            pipeline: str,
            frac_id: int,
            z_path: str,
            z_prob: float,
            teacher_id: str,
            teacher_checkpoint_path: str,
            total_agent_step: int = 0,
            decay: float = 0.99,
            warm_up_size: int = 1000,
            min_win_rate_games: int = 200,
            total_game_count: int = 0,
            payoff: Payoff = None,
    ) -> None:
        """
        Overview:
            Initialize base player metadata
        Arguments:
            - cfg (:obj:`EasyDict`): player config dict
                e.g. StarCraft has 3 races ['terran', 'protoss', 'zerg']
            - checkpoint_path (:obj:`str`): one training phase step
            - player_id (:obj:`str`): player id
            - total_agent_step (:obj:`int`):  for active player, it should be 0, \
                for historical player, it should be parent player's ``_total_agent_step`` when ``snapshot``
        """
        self.checkpoint_path = checkpoint_path
        self.player_id = player_id
        self.pipeline = pipeline
        self.frac_id = frac_id
        self.z_path = z_path
        self.z_prob = z_prob
        self.teacher_id = teacher_id
        self.teacher_checkpoint_path = teacher_checkpoint_path
        self.total_agent_step = total_agent_step
        self.warm_up_size = warm_up_size
        self.min_win_rate_games = min_win_rate_games
        self.decay = decay
        self.total_game_count = total_game_count
        # record stat
        self.payoff = Payoff(decay, warm_up_size, min_win_rate_games)
        if payoff:
            self.payoff._stat_info_record = payoff._stat_info_record

    def get_race(self):
        return random.choice(FRAC_ID[self.frac_id])

    def reset_stats(self, stat_types=[]):
        reset_stat_types = self._log_keys if not stat_types else stat_types
        for k in reset_stat_types:
            func = getattr(self, f'reset_{k}', None)
            if func is None:
                print(f'cant reset stat: {k}')
            else:
                func()

    def reset_payoff(self):
        self.payoff = Payoff(self.decay, self.warm_up_size)

    def __repr__(self):
        info = {attr_type: getattr(self, attr_type) for attr_type in self._stat_keys}
        return pprint.pformat(info)


class HistoricalPlayer(Player):
    """
    Overview:
        Historical player with fixed checkpoint, has a unique attribute ``parent_id``.
        Teacher_id and teacher_checkpoint_path will be 'none'.
    Property:
        race, payoff, checkpoint_path, player_id, total_agent_step, parent_id
    """
    _name = "HistoricalPlayer"
    _stat_keys = Player._stat_keys + ['parent_id',]

    def __init__(self,
                 checkpoint_path: str,
                 player_id: str,
                 pipeline: str,
                 frac_id: int,
                 z_path: str,
                 z_prob: float,
                 total_agent_step: int = 0,
                 decay: float = 0.995,
                 warm_up_size: int = 1000,
                 min_win_rate_games: int = 200,
                 total_game_count: int = 0,
                 parent_id: str = 'none',
                 payoff: Payoff = None,
                 ) -> None:
        """
        Overview:
            Initialize ``_parent_id`` additionally
        Arguments:
            - parent_id (:obj:`str`): id of historical player's parent, should be an active player
        """
        super(HistoricalPlayer, self).__init__(checkpoint_path, player_id, pipeline, frac_id, z_path,z_prob,
                                               'none', 'none',
                                               total_agent_step, decay, warm_up_size, min_win_rate_games,total_game_count,
                                               payoff)
        self.parent_id = parent_id


class ActivePlayer(Player):
    """
    Overview:
        Active player class, active player can be updated
    Interface:
        __init__, is_trained_enough, snapshot, mutate, get_job
    Property:
        race, payoff, checkpoint_path, player_id, total_agent_step
    """
    _name = "ActivePlayer"
    _stat_keys = Player._stat_keys + ['one_phase_step', 'chosen_weight', 'last_enough_step', 'snapshot_times', 'strong_win_rate', ]

    def __init__(self, checkpoint_path: str, player_id: str, pipeline: str, frac_id: int, z_path: str, z_prob: float,
                 teacher_id: str, teacher_checkpoint_path: str, chosen_weight:float = 1.0,
                 total_agent_step: int = 0,
                 decay: float = 0.995, warm_up_size: int = 1000, min_win_rate_games: int = 200,
                 total_game_count: int = 0,
                 one_phase_step: int = 2e8, last_enough_step: int = 0, snapshot_times: int = 0,
                 strong_win_rate: float = 0.7,
                 payoff: Payoff = None,
                 teammate_payoff: Payoff = None,
                 opponent_payoff: Payoff = None,
                 dist_stat: DistStat = None,
                 cum_stat: CumStat = None,
                 unit_num_stat: UnitNumStat = None,
                 successive_model_path: str = None,
                 ) -> None:
        """
        Overview:
            Initialize player metadata, depending on the game
        Note:
            - one_phase_step (:obj:`int`): active player will be considered trained enough after one phase step
            - last_enough_step (:obj:`int`): player's last step number that satisfies ``_is_trained_enough``
            - exploration (:obj:`function`): exploration function, e.g. epsilon greedy with decay
            - snapshot flag and reset flag will change only when we use league update
        """
        super(ActivePlayer, self).__init__(checkpoint_path, player_id, pipeline, frac_id, z_path,z_prob,
                                           teacher_id, teacher_checkpoint_path,
                                           total_agent_step, decay, warm_up_size, min_win_rate_games, total_game_count,
                                           payoff)
        # ``one_phase_step`` is like 1e9
        self.one_phase_step = one_phase_step
        self.last_enough_step = last_enough_step
        self.snapshot_times = snapshot_times
        self.strong_win_rate = strong_win_rate
        self.snapshot_flag = False
        self.reset_flag = False
        self.chosen_weight = chosen_weight
        if successive_model_path is None:
            self.successive_model_path = self.checkpoint_path
        else:
            self.successive_model_path = successive_model_path
        self.last_successive_step = last_enough_step
        self.teammate_payoff = Payoff(decay, warm_up_size, min_win_rate_games)
        self.opponent_payoff = Payoff(decay, warm_up_size, min_win_rate_games)
        if opponent_payoff:
            self.opponent_payoff._stat_info_record = opponent_payoff.stat_info_record
        if teammate_payoff:
            self.teammate_payoff._stat_info_record = teammate_payoff.stat_info_record
        self.dist_stat = dist_stat if dist_stat else DistStat(decay, warm_up_size)
        self.cum_stat = cum_stat if cum_stat else CumStat(decay, warm_up_size)
        self.unit_num_stat = unit_num_stat if unit_num_stat else UnitNumStat(decay, warm_up_size)

    @abstractmethod
    def get_branch_opponent(self, historical_players, active_players, branch_probs_dict, pfsp_train_bot=False):
        raise NotImplementedError

    @abstractmethod
    def is_trained_enough(self, historical_players, active_players, *args, **kwargs) -> bool:
        raise NotImplementedError

    def is_save_successive_model(self):
        step_passed = self.total_agent_step - self.last_successive_step
        if step_passed > self.one_phase_step/2:
            self.last_successive_step = self.total_agent_step
            return True
        else:
            return False

    def snapshot(self) -> HistoricalPlayer:
        """
        Overview:
            Generate a snapshot historical player from the current player, called in league manager's ``_snapshot``.
        Returns:
            - snapshot_player (:obj:`HistoricalPlayer`): new instantiated historical player
        Note:
            This method only generates a historical player object without saving the checkpoint, which should be
            completed by the interaction between coordinator and learner.
        """
        self.snapshot_times += 1
        h_player_id = self.player_id + f'H{self.snapshot_times}'
        h_player_path = self.checkpoint_path.split(
            '.pth')[0] + '_{}'.format(self.total_agent_step) + '.pth'
        return HistoricalPlayer(checkpoint_path=h_player_path, player_id=h_player_id, pipeline=self.pipeline,
                                frac_id=self.frac_id, z_path=self.z_path,z_prob=self.z_prob,
                                total_agent_step=self.total_agent_step, decay=self.decay,
                                warm_up_size=self.warm_up_size, min_win_rate_games=self.min_win_rate_games,
                                parent_id=self.player_id, )

    def is_reset(self):
        return False
    
    def reset_teammate_payoff(self):
        self.teammate_payoff = Payoff(self.decay, self.warm_up_size)

    def reset_opponent_payoff(self):
        self.opponent_payoff = Payoff(self.decay, self.warm_up_size)

    def reset_dist_stat(self):
        self.dist_stat = DistStat(self.decay, self.warm_up_size)

    def reset_cum_stat(self):
        self.cum_stat = CumStat(self.decay, self.warm_up_size)

    def reset_unit_num_stat(self):
        self.unit_num_stat = UnitNumStat(self.decay, self.warm_up_size)


class MainPlayer(ActivePlayer):
    _name = "MainPlayer"

    def get_branch_opponent(self, historical_players, active_players, branch_probs_dict, pfsp_train_bot):
        branch_probs = branch_probs_dict[self._name]
        branches, branch_weights = list(branch_probs.keys()), branch_probs.values()
        branch = random.choices(branches, weights=branch_weights, k=1)[0]
        if branch == 'sp':
            main_players = [p for p in active_players.values() if isinstance(p, MainPlayer)]
            opponent_player = random.choice(main_players)
            if opponent_player != self and self.payoff.pfsp_winrate_info_dict.get(opponent_player.player_id, 0.5) < 0.3:
                hist_player_keys = [hist_player_id for hist_player_id, hist_player in historical_players.items() if
                                    hist_player.parent_id == opponent_player.player_id]
                if len(hist_player_keys) == 0:
                    hist_player_keys = [hist_player_id for hist_player_id, hist_player in historical_players.items() if
                                        hist_player.pipeline != 'bot']
                hist_player_weights = [self.payoff.pfsp_winrate_info_dict.get(player_id, 0.5)
                                       for player_id in hist_player_keys]
                hist_player_pfps_probs = pfsp(np.array(hist_player_weights), weighting='variance')
                opponent_player_id = random.choices(hist_player_keys, weights=hist_player_pfps_probs, k=1)[0]
                opponent_player = historical_players[opponent_player_id]
            home_team = [self]
            opponent_team = [opponent_player]
        elif branch == 'pfsp':
            home_team = [self]
            if pfsp_train_bot:
                hist_player_keys = list(historical_players.keys())
            else:
                hist_player_keys = [hist_player_id for hist_player_id, hist_player in historical_players.items() if
                                    hist_player.pipeline != 'bot']
            assert len(hist_player_keys) != 0
            hist_player_weights = [self.payoff.pfsp_winrate_info_dict.get(player_id, 0.5) for player_id in
                                   hist_player_keys]
            hist_player_pfps_probs = pfsp(np.array(hist_player_weights), weighting='squared')
            opponent_player_id = random.choices(hist_player_keys, weights=hist_player_pfps_probs, k=1)[0]
            opponent_player = historical_players[opponent_player_id]
            opponent_team = [opponent_player]
        elif branch == 'eval':
            home_team = [self]
            hist_player_keys = list(historical_players.keys())
            opponent_player_id = random.choice(hist_player_keys)
            opponent_player = historical_players[opponent_player_id]
            opponent_team = [opponent_player]
        else:
            print('Not implement such branch!')
            raise NotImplementedError
        return branch, home_team, opponent_team

    def is_trained_enough(self, historical_players, active_players, pfsp_train_bot=False, *args, **kwargs) -> bool:
        """
        Overview:
            Judge whether this player is trained enough for further operation
        Returns:
            - flag (:obj:`bool`): whether this player is trained enough
        """
        if self.snapshot_flag:
            self.snapshot_flag = False
            self.last_enough_step = self.total_agent_step
            return True
        step_passed = self.total_agent_step - self.last_enough_step
        if step_passed < self.one_phase_step / 2:
            return False
        if step_passed >= self.one_phase_step:
            self.last_enough_step = self.total_agent_step
            return True
        if pfsp_train_bot:
            hist_player_keys = list(historical_players.keys())
        else:
            hist_player_keys = [hist_player_id for hist_player_id, hist_player in historical_players.items() if
                                hist_player.pipeline != 'bot']

        hist_flags = [self.payoff.stat_info_record[player_id]['winrate'].val > self.strong_win_rate + 0.1 and \
                      self.payoff.stat_info_record[player_id]['winrate'].count >= self.warm_up_size for player_id in
                      hist_player_keys]
        if False not in hist_flags:
            return True

        active_player_keys = [player_id for player_id in active_players.keys() if player_id != self.player_id]
        opponent_keys = hist_player_keys + active_player_keys
        for player_id in opponent_keys:
            if player_id not in self.payoff.stat_info_record.keys():
                return False
            if self.payoff.stat_info_record[player_id]['winrate'].val > self.strong_win_rate and \
                    self.payoff.stat_info_record[player_id]['winrate'].count >= self.warm_up_size:
                continue
            else:
                return False
        self.last_enough_step = self.total_agent_step
        return True

    def reset_checkpoint(self, active_players, historical_players, new_player_id):
        return self.teacher_checkpoint_path


class ExploiterPlayer(ActivePlayer):
    _name = "ExploiterPlayer"
    _reset_prob = 0.25

    def get_branch_opponent(self, historical_players, active_players, branch_probs_dict, pfsp_train_bot):
        branch_probs = branch_probs_dict[self._name]
        branches, branch_weights = list(branch_probs.keys()), branch_probs.values()
        branch = random.choices(branches, weights=branch_weights, k=1)[0]
        hist_player_keys = list(historical_players.keys())

        if branch == 'pfsp':
            if pfsp_train_bot:
                hist_player_keys = list(historical_players.keys())
            else:
                hist_player_keys = [hist_player_id for hist_player_id, hist_player in historical_players.items() if
                                    hist_player.pipeline != 'bot']
            hist_player_weights = [self.payoff.pfsp_winrate_info_dict.get(player_id, 0.5) for player_id in
                                   hist_player_keys]
            hist_player_pfps_probs = pfsp(np.array(hist_player_weights), weighting='normal')
            opponent_player_id = random.choices(hist_player_keys, weights=hist_player_pfps_probs, k=1)[0]
            opponent_player = historical_players[opponent_player_id]
        elif branch == 'eval':
            opponent_player_id = random.choice(hist_player_keys)
            opponent_player = historical_players[opponent_player_id]
        else:
            print('Not implement such branch!')
            raise NotImplementedError
        home_team = [self]
        opponent_team = [opponent_player]
        return branch, home_team, opponent_team

    def is_trained_enough(self, historical_players, active_players, pfsp_train_bot=False, *args, **kwargs) -> bool:
        """
        Overview:
            Judge whether this player is trained enough for further operation
        Returns:
            - flag (:obj:`bool`): whether this player is trained enough
        """
        if self.snapshot_flag:
            self.snapshot_flag = False
            self.last_enough_step = self.total_agent_step
            return True
        step_passed = self.total_agent_step - self.last_enough_step
        if step_passed < self.one_phase_step / 2:
            return False
        if step_passed >= self.one_phase_step:
            self.last_enough_step = self.total_agent_step
            return True

        if pfsp_train_bot:
            hist_player_keys = list(historical_players.keys())
        else:
            hist_player_keys = [hist_player_id for hist_player_id, hist_player in historical_players.items() if
                                hist_player.pipeline != 'bot']

        opponent_keys = hist_player_keys
        for player_id in opponent_keys:
            if player_id not in self.payoff.stat_info_record.keys():
                return False
            if self.payoff.stat_info_record[player_id]['winrate'].val > self.strong_win_rate and \
                    self.payoff.stat_info_record[player_id]['winrate'].count >= self.warm_up_size:
                continue
            else:
                return False
        self.last_enough_step = self.total_agent_step
        return True

    def is_reset(self):
        if self.reset_flag:
            self.reset_flag = False
            return True
        p = np.random.uniform()
        if p < self._reset_prob:
            return True
        else:
            return False


class ExpertExploiterPlayer(ActivePlayer):
    _name = "ExpertExploiterPlayer"

    def __init__(self, checkpoint_path: str, player_id: str, pipeline: str, frac_id: int, z_path: str, z_prob: float, teacher_id: str, teacher_checkpoint_path: str, chosen_weight: float = 1, total_agent_step: int = 0, decay: float = 0.995, warm_up_size: int = 1000, min_win_rate_games: int = 200, total_game_count: int = 0, one_phase_step: int = 200000000, last_enough_step: int = 0, snapshot_times: int = 0, strong_win_rate: float = 0.7, payoff: Payoff = None, teammate_payoff: Payoff = None, opponent_payoff: Payoff = None, dist_stat: DistStat = None, cum_stat: CumStat = None, unit_num_stat: UnitNumStat = None, successive_model_path: str = None) -> None:
        super(ExpertExploiterPlayer, self).__init__(checkpoint_path, player_id, pipeline, frac_id, z_path, z_prob, teacher_id, teacher_checkpoint_path, chosen_weight=chosen_weight, total_agent_step=total_agent_step, decay=decay, warm_up_size=warm_up_size, min_win_rate_games=min_win_rate_games, total_game_count=total_game_count, one_phase_step=one_phase_step, last_enough_step=last_enough_step, snapshot_times=snapshot_times, strong_win_rate=strong_win_rate, payoff=payoff, teammate_payoff=teammate_payoff, opponent_payoff=opponent_payoff, dist_stat=dist_stat, cum_stat=cum_stat, unit_num_stat=unit_num_stat, successive_model_path=successive_model_path)
        assert isinstance(self.z_path, list)
        self.z_paths = self.z_path
        self.z_path = random.choice(self.z_paths)

    def get_branch_opponent(self, historical_players, active_players, branch_probs_dict, pfsp_train_bot):
        branch_probs = branch_probs_dict[self._name]
        branches, branch_weights = list(branch_probs.keys()), branch_probs.values()
        branch = random.choices(branches, weights=branch_weights, k=1)[0]
        hist_player_keys = list(historical_players.keys())

        if branch == 'pfsp':
            if pfsp_train_bot:
                hist_player_keys = list(historical_players.keys())
            else:
                hist_player_keys = [hist_player_id for hist_player_id, hist_player in historical_players.items() if
                                    hist_player.pipeline != 'bot']
            hist_player_weights = [self.payoff.pfsp_winrate_info_dict.get(player_id, 0.5) for player_id in
                                   hist_player_keys]
            hist_player_pfps_probs = pfsp(np.array(hist_player_weights), weighting='normal')
            opponent_player_id = random.choices(hist_player_keys, weights=hist_player_pfps_probs, k=1)[0]
            opponent_player = historical_players[opponent_player_id]
        elif branch == 'eval':
            opponent_player_id = random.choice(hist_player_keys)
            opponent_player = historical_players[opponent_player_id]
        else:
            print('Not implement such branch!')
            raise NotImplementedError
        home_team = [self]
        opponent_team = [opponent_player]
        return branch, home_team, opponent_team

    def is_trained_enough(self, historical_players, active_players, pfsp_train_bot=False, *args, **kwargs) -> bool:
        """
        Overview:
            Judge whether this player is trained enough for further operation
        Returns:
            - flag (:obj:`bool`): whether this player is trained enough
        """
        if self.snapshot_flag:
            self.snapshot_flag = False
            self.last_enough_step = self.total_agent_step
            return True
        step_passed = self.total_agent_step - self.last_enough_step
        if step_passed < self.one_phase_step / 2:
            return False
        if step_passed >= self.one_phase_step:
            self.last_enough_step = self.total_agent_step
            return True

        if pfsp_train_bot:
            hist_player_keys = list(historical_players.keys())
        else:
            hist_player_keys = [hist_player_id for hist_player_id, hist_player in historical_players.items() if
                                hist_player.pipeline != 'bot']

        opponent_keys = hist_player_keys
        for player_id in opponent_keys:
            if player_id not in self.payoff.stat_info_record.keys():
                return False
            if self.payoff.stat_info_record[player_id]['winrate'].val > self.strong_win_rate and \
                    self.payoff.stat_info_record[player_id]['winrate'].count >= self.warm_up_size:
                continue
            else:
                return False
        self.last_enough_step = self.total_agent_step
        return True

    def is_reset(self):
        self.z_path = random.choice(self.z_paths)
        return True

    def snapshot(self) -> HistoricalPlayer:
        """
        Overview:
            Generate a snapshot historical player from the current player, called in league manager's ``_snapshot``.
        Returns:
            - snapshot_player (:obj:`HistoricalPlayer`): new instantiated historical player
        Note:
            This method only generates a historical player object without saving the checkpoint, which should be
            completed by the interaction between coordinator and learner.
        """
        self.snapshot_times += 1
        h_player_id = self.player_id + f'H{self.snapshot_times}' + '_' + self.z_path.split('.')[0]
        h_player_path = self.checkpoint_path.split(
            '.pth')[0] + '_{}'.format(self.total_agent_step) + '.pth'
        return HistoricalPlayer(checkpoint_path=h_player_path, player_id=h_player_id, pipeline=self.pipeline,
                                frac_id=self.frac_id, z_path=self.z_path,z_prob=self.z_prob,
                                total_agent_step=self.total_agent_step, decay=self.decay,
                                warm_up_size=self.warm_up_size, min_win_rate_games=self.min_win_rate_games,
                                parent_id=self.player_id, )

    def reset_checkpoint(self, active_players, historical_players, new_player_id):
        main_players = [p for p in historical_players.keys() if 'MP' in p]
        main_players = sorted(main_players, key=lambda x: int(x.split('H')[-1]))
        reset_player = historical_players[main_players[-1]]
        return reset_player.checkpoint_path


class MainExploiterPlayer(ActivePlayer):
    _name = "MainExploiterPlayer"

    def get_branch_opponent(self, historical_players, active_players, branch_probs_dict, pfsp_train_bot):
        main_player_id = f'MP{self.player_id[-1]}'
        main_player = active_players[main_player_id]
        branch_probs = branch_probs_dict[self._name]
        branches, branch_weights = list(branch_probs.keys()), branch_probs.values()
        branch = random.choices(branches, weights=branch_weights, k=1)[0]
        if branch == 'vs_main':
            if self.payoff.pfsp_winrate_info_dict.get(main_player.player_id, 0.5) > 0.2:
                return branch, [self], [main_player]
            else:
                branch = 'pfsp'
        elif branch == 'eval':
            return 'vs_main_eval', [self], [main_player]
        if branch == 'pfsp':
            hist_player_keys = [hist_player_id for hist_player_id, hist_player in historical_players.items() if
                                hist_player.parent_id == main_player_id]
            hist_player_weights = [self.payoff.pfsp_winrate_info_dict.get(player_id, 0.5) for player_id in
                                   hist_player_keys]
            hist_player_pfps_probs = pfsp(np.array(hist_player_weights))
            opponent_player_id = random.choices(hist_player_keys, weights=hist_player_pfps_probs, k=1)[0]
            opponent_player = historical_players[opponent_player_id]
        else:
            print('Not implement such branch!')
            raise NotImplementedError
        home_team = [self]
        opponent_team = [opponent_player]
        return branch, home_team, opponent_team

    def is_trained_enough(self, historical_players, active_players, pfsp_train_bot=False, *args, **kwargs) -> bool:
        """
        Overview:
            Judge whether this player is trained enough for further operation
        Returns:
            - flag (:obj:`bool`): whether this player is trained enough
        """
        if self.snapshot_flag:
            self.snapshot_flag = False
            self.last_enough_step = self.total_agent_step
            return True
        step_passed = self.total_agent_step - self.last_enough_step

        if step_passed >= self.one_phase_step:
            self.last_enough_step = self.total_agent_step
            return True
        # main_player_id = f'MP{self.player_id[-1]}'
        main_players = [active_player for active_player in active_players.keys() if 'MP' in active_player]

        opponent_keys = main_players
        for player_id in opponent_keys:
            if player_id not in self.payoff.stat_info_record.keys():
                return False
            if self.payoff.stat_info_record[player_id]['winrate'].val > self.strong_win_rate and \
                    self.payoff.stat_info_record[player_id]['winrate'].count >= self.warm_up_size:
                continue
            else:
                return False
        self.last_enough_step = self.total_agent_step
        return True

    def is_reset(self):
        return True

class ExpertPlayer(ActivePlayer):
    _name = 'ExpertPlayer'
    def get_branch_opponent(self, historical_players, active_players, branch_probs_dict, pfsp_train_bot):
        branch_probs = branch_probs_dict[self._name]
        branches, branch_weights = list(branch_probs.keys()), branch_probs.values()
        branch = random.choices(branches, weights=branch_weights, k=1)[0]
        if branch == 'pfsp':
            home_team = [self]
            hist_player_keys = [hist_player_id for hist_player_id in historical_players.keys() if
                                   'EX' not in hist_player_id]
            assert len(hist_player_keys) != 0
            hist_player_weights = [self.payoff.pfsp_winrate_info_dict.get(player_id, 0.1) for player_id in
                                   hist_player_keys]
            hist_player_pfps_probs = pfsp(np.array(hist_player_weights), weighting='variance')
            opponent_player_id = random.choices(hist_player_keys, weights=hist_player_pfps_probs, k=1)[0]
            opponent_player = historical_players[opponent_player_id]
            opponent_team = [opponent_player]
        elif branch == 'eval':
            home_team = [self]
            hist_player_keys = list(historical_players.keys())
            opponent_player_id = random.choice(hist_player_keys)
            opponent_player = historical_players[opponent_player_id]
            opponent_team = [opponent_player]
        else:
            print('Not implement such branch!')
            raise NotImplementedError
        return branch, home_team, opponent_team

    def is_trained_enough(self, historical_players, active_players, pfsp_train_bot=False, *args, **kwargs) -> bool:
        """
        Overview:
            Judge whether this player is trained enough for further operation
        Returns:
            - flag (:obj:`bool`): whether this player is trained enough
        """
        if self.snapshot_flag:
            self.snapshot_flag = False
            self.last_enough_step = self.total_agent_step
            return True
        step_passed = self.total_agent_step - self.last_enough_step
        if step_passed >= self.one_phase_step:
            self.last_enough_step = self.total_agent_step
            return True
        else:
            return False


class AdaptiveEvolutionaryExploiterPlayer(ActivePlayer):
    _name = "AdaptiveEvolutionaryExploiterPlayer"
    _stat_keys = ActivePlayer._stat_keys + ['init_players',]
    def __init__(self, checkpoint_path: str, player_id: str, pipeline: str, frac_id: int, z_path: str, z_prob: float,
                 teacher_id: str, teacher_checkpoint_path: str, chosen_weight,
                 total_agent_step: int = 0,
                 decay: float = 0.995, warm_up_size: int = 1000, min_win_rate_games: int = 200,
                 total_game_count: int = 0,
                 one_phase_step: int = 2e8, last_enough_step: int = 0, snapshot_times: int = 0,
                 strong_win_rate: float = 0.7,
                 payoff: Payoff = None,
                 teammate_payoff: Payoff = None,
                 opponent_payoff: Payoff = None,
                 dist_stat: DistStat = None,
                 cum_stat: CumStat = None,
                 unit_num_stat: UnitNumStat = None,
                 successive_model_path: str = None,init_players=[]):
        super(AdaptiveEvolutionaryExploiterPlayer, self).__init__(checkpoint_path, player_id, pipeline, frac_id, z_path, z_prob,
                                           teacher_id, teacher_checkpoint_path, chosen_weight,
                                           total_agent_step, decay, warm_up_size, min_win_rate_games, total_game_count,
                                            one_phase_step, last_enough_step, snapshot_times,strong_win_rate,
                                           payoff, teammate_payoff, opponent_payoff, dist_stat,cum_stat, unit_num_stat, successive_model_path)

        self.init_players = init_players
        self.reset_prob = 0.25

    def get_branch_opponent(self, historical_players, active_players, branch_probs_dict, pfsp_train_bot):
        main_player_ids = [active_player for active_player in active_players.keys() if 'MP' in active_player]
        main_player_id = random.choice(main_player_ids)
        main_player = active_players[main_player_id]
        branch_probs = branch_probs_dict[self._name]
        branches, branch_weights = list(branch_probs.keys()), branch_probs.values()
        branch = random.choices(branches, weights=branch_weights, k=1)[0]
        if branch == 'vs_main':
            if self.payoff.pfsp_winrate_info_dict.get(main_player.player_id, 0.5) > 0.2:
                return branch, [self], [main_player]
            else:
                branch = 'pfsp'
        elif branch == 'eval':
            return 'vs_main_eval', [self], [main_player]
        if branch == 'pfsp':
            hist_player_keys = [hist_player_id for hist_player_id, hist_player in historical_players.items() if
                                hist_player.parent_id == main_player_id]
            hist_player_weights = [self.payoff.pfsp_winrate_info_dict.get(player_id, 0.5) for player_id in
                                   hist_player_keys]
            hist_player_pfps_probs = pfsp(np.array(hist_player_weights))
            opponent_player_id = random.choices(hist_player_keys, weights=hist_player_pfps_probs, k=1)[0]
            opponent_player = historical_players[opponent_player_id]
        else:
            print('Not implement such branch!')
            raise NotImplementedError
        home_team = [self]
        opponent_team = [opponent_player]
        return branch, home_team, opponent_team

    def is_trained_enough(self, historical_players, active_players, pfsp_train_bot=False, *args, **kwargs) -> bool:
        """
        Overview:
            Judge whether this player is trained enough for further operation
        Returns:
            - flag (:obj:`bool`): whether this player is trained enough
        """
        if self.snapshot_flag:
            self.snapshot_flag = False
            self.last_enough_step = self.total_agent_step
            return True
        step_passed = self.total_agent_step - self.last_enough_step

        if step_passed >= self.one_phase_step:
            self.last_enough_step = self.total_agent_step
            return True
        # main_player_id = f'MP{self.player_id[-1]}'
        main_players = [active_player for active_player in active_players.keys() if 'MP' in active_player]
        # if pfsp_train_bot:
        #     hist_player_keys = [hist_player_id for hist_player_id, hist_player in historical_players.items() if
        #                         hist_player.parent_id == main_player_id]
        # else:
        #     hist_player_keys = [hist_player_id for hist_player_id, hist_player in historical_players.items() if
        #                         hist_player.parent_id == main_player_id and hist_player.pipeline != 'bot']

        opponent_keys = main_players
        for player_id in opponent_keys:
            if player_id not in self.payoff.stat_info_record.keys():
                return False
            if self.payoff.stat_info_record[player_id]['winrate'].val > self.strong_win_rate and \
                    self.payoff.stat_info_record[player_id]['winrate'].count >= self.warm_up_size:
                continue
            else:
                return False
        self.last_enough_step = self.total_agent_step
        return True

    def is_reset(self):
        return True

    def reset_checkpoint(self,active_players, historical_players, new_player_id):
        main_players = [active_player for active_player in active_players.keys() if 'MP' in active_player]
        main_player = random.choice(main_players)
        p = random.random()
        if p < self.reset_prob:
            self.init_players.append(new_player_id)
            return self.teacher_checkpoint_path
        best_player_id = None
        best_winrate = 0.0
        best_idx = 0
        win_rate = self.payoff.stat_info_record[main_player]['winrate'].val
        if win_rate <= 0.5 and win_rate >= 0.2:
            best_winrate = win_rate
            best_player_id = new_player_id
            best_idx = -1
            
        for idx,player_id in enumerate(self.init_players):
            win_rate = 1 - active_players[main_player].payoff.stat_info_record[player_id]['winrate'].val
            if win_rate <= 0.5 and win_rate >= 0.2 and win_rate > best_winrate:
                best_player_id = player_id
                best_idx = idx
        if best_player_id is not None and best_idx!=-1:
            del self.init_players[best_idx]
        if new_player_id is not None and best_idx!=-1:
            self.init_players.append(new_player_id)
        return historical_players[best_player_id].checkpoint_path
