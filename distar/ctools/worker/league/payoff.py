from collections import defaultdict
from functools import partial

from tabulate import tabulate

from distar.ctools.utils.log_helper import MoveAverageMeter


class Payoff:
    """
    Overview:
        Payoff data structure to record historical match result, each player owns one specific payoff
    Interface:
        __init__, __getitem__, add_player, update
    Property:
        players
    """
    data_keys = ['winrate', 'game_steps', 'game_iters', 'game_duration']

    def __init__(self, decay: float = 0.999, warm_up_size: int = 1000, min_win_rate_games=1000) -> None:
        """
        Overview: Initialize payoff
        Arguments:
            - home_id (:obj:`str`): home player id
            - decay (:obj:`float`): update step decay factor
            - min_win_rate_games (:obj:`int`): min games for win rate calculation
        """
        self._decay = decay
        self._warm_up_size = warm_up_size
        self._min_win_rate_games = min_win_rate_games
        self._stat_info_record = defaultdict(
            partial(self._stat_template, self._decay, self._warm_up_size, self.data_keys))

    def win_rate_opponent(self, opponent_id, min_win_rate_games=True) -> float:
        """
        Overview:
            Get win rate against an opponent player
        Arguments:
            - player (:obj:`Player`): the opponent player to calculate win rate
        Returns:
            - win rate (:obj:`float`): float win rate value. \
                Only when total games is no less than ``self._min_win_rate_games``, \
                can the win rate be calculated according to [win, draw, loss, game], or return 0.5 by default.
        """
        # not enough game record case
        if (self._stat_info_record[opponent_id]['winrate'].count < self._min_win_rate_games) and min_win_rate_games:
            return 0.5
        else:
            return self._stat_info_record[opponent_id]['winrate'].val

    def update(self, opponent_id, stat_info) -> bool:
        """
        Overview:
            Update payoff with a match_info
        Arguments:
            - match_info (:obj:`dict`): a dict contains match information,
                owning at least 3 keys('home_id', 'away_id', 'result')
        Returns:
            - result (:obj:`bool`): whether update is successful
        """
        # check
        for item in self.data_keys:
            self._stat_info_record[opponent_id][item].update(stat_info[item])
        return True

    @property
    def pfsp_winrate_info_dict(self, ):
        return {p: self.win_rate_opponent(p, min_win_rate_games=True) for p in self._stat_info_record}

    @property
    def stat_info_dict(self, ):
        stat_info_dict = {}
        for opponent_id, stat_info in self.stat_info_record.items():
            stat_info_dict[opponent_id] = {item: stat_info[item].val for item in self.data_keys}
        return stat_info_dict

    @property
    def stat_info_record(self):
        return self._stat_info_record

    @property
    def game_count(self):
        return {opponent_id: val['winrate'].count for opponent_id, val in self._stat_info_record.items()}

    @staticmethod
    def _stat_template(decay, warm_up_size, data_keys):
        return {item: MoveAverageMeter(warm_up_size, ) for item in data_keys}

    def get_text(self):
        headers = ["opponent"] + self.data_keys + ['game_count']
        table_data = []
        for opponent_id, stat_info in sorted(self._stat_info_record.items()):
            line_data = [opponent_id] + [stat_info[item].val for item in self.data_keys] + \
                        [stat_info['winrate'].count]
            table_data.append(line_data)
        table_text = "\n" + tabulate(table_data, headers=headers, tablefmt='grid')
        return table_text
