from collections import defaultdict

from tabulate import tabulate

from distar.ctools.utils.log_helper import EmaMeter
from functools import partial



class DistStat:
    """
    Overview:
        Payoff data structure to record historical match result, each player owns one specific payoff
    Interface:
        __init__, __getitem__, add_player, update
    Property:
        players
    """
    data_keys = ['dist/bo','dist/cum','bo_reward','cum_reward','bo_len','dist/bo_location']
    def __init__(self, decay, warm_up_size) -> None:
        """
        Overview: Initialize payoff
        Arguments:
            - home_id (:obj:`str`): home player id
            - decay (:obj:`float`): update step decay factor
            - min_win_rate_games (:obj:`int`): min games for win rate calculation
        """
        self._decay = decay
        self._warm_up_size = warm_up_size
        self._stat_info_record = defaultdict(partial(self._stat_template,self._decay,self._warm_up_size,self.data_keys))

    def update(self, frac_id, stat_info) -> bool:
        """
        Overview:
            Update payoff with a match_info
        Arguments:
            - match_info (:obj:`dict`): a dict contains match information,
                owning at least 3 keys('home_id', 'away_id', 'result')
        Returns:
            - result (:obj:`bool`): whether update is successful
        """
        self._stat_info_record[frac_id]['game_count'] += 1
        for item in self.data_keys:
            if stat_info[item] is not None:
                self._stat_info_record[frac_id][item].update(stat_info[item])
        return True

    @property
    def stat_info_record(self):
        return self._stat_info_record

    @property
    def game_count(self):
        return {frac_id: val['game_count'] for frac_id, val in
                self.stat_info_record.items()}

    @property
    def stat_info_dict(self):
        stat_info_dict = {}
        for frac_id, stat_info in self.stat_info_record.items():
            stat_info_dict[frac_id] = {item: stat_info[item].val for item in self.data_keys}
        return stat_info_dict

    def get_text(self):
        headers = ['frac_id'] + self.data_keys + ['game_count']
        table_data = []
        for frac_id, stat_info in sorted(self.stat_info_record.items()):
            line_data = [frac_id] + [stat_info[item].val for item in self.data_keys] +\
                        [stat_info['game_count']]
            table_data.append(line_data)
        table_text = "\n" + tabulate(table_data, headers=headers, tablefmt='grid')
        return table_text

    @staticmethod
    def _stat_template(decay,  warm_up_size, data_keys):
        template = {item: EmaMeter(decay,warm_up_size) for item in data_keys}
        template['game_count'] = 0
        return template

if __name__ == '__main__':
    dist_stat = DistStat(decay=0.999,warm_up_size=1000)
    import pickle

    x = pickle.dumps(dist_stat)
    y = pickle.loads(x)
    print(y)
    print(y.get_text())
