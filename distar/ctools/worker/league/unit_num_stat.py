from collections import defaultdict
from functools import partial

from distar.ctools.utils.log_helper import EmaMeter
from tabulate import tabulate


class UnitNumStat:
    """
    Overview:
        Payoff data structure to record historical match result, each player owns one specific payoff
    Interface:
        __init__, __getitem__, add_player, update
    Property:
        players
    """
    data_keys_dict = {0: [
        'BroodLord',
        'Lurker',
        'OverlordTransport',
        'Overseer',
        'Ravager',
        'Baneling',
        'Corruptor',
        'Drone',
        'Hydralisk',
        'Infestor',
        'Mutalisk',
        'Overlord',
        'Queen',
        'Roach',
        'SwarmHost',
        'Ultralisk',
        'Viper',
        'Zergling'
    ],
        1: [
            'Banshee',
            'Battlecruiser',
            'Cyclone',
            'Ghost',
            'Hellbat',
            'Hellion',
            'Liberator',
            'Marauder',
            'Marine',
            'Medivac',
            'Raven',
            'Reaper',
            'SCV',
            'SiegeTank',
            'Thor',
            'VikingFighter',
            'WidowMine'
        ],
        2: [
            'Archon',
            'Mothership',
            'Adept',
            'Carrier',
            'Colossus',
            'DarkTemplar',
            'Disruptor',
            'HighTemplar',
            'Immortal',
            'MothershipCore',
            'Mothership',
            'Observer',
            'Oracle',
            'Phoenix',
            'Probe',
            'Sentry',
            'Stalker',
            'Tempest',
            'VoidRay',
            'Adept',
            'DarkTemplar',
            'HighTemplar',
            'WarpPrism',
            'Sentry',
            'Stalker',
            'Zealot',
            'Zealot'
        ]
    }

    def __init__(self, decay, warm_up_size, frac_id=0) -> None:
        """
        Overview: Initialize payoff
        Arguments:
            - home_id (:obj:`str`): home player id
            - decay (:obj:`float`): update step decay factor
            - min_win_rate_games (:obj:`int`): min games for win rate calculation
        """
        self.decay = decay
        self.warm_up_size = warm_up_size
        self.frac_id = frac_id
        self.data_keys = self.data_keys_dict[self.frac_id]
        self.reset()

    def update(self, stat_info) -> bool:
        """
        Overview:
            Update payoff with a match_info
        Arguments:
            - match_info (:obj:`dict`): a dict contains match information,
                owning at least 3 keys('home_id', 'away_id', 'result')
        Returns:
            - result (:obj:`bool`): whether update is successful
        """
        self.stat_info_record['game_count'] += 1
        max_unit_num = stat_info.pop('max_unit_num')
        if max_unit_num <= 0:
            return True
        for item in self.data_keys:
            self.stat_info_record[item].update(stat_info.get(item, 0) / max_unit_num)
        return True

    @property
    def game_count(self):
        return self.stat_info_record['game_count']

    def stat_info_dict(self):
        stat_info_dict = {k: self.stat_info_record[k].val for k in self.data_keys}
        return stat_info_dict

    def get_text(self):
        headers = [item[:5] for item in self.data_keys] + ['game_count']
        table_data = []
        line_data = [f"{self.stat_info_record[k].val:.3f}" for k in self.data_keys] + \
                    [self.stat_info_record['game_count']]
        table_data.append(line_data)
        table_text = "\n" + tabulate(table_data, headers=headers, tablefmt='grid')
        return table_text

    @staticmethod
    def stat_template(decay, warm_up_size, data_keys):
        template = {item: EmaMeter(decay, warm_up_size) for item in data_keys}
        template['game_count'] = 0
        return template

    def reset(self):
        self.stat_info_record = self.stat_template(self.decay, self.warm_up_size, self.data_keys)

if __name__ == '__main__':
    stat = UnitNumStat(decay=0.999, warm_up_size=1000)
    import pickle
    import random
    for i in range(100):
        stat_info = {k: random.random() for k in stat.data_keys}
        stat_info['max_unit_num'] = max(stat_info.values())
        stat.update(stat_info)
    print(stat.get_text())
    x = pickle.dumps(stat)
    y = pickle.loads(x)
    print(y)
