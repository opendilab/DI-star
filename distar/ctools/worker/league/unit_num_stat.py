from collections import defaultdict

from tabulate import tabulate

from distar.ctools.utils.log_helper import EmaMeter
from functools import partial




class UnitNumStat:
    """
    Overview:
        Payoff data structure to record historical match result, each player owns one specific payoff
    Interface:
        __init__, __getitem__, add_player, update
    Property:
        players
    """
    data_keys = [
        'barracks_lv1',
        'barracks_lv2',
        'barracks_lv3',
        'vehicle_lv1',
        'vehicle_lv2',
        'vehicle_lv3',
        's_vehicle_lv1',
        's_vehicle_lv2',
        's_vehicle_lv3',
        'plane_lv1',
        'plane_lv2',
        'plane_lv3',
        'boat_lv1',
        'boat_lv2',
        'boat_lv3',
        'airattack_lv1',
    ]
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

    def update(self, frac_id, side_id, stat_info) -> bool:
        """
        Overview:
            Update payoff with a match_info
        Arguments:
            - match_info (:obj:`dict`): a dict contains match information,
                owning at least 3 keys('home_id', 'away_id', 'result')
        Returns:
            - result (:obj:`bool`): whether update is successful
        """
        total_unit_num = stat_info.pop('total_unit_num')
        if total_unit_num <= 0:
            return
        for item in self.data_keys:
            self._stat_info_record[(frac_id, side_id)][item].update(stat_info.get(item,0)/total_unit_num)
        return True

    @property
    def stat_info_record(self):
        return self._stat_info_record

    @property
    def stat_info_dict(self):
        stat_info_dict = {}
        for (frac_id, side_id), stat_info in self.stat_info_record.items():
            stat_info_dict[f'frac{frac_id}_side{side_id}'] = {item: stat_info[item].val for item in self.data_keys}
        return stat_info_dict

    def get_text(self):
        unit_num_keys_list = sorted(self._stat_info_record.keys())
        headers = ['type'] + [f'f{frac_id}_s{side_id}' for (frac_id, side_id) in unit_num_keys_list ]
        table_data = []
        game_count_line_data = ['game_count'] + [str(int(self._stat_info_record[k]['barracks_lv1'].count)) for k in unit_num_keys_list ]
        table_data.append(game_count_line_data)
        for item in self.data_keys:
            line_data = [item] + ['{:.3f}'.format(self._stat_info_record[k][item].val) for k in unit_num_keys_list ]
            table_data.append(line_data)
        try:
            table_text = "\n" + tabulate(table_data, headers=headers, tablefmt='grid',colalign='left',stralign='left',numalign='left',)
            return table_text
        except:
            return ''
    @staticmethod
    def _stat_template(decay,  warm_up_size, data_keys):
        return {item: EmaMeter(decay,warm_up_size) for item in data_keys}

if __name__ == '__main__':
    unit_stat = UnitNumStat(decay=0.999,warm_up_size=1000)
    import pickle
    x= pickle.dumps(unit_stat)
    y = pickle.loads(x)
    print(y)