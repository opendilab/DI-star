import random
from collections import defaultdict, OrderedDict
from functools import partial
from typing import DefaultDict

from tabulate import tabulate

from distar.ctools.utils.log_helper import EmaMeter

def type_template(decay,warm_up_size):
    return {0: EmaMeter(decay, warm_up_size),1: EmaMeter(decay, warm_up_size),2: EmaMeter(decay, warm_up_size),3: EmaMeter(decay, warm_up_size),}
class CumStat:
    """
    Overview:
        Payoff data structure to record historical match result, each player owns one specific payoff
    Interface:
        __init__, __getitem__, add_player, update
    Property:
        players
    """
    # data_keys = ['barracks_lv1', 'barracks_lv2', 'barracks_lv3',
    #              'vehicle_lv1', 'vehicle_lv2', 'vehicle_lv3',
    #              's_vehicle_lv1', 's_vehicle_lv2', 's_vehicle_lv3',
    #              'plane_lv1', 'plane_lv2', 'plane_lv3',
    #              'boat_lv1', 'boat_lv2', 'boat_lv3',
    #              'airattack_lv1',
    #              '指挥中心lv1', '指挥中心lv2', '指挥中心lv3', '指挥中心lv4',
    #              '施工工地', '发电站', '补给中心',
    #              '兵营lv1', '兵营lv2', '兵营lv3',
    #              '突击车工厂lv1', '突击车工厂lv2', '突击车工厂lv3',
    #              '特种车工厂lv1', '特种车工厂lv2', '特种车工厂lv3',
    #              '空军工厂lv1', '空军工厂lv2', '空军工厂lv3', '机场',
    #              '船坞lv1', '船坞lv2', '船坞lv3',
    #              '核弹', '围墙',
    #              '碉堡', '炮塔', '防空塔', '海防塔']
    not_use_keys = ['z_type','unit_num','step','winloss','agent_iters','bo_reward','cum_reward','bo_len','dist/bo','dist/bo_location','dist/cum','opponent_id','player_id','race','race_id']
    meaning_mapping = {
        'out/not': 0,
        'out/done': 1,
        'in/not': 2,
        'in/done': 3,
    }

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
        self._stat_info_record = defaultdict(
            partial(self._stat_template, self._decay, self._warm_up_size))

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
        for item, val in stat_info.items():
            if item not in self.not_use_keys:
                if stat_info[item] >= 0:
                    self._stat_info_record[frac_id][item][int(stat_info['z_type'])].update(val)
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
            stat_info_dict[frac_id] = {item: {i:stat_info[item][i].val for i in range(4)} for item in stat_info.keys() if item != 'game_count'}
        return stat_info_dict

    def get_text(self):
        info_text = ''
        for frac_id_show in range(1, 4):
            frac_side_keys_list = sorted(
                {(frac_id, side_id) for (frac_id, side_id) in self._stat_info_record.keys() if frac_id == frac_id_show})
            if not len(frac_side_keys_list):
                continue
            headers = ['type'] + [f'f{frac_id}_s{side_id}' for (frac_id, side_id) in frac_side_keys_list]
            table_data = []
            game_count_line_data = ['game_count'] + [str(int(self._stat_info_record[k]['game_count'])) for k in
                                                     frac_side_keys_list]
            table_data.append(game_count_line_data)
            for item in self.data_keys:
                line_data = [item] + [','.join([f'{self._stat_info_record[k][item][idx].val:.2f}' for idx in range(4)])
                                      for k in frac_side_keys_list]
                table_data.append(line_data)
            try:
                table_text = "\n" + tabulate(table_data, headers=headers, tablefmt='grid',
                                             stralign='left', numalign='left')
                info_text += table_text
            except:
                pass
        return info_text

    @staticmethod
    def _stat_template(decay, warm_up_size):
        template = defaultdict(partial(type_template,decay,warm_up_size))
        # template = DefaultDict(partial(EmaMeter,decay,warm_up_size))
        template['game_count'] = 0
        return template


if __name__ == '__main__':
    cum_stat = CumStat(decay=0.999, warm_up_size=1000)

    for data_num in range(100):
        # frac_id =  1
        frac_id = random.choice(range(1, 4))
        for side_id in range(1, 7):
            stat_info = {k: random.randint(0, 4) for k in cum_stat.data_keys}
            cum_stat.update(frac_id, side_id, stat_info)
    print(cum_stat.get_text())
