import argparse

import torch
from tabulate import tabulate


def get_args():
    parser = argparse.ArgumentParser(description="ladder_payoff")
    parser.add_argument("--path", "-p", required=False, default='/home/shenghan/ftp/league.resume.2021-11-08-17-15-39', type=str, help='abs/league/resume/path')
    return parser.parse_args()


def get_ladder_payoff(args):
    filepath = '/home/shenghan/ftp/league.resume.2021-11-08-17-15-39'

    resume_data = torch.load(filepath)

    # active_player = resume_data[0]
    hist_player = resume_data['historical_players']
    # hist_player_keys = [ k for k in hist_player.keys() if '0k' in k]
    hist_player_keys = list(hist_player.keys())
    short_hist_player_keys = [k.replace('baseline_', '') for k in hist_player_keys]

    headers = [''] + short_hist_player_keys

    table_data = []
    for idx, hist_player_id in enumerate(hist_player_keys):
        hist_player_winrate = [short_hist_player_keys[idx]]
        for opponent_hist_player_id in hist_player_keys:
            if opponent_hist_player_id == hist_player_id:
                hist_player_winrate.append((0.5, 0))
                # hist_player_winrate.append((0.5,0.5, 0))
                # hist_player_winrate.append(0.5)
            else:
                winrate_info = resume_data['historical_players'][hist_player_id].payoff.stat_info_record[opponent_hist_player_id][
                    'winrate']
                oppo_winrate_info = resume_data['historical_players'][opponent_hist_player_id].payoff.stat_info_record[hist_player_id]['winrate']
                hist_player_winrate.append(('{:.2f}'.format(winrate_info.val),winrate_info.count))
                # hist_player_winrate.append(('{:.2f}'.format(winrate_info.val),'{:.2f}'.format(oppo_winrate_info.val), winrate_info.count))
                # hist_player_winrate.append('{:.2f}'.format(winrate_info.val))
        table_data.append(hist_player_winrate)
    table_text = "\n" + tabulate(table_data, headers=headers, tablefmt='grid', colalign='left', stralign='left',
                                 numalign='left', )
    print(table_text)
    return table_text

if __name__ == '__main__':
    args = get_args()
    get_ladder_payoff(args)