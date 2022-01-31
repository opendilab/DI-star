from collections import defaultdict

import dill
import torch
from tabulate import tabulate
import lz4.frame


class ELORating:
    ELO_RESULT_WIN = 1
    ELO_RESULT_LOSS = -1
    ELO_RESULT_DRAW = 0

    def __init__(self, K=44, init_elo=1000, minimum_games=0):
        self.K = K
        self.init_elo = init_elo
        self.elos = defaultdict(int)
        self.wins = defaultdict(lambda: defaultdict(int))
        self.games = defaultdict(lambda: defaultdict(int))
        self.game_count = 0
        self.minimum_games = minimum_games

    def update(self, player1, player2, result):
        player1_win_prob = 1 / (1 + pow(10, (self.elos[player2] - self.elos[player1]) / 400))

        if result == self.ELO_RESULT_WIN:
            self.wins[player1][player2] += 1
            score_adjust = 1
        elif result == self.ELO_RESULT_LOSS:
            self.wins[player2][player1] += 1
            score_adjust = 0
        else:
            score_adjust = 0.5

        self.games[player1][player2] += 1
        self.games[player2][player1] += 1

        self.elos[player1] = self.elos[player1] + self.K * (score_adjust - player1_win_prob)
        self.elos[player2] = self.elos[player2] - self.K * (score_adjust - player1_win_prob)
        self.game_count += 1

    def elo_text(self, start_from_zero=True):
        elos = {k: v + self.init_elo for k, v in self.elos.items()}
        if start_from_zero:
            minimum_elo = min(list(elos.values()))
            elos = {k: v - minimum_elo for k, v in elos.items()}
        iterative_table_data = sorted(elos.items(), key=lambda item: item[1])

        player_num = len(self.elos)
        mmr = torch.ones(player_num) * self.init_elo
        payoff = torch.zeros(player_num, player_num)
        mask = torch.zeros(player_num, player_num, dtype=torch.bool)
        player_list = list(self.elos.keys())
        mmr_range = torch.arange(2000).unsqueeze(dim=1).float()
        player_games = defaultdict(int)
        player_done = defaultdict(int)

        for idx1, p1 in enumerate(player_list):
            for idx2, p2 in enumerate(player_list):
                if self.games[p1][p2] != 0:
                    payoff[idx1][idx2] = self.wins[p1][p2] / self.games[p1][p2]
                    player_games[p1] += self.games[p1][p2]
                if self.games[p1][p2] > self.minimum_games:
                    mask[idx1][idx2] = 1.
                    player_done[p1] += 1
                if 'bot10' in p1 or 'bot10' in p2:
                    mask[idx1][idx2] = 0.
                    mask[idx2][idx1] = 0.
                if idx1 == idx2:
                    mask[idx1][idx2] = 0.

        payoff.clamp_(min=0.1, max=0.9)
        log_payoff = torch.log(payoff)
        for e in range(1000):
            new_mmr = torch.ones(player_num) * self.init_elo
            for i in range(player_num):
                if 'bot10' in player_list[i]:
                    continue
                mmr_gap = mmr - mmr_range
                mmr_gap /= 400
                prob = (1. / (1 + torch.exp(mmr_gap)))
                log_prob = torch.log(prob)
                prob = ((log_prob * mask[i]).sum(dim=-1) - (log_payoff[i] * mask[i]).sum(dim=-1)).abs()
                new_mmr[i] = mmr_range[prob.argmin(dim=0)]

            mmr_delta = new_mmr - mmr
            mmr = mmr + mmr_delta * 0.1
            mmr_range = mmr_range + mmr_delta.mean() * 0.1
            if mmr_delta.abs().sum() < 1 or (mmr_delta - mmr_delta.mean()).abs().sum() / player_num < 3:
                break

        if start_from_zero:
            minimum_elo = mmr.min().item()
        else:
            minimum_elo = 0.
        elos = {k: mmr[i].item() - minimum_elo for i, k in enumerate(player_list)}
        winrate_table_data = sorted(elos.items(), key=lambda item: item[1])

        table_data = []
        for i in range(len(winrate_table_data)):
            player_id = winrate_table_data[i][0]
            table_data.append(list(iterative_table_data[i]) + list(winrate_table_data[i]) + [
                '{}({}/{})'.format(player_games[player_id], player_done[player_id], player_num - 1)])
        headers = ['player_id', 'iterative_elo', 'player_id', 'winrate_elo', 'games(done)']
        table_text = tabulate(table_data, headers=headers, tablefmt='grid', )
        return '\n' + table_text

    @property
    def payoff(self):
        player_list = list(self.elos.keys())
        player_num = len(self.elos)
        payoff = torch.zeros(player_num, player_num)
        for idx1, p1 in enumerate(player_list):
            for idx2, p2 in enumerate(player_list):
                if self.games[p1][p2] != 0:
                    payoff[idx1][idx2] = self.wins[p1][p2] / self.games[p1][p2]
        return payoff

    def winrate(self, player_id, filter_id=''):
        player_list = list(self.elos.keys())
        table = []
        for p in player_list:
            if p != player_id and filter_id in p:
                table.append((p, self.wins[player_id][p] / (self.games[player_id][p] + 1e-9), self.games[player_id][p]))
        table = sorted(table, key=lambda item: item[1])
        headers = ['player_id', 'winrate', 'games']
        table_text = tabulate(table, headers=headers, tablefmt='grid', )
        table_text = '======================{}======================\n'.format(player_id + ' ' + 'winrate') + table_text
        return table_text


if __name__ == '__main__':
    f = open(r'D:\remote\cmp_league.resume.2022-01-28-14-38-02', 'rb')
    data = dill.loads(lz4.frame.decompress(f.read()))
    print(data['elo_ratings'].winrate('bot10', filter_id='MP0H'))
    print(data['elo_ratings'].winrate('bot7', filter_id='MP0H'))
    print(data['elo_ratings'].winrate('MP0H9_0Z'))
    print(data['elo_ratings'].winrate('MP0H9'))
    elo = data['elo_ratings']
    elo.init_elo = 1000
    print(elo.elo_text(start_from_zero=True))
    exit()

    e = ELORating(init_elo=1000)
    e.update('haha', 'hehe', -1)
    e.update('haha', 'hehe', -1)
    e.update('haha', 'hehe', 1)
    e.update('hehe', 'xixi', -1)
    e.update('hehe', 'xixi', -1)
    e.update('hehe', 'xixi', 1)
    e.update('haha', 'xixi', 1)
    e.update('xixi', 'haha', 1)
    e.update('xixi', 'haha', 1)
    e.update('xixi', 'gigi', 1)
    e.update('gigi', 'xixi', 1)
    e.update('gigi', 'xixi', 1)
    print(e.elo_text(start_from_zero=True))
    print(e.winrate('haha'))
    print(e.winrate('hehe'))
    print(e.winrate('xixi'))
    print(e.winrate('gigi'))
    data = dill.dumps(e)
