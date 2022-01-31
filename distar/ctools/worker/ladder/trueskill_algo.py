import itertools
import math

import trueskill
from trueskill import Rating, rate, quality, BETA


def win_probability(team1, team2):
    delta_mu = sum(r.mu for r in team1) - sum(r.mu for r in team2)
    sum_sigma = sum(r.sigma ** 2 for r in itertools.chain(team1, team2))
    size = len(team1) + len(team2)
    denom = math.sqrt(size * (BETA * BETA) + sum_sigma)
    ts = trueskill.global_env()
    return ts.cdf(delta_mu / denom)


if __name__ == '__main__':
    r1 = Rating()  # 1P's skill
    r2 = Rating()  # 2P's skill
    r3 = Rating()  # 3P's skill
    r4 = Rating()  # 3P's skill
    t1 = [r1, ]  # Team A contains 1P and 2P
    t2 = [r3, r4]  # Team B contains 3P and 4P

    print('{:.1%} chance to draw'.format(quality([t1, t2])))
    # 13.5% chance to draw
    (new_r1,), (new_r3, new_r4) = rate([t1, t2], weights=[(2,), (0.5, 1.5)], ranks=[0, 1])
    print(win_probability([new_r1, new_r1], [new_r3, new_r4]))
    print(new_r3, new_r4)
    print(win_probability((new_r3,), (new_r4,)))
