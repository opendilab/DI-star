import numpy as np
import random
def main_func(player, all_players, historical_players, 
              active_players, main_exploiter_players, cfg):
    p = np.random.uniform()
    if p<0.35:
        #self-play
        return player
    elif p<0.85:
        #sample from all_players
        all_player_keys = list(all_players.keys())
        all_player_weights = [player.payoff.pfsp_winrate_info_dict.get(player_id,0.5) for player_id in all_player_keys]
        all_player_pfps_probs = pfsp(np.array(all_player_weights))
        all_player_id = random.choices(all_player_keys, weights=all_player_pfps_probs,k=1)[0]
        return all_players[all_player_id]
    else:
        if len(main_exploiter_players)!=0:
            players = main_exploiter_players
        else:
            players = all_players
        player_keys = list(players.keys())
        player_weights = [player.payoff.pfsp_winrate_info_dict.get(player_id,0.5) for player_id in player_keys]
        player_pfps_probs = pfsp(np.array(player_weights))
        player_id = random.choices(player_keys, weights=player_pfps_probs,k=1)[0]
        return players[player_id] 



    

def main_exploiter_func(player, all_players, historical_players, 
              active_players, main_exploiter_players, cfg):
    if player.payoff.pfsp_winrate_info_dict.get(active_players[0],0.5) < 0.2:
        players = historical_players
    else:
        p = np.random.uniform()
        if p<0.5:
            return active_players[0]
        else:
            players = historical_players
    player_keys = list(players.keys())
    player_weights = [player.payoff.pfsp_winrate_info_dict.get(player_id,0.5) for player_id in player_keys]
    player_pfps_probs = pfsp(np.array(player_weights))
    player_id = random.choices(player_keys, weights=player_pfps_probs,k=1)[0]
    return players[player_id]

def exploiter_func(player, all_players, historical_players, 
              active_players, main_exploiter_players, cfg):
    players = all_players
    player_keys = list(players.keys())
    player_weights = [player.payoff.pfsp_winrate_info_dict.get(player_id,0.5) for player_id in player_keys ]
    player_pfps_probs = pfsp(np.array(player_weights))
    player_id = random.choices(player_keys, weights=player_pfps_probs,k=1)[0]
    return players[player_id]



def pfsp(win_rates: np.ndarray, weighting: str = 'variance') -> np.ndarray:
    """
    Overview:
        Prioritized Fictitious Self-Play algorithm.
        Process win_rates with a weighting function to get priority, then calculate the selection probability of each.
    Arguments:
        - win_rates (:obj:`np.ndarray`): a numpy ndarray of win rates between one player and N opponents, shape(N)
        - weighting (:obj:`str`): pfsp weighting function type, refer to ``weighting_func`` below
    Returns:
        - probs (:obj:`np.ndarray`): a numpy ndarray of probability at which one element is selected, shape(N)
    """
    weighting_func = {
        'squared': lambda x: (1 - x) ** 2,
        'variance': lambda x: x * (1 - x),
        'normal': lambda x: np.minimum(0.5,1-x),
    }
    if weighting in weighting_func.keys():
        fn = weighting_func[weighting]
    else:
        raise KeyError("invalid weighting arg: {} in pfsp".format(weighting))

    assert isinstance(win_rates, np.ndarray)
    assert win_rates.shape[0] >= 1, win_rates.shape
    # all zero win rates case, return uniform selection prob
    if win_rates.sum() < 1e-8:
        return np.full_like(win_rates, 1.0 / len(win_rates))
    fn_win_rates = fn(win_rates)
    probs = fn_win_rates / fn_win_rates.sum()
    return probs