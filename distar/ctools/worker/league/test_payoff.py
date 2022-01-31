import numpy
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
resume_data_path = '/mnt/lustre/shenziju/league.resume.2021-11-01-11-55-07'
resume_data = torch.load(resume_data_path)

hist_players = resume_data['historical_players']
hist_ids = hist_players.keys()
print(hist_ids)
data = {}
###full payoff matrix
for player_id1 in hist_ids:
    win_rate = []
    for player_id2 in hist_ids:
        if player_id1 == player_id2:
            win_rate.append(0.5)
        else:
            if hist_players[player_id1].payoff._stat_info_record[player_id2]['winrate'].count <100:
                print(player_id1,player_id2) 
            win_rate.append(hist_players[player_id1].payoff._stat_info_record[player_id2]['winrate'].val)
    data[player_id1]=win_rate
win_rate_payoff = pd.DataFrame(data,index = hist_ids,columns=hist_ids)
print(win_rate_payoff)
winrate_payoff = win_rate_payoff.to_numpy()
winrate_value = winrate_payoff - 0.5

u,s,v = np.linalg.svd(winrate_value)
print(s)
cnt = {'1':0,'0.1':0}
for i in range(s.shape[0]):
    if s[i]>=1:
        cnt['1']+=1
        cnt['0.1']+=1
    elif s[i]>=0.1:
        cnt['0.1']+=1
print(f"sigular value >1 :{cnt['1']},{cnt['1']/s.shape[0]*100}%")
print(f"sigular value >0.1 :{cnt['0.1']},{cnt['0.1']/s.shape[0]*100}%")

def calc_embedding(value_payoff,v):
    v_2 = v[0:2,:]
    embedding_payoff = np.matmul(value_payoff, v_2.T) 
    return embedding_payoff

### calc first two dim embedding
# embedding_payoff = calc_embedding(winrate_value, v)
# x_max = max(abs(embedding_payoff[:,0]))+1.0
# y_max = max(abs(embedding_payoff[:,1]))+1.0
# avg_winrate = win_rate_payoff.mean(axis=1)
# plt.scatter(embedding_payoff[:,0],embedding_payoff[:,1],c=avg_winrate,marker='.')
# plt.axis([-x_max,x_max,-y_max,y_max])
# plt.colorbar()
# plt.savefig('agent_mode.jpg')

def plot_scatter_image(win_payoff,player_ids,save_path='agent_mode.jpg',use_annotate=False):
    small_payoff = win_payoff.loc[player_ids][player_ids]
    small_payoff_np = small_payoff.to_numpy()
    small_payoff_value = small_payoff_np-0.5
    print(small_payoff_np)
    print(small_payoff_value)
    u,s,v = np.linalg.svd(small_payoff_value)
    embedding_payoff = calc_embedding(small_payoff_value,v)
    x_max = max(abs(embedding_payoff[:,0]))+0.1
    y_max = max(abs(embedding_payoff[:,1]))+0.1
    avg_winrate = small_payoff_np.mean(axis=1)
    plt.clf()
    if use_annotate:
        for i,player_id in enumerate(player_ids):
            plt.annotate(player_id,(embedding_payoff[i,0]+.01,embedding_payoff[i,1]+.01))
    plt.scatter(embedding_payoff[:,0], embedding_payoff[:,1], c=avg_winrate, marker='.')
    plt.axis([-x_max,x_max,-y_max,y_max])
    plt.colorbar()
    plt.savefig(save_path) 

plot_scatter_image(win_rate_payoff,hist_ids,'agent_mode.jpg')
###partial data
small_player_ids = ['zh_value', 'zh_value2', 'zh_value3', 'zh_value4', 'zh_value5','zh_value6', 'zh_value7','zh_value8',]
plot_scatter_image(win_rate_payoff,small_player_ids,'small_agent_mode.jpg')

small_player_ids = ['226000_1', '226000_2', '226000_3',
                     '22600_m1', '22600_m2', '22600_m3','226000_0',
    '22600_m0',]
plot_scatter_image(win_rate_payoff,small_player_ids,'small_agent_226000.jpg',True)