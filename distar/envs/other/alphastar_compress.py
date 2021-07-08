import torch
import numpy as np


def compress_obs(obs):
    if obs is None:
        return None
    if isinstance(obs['entity_info'], dict):
        return obs
    new_obs = {}
    special_list = ['entity_info', 'spatial_info']
    for k in obs.keys():
        if k not in special_list:
            new_obs[k] = obs[k]

    new_obs['entity_info'] = {}
    entity_no_bool = 4
    new_obs['entity_info'] = {}
    new_obs['entity_info']['no_bool'] = obs['entity_info'][:, :entity_no_bool].numpy()
    entity_bool = obs['entity_info'][:, entity_no_bool:].to(torch.uint8).numpy()
    new_obs['entity_info']['bool_ori_shape'] = entity_bool.shape
    B, N = entity_bool.shape
    N_strided = N if N % 8 == 0 else (N // 8 + 1) * 8
    new_obs['entity_info']['bool_strided_shape'] = (B, N_strided)
    if N != N_strided:
        entity_bool = np.concatenate([entity_bool, np.zeros((B, N_strided - N), dtype=np.uint8)], axis=1)
    new_obs['entity_info']['bool'] = np.packbits(entity_bool)

    spatial_no_bool = 1
    new_obs['spatial_info'] = {}
    spatial_bool = obs['spatial_info'][spatial_no_bool:].to(torch.uint8).numpy()
    spatial_uint8 = obs['spatial_info'][:spatial_no_bool].mul_(256).to(torch.uint8).numpy()
    new_obs['spatial_info']['no_bool'] = spatial_uint8
    new_obs['spatial_info']['bool_ori_shape'] = spatial_bool.shape
    new_obs['spatial_info']['bool'] = np.packbits(spatial_bool)
    return new_obs


def decompress_obs(obs):
    if obs is None:
        return None
    new_obs = {}
    special_list = ['entity_info', 'spatial_info']
    for k in obs.keys():
        if k not in special_list:
            new_obs[k] = obs[k]

    new_obs['entity_info'] = {}
    entity_bool = np.unpackbits(obs['entity_info']['bool']).reshape(*obs['entity_info']['bool_strided_shape'])
    if obs['entity_info']['bool_strided_shape'][1] != obs['entity_info']['bool_ori_shape'][1]:
        entity_bool = entity_bool[:, :obs['entity_info']['bool_ori_shape'][1]]
    entity_no_bool = obs['entity_info']['no_bool']
    spatial_bool = np.unpackbits(obs['spatial_info']['bool']).reshape(*obs['spatial_info']['bool_ori_shape'])
    spatial_uint8 = obs['spatial_info']['no_bool'].astype(np.float32) / 256.
    new_obs['entity_info'] = torch.cat([torch.FloatTensor(entity_no_bool), torch.FloatTensor(entity_bool)], dim=1)
    new_obs['spatial_info'] = torch.cat([torch.FloatTensor(spatial_uint8), torch.FloatTensor(spatial_bool)], dim=0)
    return new_obs


if __name__ == '__main__':
    import os
    import copy
    data = torch.load('C:/Users/agi\Desktop\work/job_f858d7b6-5a25-11eb-bb84-2cfda1bc2763_env_0_agent_0_0e321f40-5a26-11eb-98d8-2cfda1bc2763')
    for f in os.listdir('C:/Users/agi\Desktop\work/'):
        if 'job' in f:
            path = os.path.join('C:/Users/agi\Desktop\work/', f)
            save_path = os.path.join('C:/Users/agi\Desktop\work/recom', f)
            data = torch.load(path)
            for i in range(len(data)):
                for k in ['obs_home', 'obs_away', 'obs_home_next', 'obs_away_next']:
                    if k in data[i].keys():
                        data[i][k] = compress_obs(data[i][k])
            torch.save(data, save_path)
