from collections.abc import Sequence, Mapping
from numbers import Integral
import torch
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate
from ctools.pysc2.lib.static_data import ACTIONS_REORDER_INV
from ctools.pysc2.lib.action_dict import GENERAL_ACTION_INFO_MASK
from ctools.utils import lists_to_dicts
from ctools.torch_utils.network.rnn import sequence_mask
from distar.envs.other.alphastar_compress import decompress_obs, compress_obs
import numpy as np
def as_human_collate_fn(data):
    if isinstance(data,dict):
        new_data ={}
        for k,val in data.items():
            # if k == 'cumulative_stat' and len(data[k]['effect'].shape) == 2:
            #     continue
            new_data[k] = as_human_collate_fn(val)
        return new_data
    elif isinstance(data,torch.Tensor):
        return data.unsqueeze(0)
    elif isinstance(data,list):
        return data
    elif isinstance(data,np.ndarray):
        return data
    elif data is None or isinstance(data,tuple):
        return data
    else:
        raise NotImplementedError

def policy_collate_fn(batch, max_delay=127):
    data_item = {
        'spatial_info': True,  # special op
        'scalar_info': True,
        'entity_info': False,
        'entity_raw': False,
        'actions': False,
        'map_size': False,
        'start_step': False,
        'actions_mask': False
    }

    def merge_func(data):
        new_data = lists_to_dicts(data)
        for k, merge in data_item.items():
            if merge:
                new_data[k] = default_collate(new_data[k])
                continue
            if k == 'entity_info':
                new_data['entity_num'] = torch.LongTensor([[i.shape[0]] for i in new_data[k]])
            elif k == 'actions':
                new_data[k] = lists_to_dicts(new_data[k])
                for _k, v in new_data[k].items():
                    if _k in ['action_type', 'delay', 'repeat', 'queued', 'target_units']:
                        new_data[k][_k] = torch.cat(v, dim=0)
                    elif _k == 'target_location':
                        new_data[k][_k] = torch.stack(v, dim=0)
                    elif _k == 'selected_units':
                        new_data['selected_units_num'] = torch.LongTensor([[i.shape[0]] for i in new_data[k][_k]])
            elif k == 'entity_raw':
                new_data[k] = lists_to_dicts(new_data[k])
                continue
            elif k == 'actions_mask':
                new_data[k] = lists_to_dicts(new_data[k])
                for _k, v in new_data[k].items():
                    new_data[k][_k] = torch.BoolTensor(v)
        return new_data

    total_batch = []
    traj_lens = []
    if_start = []
    for b in batch:
        if_start.append(b[0].pop('start_step'))
        traj_lens.append(len(b))
        total_batch += b
    data = merge_func(total_batch)
    data['start_step'] = if_start
    data['traj_lens'] = traj_lens
    return data


def diff_shape_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if any([isinstance(elem, type(None)) for elem in batch]):
        return batch
    elif isinstance(elem, torch.Tensor):
        shapes = [e.shape for e in batch]
        if len(set(shapes)) != 1:
            return batch
        else:
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            return diff_shape_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, Integral):
        return batch
    elif isinstance(elem, Mapping):
        return {key: diff_shape_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(diff_shape_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, Sequence):
        transposed = zip(*batch)
        return [diff_shape_collate(samples) for samples in transposed]

    raise TypeError('not support element type: {}'.format(elem_type))


def as_learner_collate_fn(batch):
    # ret keys:
    # sequence: batch_size, traj_len, prev_state
    # obs: entity_raw, entity_info, spatial_info, scalar_info, map_size
    # mask: entity_num, selected_units_num
    # action: actions, actions_mask, behaviour_output, teacher_output
    # rl: reward, game_second

    if isinstance(batch[0][0]['obs_home']['entity_info'], dict):
        obs_keys = ['obs_home', 'obs_away', 'obs_home_next', 'obs_away_next']
        for b in range(len(batch)):
            for t in range(len(batch[b])):
                for k in batch[b][t]:
                    if k in obs_keys:
                        batch[b][t][k] = decompress_obs(batch[b][t][k])

    ret = {}
    # bs, traj -> traj, bs
    ret['batch_size'] = batch_size = len(batch)
    batch = list(zip(*batch))
    ret['traj_len'] = traj_len = len(batch)

    ret['prev_state'] = [d.pop('prev_state') for d in batch[0]]

    obs_home_next = [d.pop('obs_home_next') for d in batch[-1]]
    obs_away_next = [d.pop('obs_away_next') for d in batch[-1]]
    new_batch = []
    for s in range(len(batch)):
        new_batch += batch[s]
    if 'obs_home_next' in new_batch[0].keys():
        new_batch[0].pop('obs_home_next')
        new_batch[0].pop('obs_away_next')
    new_batch = lists_to_dicts(new_batch)

    new_batch['obs_home'] += obs_home_next
    new_batch['obs_away'] += obs_away_next
    obs = new_batch['obs_home'] + new_batch['obs_away']
    obs = lists_to_dicts(obs)
    if 'actions' in obs.keys():
        obs.pop('actions')

    for k in ['spatial_info', 'scalar_info']:
        ret[k] = default_collate(obs[k])


    entity_raw = lists_to_dicts(obs['entity_raw'])
    entity_raw['location'] = torch.nn.utils.rnn.pad_sequence(entity_raw['location'], batch_first=True)
    ret['entity_raw'] = entity_raw
    ret['entity_num'] = torch.LongTensor([[i.shape[0]] for i in obs['entity_info']])
    ret['entity_info'] = torch.nn.utils.rnn.pad_sequence(obs['entity_info'], batch_first=True)
    ret['map_size'] = obs['map_size']

    ret['selected_units_num'] = torch.stack(new_batch['selected_units_num'], dim=0)
    max_selected_units_num = ret['selected_units_num'].max()

    actions = lists_to_dicts(new_batch['actions'])
    actions_mask = {k: [] for k in actions.keys()}
    for i in range(len(actions['action_type'])):
        action_type = actions['action_type'][i].item()
        flag = action_type == 0
        inv_action_type = ACTIONS_REORDER_INV[action_type]
        actions_mask['action_type'].append(False) if flag else actions_mask['action_type'].append(True)
        actions_mask['delay'].append(False) if flag else actions_mask['delay'].append(True)
        for k in ['queued', 'target_units', 'selected_units', 'target_location']:
            if flag or not GENERAL_ACTION_INFO_MASK[inv_action_type][k]:
                actions_mask[k].append(False)
            else:
                actions_mask[k].append(True)

    for k in actions_mask.keys():
        actions_mask[k] = torch.BoolTensor(actions_mask[k])

    map_size = list(zip(*obs['map_size']))
    assert len(set(map_size[0])) == 1 and len(set(map_size[1])) == 1, 'only support same size map'
    map_size = obs['map_size'][0]
    for k, v in actions.items():
        if k in ['action_type', 'delay', 'repeat', 'queued', 'target_units']: 
            actions[k] = torch.cat(v, dim=0)            
        elif k == 'target_location':
            actions[k] = torch.stack(v)
            actions[k] = actions[k][:, 0] * map_size[1] + actions[k][:, 1]
            actions[k] = actions[k].long()
        else:
            actions[k] = torch.nn.utils.rnn.pad_sequence(actions[k], batch_first=True)
            actions[k] = actions[k][:, :max_selected_units_num].contiguous()

    ret['actions'] = actions

    ret['reward'] = {}
    reward = default_collate(new_batch['reward'])
    for k, v in reward.items():
        ret['reward'][k] = v.view(traj_len, batch_size)

    ret['game_second'] = torch.LongTensor(new_batch['game_second'])

    home_size = len(ret['game_second'])
    max_entity_num = ret['entity_num'][:home_size].max()
    for k in ['behaviour_output', 'teacher_output']:
        data = lists_to_dicts(new_batch[k])
        for _k in data.keys():
            if _k in ['action_type', 'delay', 'repeat', 'queued', 'target_location']:
                data[_k] = default_collate(data[_k])
            elif _k == 'selected_units':
                for i in range(len(data[_k])):
                    if len(data[_k][i].shape) == 1:
                        data[_k][i] = data[_k][i].unsqueeze(0)
                    data[_k][i] = data[_k][i][:, :max_entity_num + 1]
                    data[_k][i] = torch.nn.functional.pad(data[_k][i], (0, max_entity_num + 1 - data[_k][i].shape[1]),
                                                          'constant', -1e9)

                data[_k] = torch.nn.utils.rnn.pad_sequence(data[_k], batch_first=True)
                data[_k] = data[_k][:, :max_selected_units_num].contiguous()
            elif _k == 'target_units':
                data[_k] = torch.nn.utils.rnn.pad_sequence(data[_k], batch_first=True, padding_value=-1e9)
                data[_k] = data[_k][:, :max_entity_num].contiguous()
        ret[k] = data

    mask = {}
    mask['actions_mask'] = actions_mask
    mask['selected_units_mask'] = sequence_mask(ret['selected_units_num'][:home_size])
    entity_num = ret['entity_num']
    mask['target_units_logits_mask'] = sequence_mask(entity_num[:home_size])
    plus_entity_num = entity_num + 1  # selected units head have one more end embedding
    mask['selected_units_logits_mask'] = sequence_mask(plus_entity_num[:home_size])
    ret['mask'] = mask
    return ret


def as_eval_collate_fn(data):
    data_item = {
        'spatial_info': True,  # special op
        'scalar_info': True,
        'entity_info': False,
        'entity_raw': False,
        'map_size': False,
    }

    def merge_func(data):
        new_data = lists_to_dicts(data)
        for k, merge in data_item.items():
            if merge:
                new_data[k] = default_collate(new_data[k])
                continue
            if k == 'entity_info':
                new_data['entity_num'] = torch.LongTensor([[i.shape[0]] for i in new_data[k]])
                new_data[k] = torch.nn.utils.rnn.pad_sequence(new_data[k], batch_first=True)
            elif k == 'entity_raw':
                new_data[k] = lists_to_dicts(new_data[k])
                if isinstance(new_data[k]['location'][0], list):
                    new_data[k]['location'] = [torch.LongTensor(item) for item in new_data[k]['location']]
                new_data[k]['location'] = torch.nn.utils.rnn.pad_sequence(new_data[k]['location'], batch_first=True)
        return new_data

    data = list(zip(*data))
    for i in range(len(data)):
        data[i] = merge_func(data[i])
    return data


def as_eval_decollate_fn(batch):
    ignore_item = ['prev_state', 'teacher_prev_state', 'action']
    if batch is None:
        return None
    elif isinstance(batch, torch.Tensor):
        batch = torch.split(batch, 1, dim=0)
        # squeeze if original batch's shape is like (B, dim1, dim2, ...);
        # otherwise directly return the list.
        if len(batch[0].shape) > 1:
            batch = [elem.squeeze(0) for elem in batch]
        batch = [b.clone() for b in batch]
        return list(batch)
    elif isinstance(batch, Sequence):
        return list(zip(*[as_eval_decollate_fn(e) for e in batch]))
    elif isinstance(batch, Mapping):
        tmp = {k: v if k in ignore_item else as_eval_decollate_fn(v) for k, v in batch.items()}
        B = len(list(tmp.values())[0])
        return [{k: tmp[k][i] for k in tmp.keys()} for i in range(B)]

    raise TypeError("not support batch type: {}".format(type(batch)))
