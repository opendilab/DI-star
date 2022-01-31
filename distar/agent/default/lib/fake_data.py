import torch

from .features import MAX_ENTITY_NUM, MAX_DELAY, MAX_SELECTED_UNITS_NUM, SPATIAL_SIZE
from .actions import NUM_ACTIONS, NUM
from .data_transfer import MAX_ENTITY_NUM, MAX_DELAY, MAX_SELECTED_UNITS_NUM, SPATIAL_SIZE
from .action_dict import NUM_ACTIONS, NUM_BOOSTS
from .unit_dict import UNIT_GROUP_NUM, ENTITY_TYPE_NUM, MECHANIC_NUM
from distar.ctools.data.collate_fn import default_collate_with_dim

currTrainCount_MAX = 5


def hidden_state():
    return [(torch.zeros(size=(128,)), torch.zeros(size=(128,))) for _ in range(1)]


def spatial_info():
    return torch.rand(23, 128, 128)


def entity_info():
    data =  { # entity info
            'progressBuild': torch.randint(0, 11, size=(MAX_ENTITY_NUM, ), dtype=torch.float),
            'progressUpgrade': torch.randint(0, 11, size=(MAX_ENTITY_NUM, ), dtype=torch.float),
            'progressCaptured': torch.randint(0, 11, size=(MAX_ENTITY_NUM, ), dtype=torch.float),
            'progressTrain': torch.randint(0, 11, size=(MAX_ENTITY_NUM, ), dtype=torch.float),
            'hp': torch.randint(0, 11, size=(MAX_ENTITY_NUM, ), dtype=torch.float),
            'visible': torch.randint(0, 3, size=(MAX_ENTITY_NUM, ), dtype=torch.float),
            'x': torch.randint(0, SPATIAL_SIZE, size=(MAX_ENTITY_NUM, ), dtype=torch.float),
            'y': torch.randint(0, SPATIAL_SIZE, size=(MAX_ENTITY_NUM, ), dtype=torch.float),
            'task': torch.randint(0, 12, size=(MAX_ENTITY_NUM, ), dtype=torch.float),
            'isGround': torch.randint(0, 2, size=(MAX_ENTITY_NUM, ), dtype=torch.float),
            'isBoat': torch.randint(0, 2, size=(MAX_ENTITY_NUM, ), dtype=torch.float),
            'isPlane': torch.randint(0, 2, size=(MAX_ENTITY_NUM, ), dtype=torch.float),
            'currTrainType': torch.randint(0, ENTITY_TYPE_NUM, size=(MAX_ENTITY_NUM, ), dtype=torch.float),
            'currTrainCount': torch.randint(0, currTrainCount_MAX, size=(MAX_ENTITY_NUM, ), dtype=torch.float),
            'buState': torch.randint(0, 4, size=(MAX_ENTITY_NUM, ), dtype=torch.float),
            'nuclearIn': torch.randint(0, 2, size=(MAX_ENTITY_NUM, ), dtype=torch.float),
            'capture_alliance': torch.randint(0, 3, size=(MAX_ENTITY_NUM, ), dtype=torch.float),
            'level': torch.randint(0, 4, size=(MAX_ENTITY_NUM, ), dtype=torch.float),
            'remainCapacity': torch.randint(0, 21, size=(MAX_ENTITY_NUM, ), dtype=torch.float),
            'constrPaused': torch.randint(0, 2, size=(MAX_ENTITY_NUM, ), dtype=torch.float),
            'protoId': torch.randint(0, ENTITY_TYPE_NUM, size=(MAX_ENTITY_NUM, ), dtype=torch.float),
            'sideId': torch.randint(0, 4, size=(MAX_ENTITY_NUM, ), dtype=torch.float),
            }
    return data


def scalar_info():
    data =  {
            'race1': torch.randint(0, 4, size=(), dtype=torch.float),
            'race2': torch.randint(0, 4, size=(), dtype=torch.float),
            'race3': torch.randint(0, 4, size=(), dtype=torch.float),
            'race4': torch.randint(0, 4, size=(), dtype=torch.float),
            'race5': torch.randint(0, 4, size=(), dtype=torch.float),
            'race6': torch.randint(0, 4, size=(), dtype=torch.float),
            'agent_statistics': torch.rand(6),
            'time': torch.randint(0, 100, size=(), dtype=torch.float),
            'unit_counts_bow': torch.randint(0, 10, size=(ENTITY_TYPE_NUM,), dtype=torch.float),
            'beginning_build_order': torch.randint(0, 20, size=(20, ), dtype=torch.float),
            'cumulative_stat': torch.randint(0, 2, size=(ENTITY_TYPE_NUM, ), dtype=torch.float),
            'boost_vector': torch.randint(0, 2, size=(NUM_BOOSTS,), dtype=torch.float),
            'mechanic_vector': torch.randint(0, 2, size=(MECHANIC_NUM,), dtype=torch.float),
            'last_delay': torch.randint(0, MAX_DELAY, size=(), dtype=torch.float),
            'last_action_type': torch.randint(0, NUM_ACTIONS, size=(), dtype=torch.float),
            'cur_train_queue': torch.randint(0, 2, size=(ENTITY_TYPE_NUM,), dtype=torch.float),
            'electric_deficient': torch.randint(0, 2, size=(), dtype=torch.float),
            'time_remain': torch.randint(0, 2, size=(), dtype=torch.float),
            'oral_messages': torch.randint(0, 2, size=(9,), dtype=torch.float)
        }
    return data


def action_info():
    data = { # action info
            'action_type': torch.randint(0, NUM_ACTIONS, size=(), dtype=torch.long),
            'delay': torch.randint(0, MAX_DELAY, size=(), dtype=torch.long),
            'selected_units': torch.randint(0, 5, size=(MAX_SELECTED_UNITS_NUM, ), dtype=torch.long),
            'target_unit': torch.randint(0, MAX_ENTITY_NUM, size=(), dtype=torch.long),
            'target_location': torch.randint(0, SPATIAL_SIZE, size=(), dtype=torch.long)
        }
    return data


def action_mask():
    data = { # action mask
            'action_type': torch.randint(0, 1, size=(), dtype=torch.long),
            'delay': torch.randint(0, 1, size=(), dtype=torch.long),
            'selected_units': torch.randint(0, 1, size=(), dtype=torch.long),
            'target_unit': torch.randint(0, 1, size=(), dtype=torch.long),
            'target_location': torch.randint(0, 1, size=(), dtype=torch.long)
        }
    return data


def action_logp():
    data = {
            'action_type': torch.rand(size=()) + 2,
            'delay': torch.rand(size=()) + 2,
            'selected_units': torch.rand(size=(MAX_SELECTED_UNITS_NUM, )) + 2,
            'target_unit': torch.rand(size=()) + 2,
            'target_location': torch.rand(size=()) + 2
            }
    return data


def action_logits():
    data = {
            'action_type': torch.rand(size=(NUM_ACTIONS+1, )) - 1,
            'delay': torch.rand(size=(MAX_DELAY, )) - 1,
            'selected_units': torch.rand(size=(MAX_SELECTED_UNITS_NUM, MAX_ENTITY_NUM + 1)) - 1,
            'target_unit': torch.rand(size=(MAX_ENTITY_NUM, )) - 1,
            'target_location': torch.rand(size=(16384,)) - 1
        }

    mask = dict()
    mask['selected_units_logits_mask'] = data['selected_units'].sum(0)
    mask['target_units_logits_mask'] = data['target_unit']
    mask['actions_mask'] = {k: val.sum() for k,val in data.items()}
    mask['selected_units_mask'] = data['selected_units'].sum(-1)

    return data, mask


def fake_step_data():
    data = (
        spatial_info(),
        entity_info(),
        scalar_info(),
        action_info(),
        action_mask(),
        torch.randint(5, 100, size=(1, ), dtype=torch.long), # entity num
        torch.randint(0, MAX_SELECTED_UNITS_NUM, size=(), dtype=torch.long), # selected_units_num
        torch.randint(0, SPATIAL_SIZE, size=(512 , 2), dtype=torch.long) # entity location
    )
    return data

def fake_inference_data():
    data = (
        spatial_info(),
        entity_info(),
        scalar_info(),
        torch.randint(5, 100, size=(1,), dtype=torch.long),  # entity_num
        torch.randint(0, SPATIAL_SIZE, size=(512, 2), dtype=torch.long),  # entity_location
    )
    return data

def rl_step_data():
    teacher_action_logits, mask = action_logits()
    data = {
        'spatial_info': spatial_info(),
        'entity_info': entity_info(),
        'scalar_info': scalar_info(),
        'entity_num': torch.randint(5, 100, size=(1,), dtype=torch.long),
        'selected_units_num': torch.randint(0, MAX_SELECTED_UNITS_NUM, size=(), dtype=torch.long),
        'entity_location': torch.randint(0, SPATIAL_SIZE, size=(512, 2), dtype=torch.long),
        'hidden_state': [(torch.zeros(size=(128,)), torch.zeros(size=(128,))) for _ in range(1)], # hidden state
        'action_info': action_info(),
        'behaviour_logp': action_logp(),
        'teacher_logit': teacher_action_logits,
        'reward': {'winloss':torch.randint(-1, 1, size=(), dtype=torch.float),
                   'build_order': torch.randint(-1, 1, size=(), dtype=torch.float),
                   'built_unit': torch.randint(-1, 1, size=(), dtype=torch.float),
                   'effect': torch.randint(-1, 1, size=(), dtype=torch.float),
                   'upgrade': torch.randint(-1, 1, size=(), dtype=torch.float),
                   'battle': torch.randint(-1, 1, size=(), dtype=torch.float),
                   },
        'step': torch.randint(100, 1000, size=(), dtype=torch.long),
        'mask': mask,
        }
    return data

def transfer_data(data, cuda=False, share_memory=False, device=None):
    assert cuda or share_memory
    new_data = []
    for d in data:
        if isinstance(d, torch.Tensor):
            if cuda:
                d = d.to(device)
            if share_memory:
                d.share_memory_()
            new_data.append(d)
        elif isinstance(d, dict):
            if cuda:
                d = {k: v.to(device) for k, v in d.items()}
            if share_memory:
                d = {k: v.share_memory_() for k, v in d.items()}
            new_data.append(d)
    return tuple(new_data)


def fake_step_data_share_memory():
    data = fake_step_data()
    data = transfer_data(data)
    return data

def fake_rl_data_batch(batch_size=1):
    list_step_data = []
    # list_hidden_state = []
    for i in range(batch_size):
        step_data = rl_step_data()
        # hidden_state = step_data.pop('hidden_state')
        list_step_data.append(step_data)
        # list_hidden_state.append(hidden_state)

    step_data_batch = default_collate_with_dim(list_step_data)
    # hidden_state_batch = default_collate_with_dim(list_hidden_state,dim=0)
    batch = step_data_batch
    # batch['hidden_state'] = hidden_state_batch
    return batch

def fake_rl_data_batch_with_last(unroll_len=3):
    list_step_data = []
    # list_hidden_state = []
    for i in range(unroll_len):
        step_data = rl_step_data()
        # hidden_state = step_data.pop('hidden_state')
        list_step_data.append(step_data)
        # list_hidden_state.append(hidden_state)
    last_step_data = {
        'spatial_info': spatial_info(),
        'entity_info': entity_info(),
        'scalar_info': scalar_info(),
        'entity_num': torch.randint(5, 100, size=(1,), dtype=torch.long),
        # 'selected_units_num': torch.randint(0, MAX_SELECTED_UNITS_NUM, size=(), dtype=torch.long),
        'entity_location': torch.randint(0, SPATIAL_SIZE, size=(512, 2), dtype=torch.long),
        'hidden_state': [(torch.zeros(size=(128,)), torch.zeros(size=(128,))) for _ in range(1)], # hidden state
        }
    list_step_data.append(last_step_data)
    step_data_batch = default_collate_with_dim(list_step_data)
    # hidden_state_batch = default_collate_with_dim(list_hidden_state,dim=0)
    batch = step_data_batch
    # batch['hidden_state'] = hidden_state_batch
    return batch

def fake_rl_learner_data_batch(batch_size = 6, unroll_len = 4):
    data_batch_list = [fake_rl_data_batch_with_last(unroll_len) for _ in range(batch_size)]
    data_batch = default_collate_with_dim(data_batch_list,dim = 1)
    return data_batch



from typing import Sequence
def flat(data):
    if isinstance(data,torch.Tensor):
        return torch.flatten(data, start_dim=0, end_dim=1) # (1, (T+1) * B)
    elif isinstance(data,dict):
        new_data = {}
        for k,val in data.items():
            new_data[k] = flat(val)
        return new_data
    elif isinstance(data,Sequence):
        new_data = [flat(v) for v in data]
        return new_data
    else:
        print(type(data))

def rl_learner_forward_data(batch_size = 6, unroll_len = 4):
    data = fake_rl_learner_data_batch(batch_size, unroll_len)
    new_data = {}
    for k,val in data.items():
        if k in  ['spatial_info',
                            'entity_info',
                            'scalar_info',
                            'entity_num',
                            # 'action_info'
                            # 'selected_units_num',
                            'entity_location',
                            'hidden_state',]:
            new_data[k] = flat(val)
        else:
            new_data[k] = val
    new_data['batch_size'] = batch_size
    new_data['unroll_len'] = unroll_len
    return new_data

if __name__ == '__main__':
    data = fake_rl_learner_data_batch()
