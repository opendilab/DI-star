from collections import namedtuple
import copy

import torch
import torch.nn as nn

from ctools.pysc2.lib.action_dict import GENERAL_ACTION_INFO_MASK, ACTIONS_STAT
from ctools.pysc2.lib.static_data import NUM_UNIT_TYPES, UNIT_TYPES_REORDER, ACTIONS_REORDER_INV, PART_ACTIONS_MAP,\
    PART_ACTIONS_MAP_INV, SELECTED_UNITS_MASK, TARGET_UNITS_MASK
from distar.envs import get_location_mask
from .head import DelayHead, QueuedHead, SelectedUnitsHead, TargetUnitHead, LocationHead, ActionTypeHead


def build_head(name):
    head_dict = {
        'action_type_head': ActionTypeHead,
        'base_action_type_head': ActionTypeHead,
        'spec_action_type_head': ActionTypeHead,
        'delay_head': DelayHead,
        'queued_head': QueuedHead,
        'selected_units_head': SelectedUnitsHead,
        'target_unit_head': TargetUnitHead,
        'location_head': LocationHead,
    }
    return head_dict[name]


class Policy(nn.Module):
    MimicInput = namedtuple(
        'MimicInput', [
            'actions', 'entity_raw', 'action_type_mask', 'lstm_output', 'entity_embeddings', 'map_skip',
            'scalar_context', 'spatial_info', 'entity_num', 'selected_units_num'
        ]
    )

    EvaluateInput = namedtuple(
        'EvaluateInput', [
            'entity_raw', 'action_type_mask', 'lstm_output', 'entity_embeddings', 'map_skip', 'scalar_context',
            'spatial_info', 'entity_num'
        ]
    )

    def __init__(self, cfg):
        super(Policy, self).__init__()
        self.cfg = cfg
        self.head = nn.ModuleDict()
        for item in cfg.head.head_names:
            self.head[item] = build_head(item)(cfg.head[item])

    def _look_up_action_attr(self, action_type, entity_raw, units_num, spatial_info):
        action_arg_mask = {
            'selected_units_type_mask': [],
            'selected_units_mask': [],
            'target_units_type_mask': [],
            'target_units_mask': [],
            'location_mask': []
        }
        device = action_type[0].device
        action_attr = {'queued': [], 'selected_units': [], 'target_units': [], 'target_location': []}
        for idx, action in enumerate(action_type):
            action_type_val = ACTIONS_REORDER_INV[action.item()]
            action_info_hard_craft = GENERAL_ACTION_INFO_MASK[action_type_val]
            try:
                action_info_stat = ACTIONS_STAT[action_type_val]
            except KeyError as e:
                print('We are issuing a command (reordered:{}), never seen in replays'.format(action_type_val))
                action_info_stat = {'selected_type': [], 'target_type': []}
            # else case is the placeholder
            if action_info_hard_craft['selected_units']:
                type_hard_craft = set(action_info_hard_craft['avail_unit_type_id'])
                type_stat = set(action_info_stat['selected_type'])
                type_set = type_hard_craft.union(type_stat)
                reorder_type_list = [UNIT_TYPES_REORDER[t] for t in type_set]
                selected_units_type_mask = torch.zeros(NUM_UNIT_TYPES)
                selected_units_type_mask[reorder_type_list] = 1
                action_arg_mask['selected_units_type_mask'].append(selected_units_type_mask.to(device))
                selected_units_mask = torch.zeros(units_num[idx])
                for i, t in enumerate(entity_raw[idx]['type']):
                    if t in type_set:
                        selected_units_mask[i] = 1
                action_arg_mask['selected_units_mask'].append(selected_units_mask.to(device))
            else:
                action_arg_mask['selected_units_mask'].append(torch.zeros(units_num[idx]).to(device))
                action_arg_mask['selected_units_type_mask'].append(torch.zeros(NUM_UNIT_TYPES).to(device))
            if action_info_hard_craft['target_units']:
                type_set = set(action_info_stat['target_type'])
                reorder_type_list = [UNIT_TYPES_REORDER[t] for t in type_set]
                target_units_type_mask = torch.zeros(NUM_UNIT_TYPES)
                target_units_type_mask[reorder_type_list] = 1
                action_arg_mask['target_units_type_mask'].append(target_units_type_mask.to(device))
                target_units_mask = torch.zeros(units_num[idx])
                for i, t in enumerate(entity_raw[idx]['type']):
                    if t in type_set:
                        target_units_mask[i] = 1
                action_arg_mask['target_units_mask'].append(target_units_mask.to(device))
            else:
                action_arg_mask['target_units_mask'].append(torch.zeros(units_num[idx]).to(device))
                action_arg_mask['target_units_type_mask'].append(torch.zeros(NUM_UNIT_TYPES).to(device))
            # TODO(nyz) location mask for different map size
            if action_info_hard_craft['target_location']:
                location_mask = get_location_mask(action_type_val, spatial_info[idx])
                action_arg_mask['location_mask'].append(location_mask)
            else:
                shapes = spatial_info[idx].shape[-2:]
                action_arg_mask['location_mask'].append(torch.zeros(1, *shapes).to(device))
            # get action attribute(which args the action type owns)
            for k in action_attr.keys():
                action_attr[k].append(action_info_hard_craft[k])
                # if no available units, set the corresponding attribute False
                # TODO(nyz) deal with these illegal action in the interaction between agent and env
                # if k in ['selected_units', 'target_units']:
                #     if action_attr[k][-1] and action_arg_mask[k + '_mask'][-1].abs().sum() < 1e-6:
                #         print('[WARNING]: action_type {} has no available units'.format(action_type_val))
                #         action_attr[k][-1] = False
        # stack mask
        for k in ['selected_units_type_mask', 'target_units_type_mask', 'location_mask']:
            action_arg_mask[k] = torch.stack(action_arg_mask[k], dim=0)
        return action_attr, action_arg_mask

    def _action_type_forward(self, lstm_output, scalar_context, action_type_mask, temperature, action_type=None):
        kwargs = {
            'lstm_output': lstm_output,
            'scalar_context': scalar_context,
            'action_type_mask': action_type_mask,
            'temperature': temperature,
            'action_type': action_type
        }
        if 'action_type_head' in self.head.keys():
            return self.head['action_type_head'](**kwargs)
        elif 'base_action_type_head' in self.head.keys() and 'spec_action_type_head' in self.head.keys():
            # get part action mask
            base_action_type_mask = action_type_mask[:, list(PART_ACTIONS_MAP['base'].keys())]
            spec_action_type_mask = action_type_mask[:, list(PART_ACTIONS_MAP['spec'].keys())]
            if action_type is not None:
                base_action_type = action_type.clone()
                spec_action_type = action_type.clone()
                # to part action type id
                for idx, val in enumerate(action_type):
                    val = val.item()
                    if val == 0:
                        continue
                    elif val in PART_ACTIONS_MAP['base'].keys():
                        spec_action_type[idx] = 0
                        base_action_type[idx] = PART_ACTIONS_MAP['base'][val]
                    else:
                        spec_action_type[idx] = PART_ACTIONS_MAP['spec'][val]
                        base_action_type[idx] = 0
                # double head forward
                kwargs['action_type'] = base_action_type
                kwargs['action_type_mask'] = base_action_type_mask
                base_logits, base_action_type, base_embeddings = self.head['base_action_type_head'](**kwargs)
                kwargs['action_type'] = spec_action_type
                kwargs['action_type_mask'] = spec_action_type_mask
                spec_logits, spec_action_type, spec_embeddings = self.head['spec_action_type_head'](**kwargs)
            else:
                kwargs['action_type_mask'] = base_action_type_mask
                base_logits, base_action_type, base_embeddings = self.head['base_action_type_head'](**kwargs)
                kwargs['action_type_mask'] = spec_action_type_mask
                spec_logits, spec_action_type, spec_embeddings = self.head['spec_action_type_head'](**kwargs)
            # to total action type id
            for idx, val in enumerate(base_action_type):
                base_action_type[idx] = PART_ACTIONS_MAP_INV['base'][val.item()]
            for idx, val in enumerate(spec_action_type):
                spec_action_type[idx] = PART_ACTIONS_MAP_INV['spec'][val.item()]
            mask = torch.where(
                spec_action_type == 0, torch.ones_like(spec_action_type), torch.zeros_like(spec_action_type)
            )  # noqa
            action_type = mask * base_action_type + spec_action_type
            mask = mask.view(-1, *[1 for _ in range(len(base_embeddings.shape) - 1)]).to(
                base_embeddings.dtype
            )  # batch is the first dim  # noqa
            embeddings = mask * base_embeddings + (1 - mask) * spec_embeddings
            return [base_logits, spec_logits], action_type, embeddings
        else:
            raise KeyError("no necessary action type head in heads{}".format(self.head.keys()))

    def mimic(self, inputs, temperature=1.0):
        '''
            Overview: supervised learning policy forward graph
            Arguments:
                - inputs (:obj:`Policy.Input`) namedtuple
                - temperature (:obj:`float`) logits sample temperature
            Returns:
                - logits (:obj:`dict`) logits(or other format) for calculating supervised learning loss
        '''
        actions, entity_raw, action_type_mask, lstm_output, \
        entity_embeddings, map_skip, scalar_context, spatial_info, entity_num, selected_units_num = inputs
        logits = {'queued': [], 'selected_units': [], 'target_units': [], 'target_location': []}
        # action type
        logits['action_type'], _, embeddings = self._action_type_forward(
            lstm_output, scalar_context, action_type_mask, temperature, actions['action_type']
        )
        selected_units_type_mask = SELECTED_UNITS_MASK[actions['action_type']].to(lstm_output.device)
        target_units_type_mask = TARGET_UNITS_MASK[actions['action_type']].to(lstm_output.device)

        # action arg delay
        logits['delay'], _, embeddings = self.head['delay_head'](embeddings, actions['delay'])

        # action arg queued
        logits['queued'], _, embeddings = self.head['queued_head'](embeddings, temperature, actions['queued'])

        logits['selected_units'], _, embeddings, _ = self.head['selected_units_head'](
            embeddings, selected_units_type_mask, entity_embeddings, temperature, actions['selected_units'], entity_num,
            selected_units_num
        )

        logits['target_units'], _ = self.head['target_unit_head'](
            embeddings, target_units_type_mask, entity_embeddings, temperature, actions['target_units'], entity_num
        )

        logits['target_location'], _ = self.head['location_head'](
            embeddings, map_skip, temperature, actions['target_location']
        )

        return logits

    def evaluate(self, inputs, temperature=1.0):
        '''
            Overview: agent(policy) evaluate forward graph, or in reinforcement learning
            Arguments:
                - inputs (:obj:`Policy.Input`) namedtuple
                - temperature (:obj:`float`) logits sample temperature
            Returns:
                - logits (:obj:`dict`) logits
                - action (:obj:`dict`) action predicted by agent(policy)
        '''
        # inputs = torch.load(r'C:\Program Files (x86)\StarCraft II\Maps\Ladder2019Season2/data.d')
        entity_raw, action_type_mask, lstm_output, entity_embeddings, map_skip, scalar_context, spatial_info, entity_num = inputs

        action = {'queued': [], 'selected_units': [], 'target_units': [], 'target_location': []}
        logits = {'queued': [], 'selected_units': [], 'target_units': [], 'target_location': []}

        # action type
        logits['action_type'], action['action_type'], embeddings = self._action_type_forward(
            lstm_output, scalar_context, action_type_mask, temperature
        )

        selected_units_type_mask = SELECTED_UNITS_MASK[action['action_type']].to(lstm_output.device)
        target_units_type_mask = TARGET_UNITS_MASK[action['action_type']].to(lstm_output.device)

        # action arg delay
        logits['delay'], action['delay'], embeddings = self.head['delay_head'](embeddings)

        logits['queued'], action['queued'], embeddings = self.head['queued_head'](embeddings, temperature)

        logits['selected_units'], action['selected_units'], embeddings, selected_units_num = self.head[
            'selected_units_head'](
                embeddings, selected_units_type_mask, entity_embeddings, temperature, entity_num=entity_num
            )

        logits['target_units'], action['target_units'] = self.head['target_unit_head'](
            embeddings, target_units_type_mask, entity_embeddings, temperature, entity_num=entity_num
        )

        logits['target_location'], action['target_location'] = self.head['location_head'](
            embeddings, map_skip, temperature
        )

        action = {'action': action, 'entity_raw': entity_raw, 'selected_units_num': selected_units_num}
        return action, logits

    def forward(self, inputs, mode=None, **kwargs):
        assert (mode in ['mimic', 'evaluate'])
        return getattr(self, mode)(inputs, **kwargs)
