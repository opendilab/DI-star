import torch
import copy
from typing import Any, Optional
from collections import OrderedDict
from ctools.worker.agent import BaseAgent, add_plugin, IAgentStatelessPlugin, AgentAggregator
from ctools.pysc2.lib.action_dict import GENERAL_ACTION_INFO_MASK
from ctools.pysc2.lib.static_data import ACTIONS_REORDER_INV


def post_processing(data, bs):
    action, action_output = data[0]
    action, entity_raw, selected_units_num = list(action.values())
    algo_action = copy.deepcopy(action)

    entity_raw = [{k: entity_raw[k][b] for k in entity_raw.keys()} for b in range(bs)]
    env_action = [{k: action[k][b] for k in action.keys()} for b in range(bs)]
    for b in range(bs):
        action_type = env_action[b]['action_type'].item()
        action_type = ACTIONS_REORDER_INV[action_type]
        for k in ['queued', 'selected_units', 'target_units', 'target_location']:
            if not GENERAL_ACTION_INFO_MASK[action_type][k]:
                env_action[b][k] = None
        if env_action[b]['selected_units'] is not None:
            env_action[b]['selected_units'] = env_action[b]['selected_units'][:selected_units_num[b] - 1
                                                                              ]  # exclude end flag

    output = {}
    output['action'] = [{'action': a, 'entity_raw': e} for a, e in zip(env_action, entity_raw)]
    output['action_output'] = action_output
    output['algo_action'] = algo_action
    output['selected_units_num'] = selected_units_num
    output['next_state'] = data[1]
    if data[2] is not None:
        output['baselines'] = data[2]
    return output


class ASDataTransformPlugin(IAgentStatelessPlugin):

    @classmethod
    def register(cls: type, agent: BaseAgent) -> None:

        def data_transform_wrapper(fn):

            def wrapper(data, **kwargs):
                ret = fn(data, **kwargs)
                return post_processing(ret, bs=len(ret[1]))

            return wrapper

        agent.forward = data_transform_wrapper(agent.forward)


add_plugin('as_data_transform', ASDataTransformPlugin)


class AlphaStarActorAgent(BaseAgent):

    def load_state_dict(self, state_dict: dict) -> None:
        actor_state_dict = state_dict['model']
        actor_state_dict = OrderedDict({k: v for k, v in actor_state_dict.items() if 'value_networks' not in k})
        self._model.load_state_dict(actor_state_dict, strict=False)

    def forward(self, data: Any, param: Optional[dict] = None) -> dict:
        if param is None:
            param = {}
        if 'teacher' in self._plugin_cfg:
            param['mode'] = 'mimic_single'
        else:
            param['mode'] = 'compute_action'
        return super().forward(data, param)


def create_as_actor_agent(model, state_num, use_teacher=False):
    plugin_cfg = {
        'main': OrderedDict(
            {
                'as_data_transform': {},
                'hidden_state': {
                    'state_num': state_num,
                    'save_prev_state': True,
                },
                'grad': {
                    'enable_grad': False
                },
            }
        ),
    }
    if use_teacher:
        plugin_cfg['teacher'] = OrderedDict(
            {
                'hidden_state': {
                    'state_num': state_num,
                    'save_prev_state': True,
                },
                'teacher': {
                    'teacher_cfg': {}
                },
                'grad': {
                    'enable_grad': False
                },
            }
        )
    agent = AgentAggregator(AlphaStarActorAgent, model, plugin_cfg)
    return agent


class AlphaStarLearnerAgent(BaseAgent):

    def forward(self, data: Any, param: Optional[dict] = None) -> dict:
        if param is None:
            param = {}
        param['mode'] = 'compute_action_value'
        return super().forward(data, param)


def create_as_learner_agent(model, state_num):
    plugin_cfg = {
        'main': OrderedDict({
            'grad': {
                'enable_grad': True
            },
        })
    }
    agent = AgentAggregator(AlphaStarLearnerAgent, model, plugin_cfg)
    return agent

class SupervisedStarLearnerAgent(BaseAgent):

    def forward(self, data: Any, param: Optional[dict] = None) -> dict:
        if param is None:
            param = {}
        param['mode'] = 'mimic_parallel'
        for i in range(len(data['prev_state'])):
            if data['prev_state'][i] is not None:
                data['prev_state'][i] = (s.detach() for s in data['prev_state'][i])

        return super().forward(data, param)


def create_sl_learner_agent(model, state_num):
    plugin_cfg = {
        'main': OrderedDict({
            'hidden_state': {
                'state_num': state_num,
                'save_prev_state': False,
            },
            'grad': {
                'enable_grad': True
            },
        })
    }
    agent = AgentAggregator(SupervisedStarLearnerAgent, model, plugin_cfg)
    return agent
