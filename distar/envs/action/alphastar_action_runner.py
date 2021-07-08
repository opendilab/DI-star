from typing import List, Tuple
from collections import Sequence
import copy
from ctools.pysc2.lib.actions import FunctionCall
from ctools.pysc2.lib.action_dict import GENERAL_ACTION_INFO_MASK
from ctools.envs.env.base_env import BaseEnv
from ctools.envs.common import EnvElementRunner
from .alphastar_action import AlphaStarRawAction


class AlphaStarRawActionRunner(EnvElementRunner):
    # override
    def _init(self, cfg: dict) -> None:
        self._core = AlphaStarRawAction(cfg.action)
        self._agent_num = cfg.agent_num

    # override
    def get(self, engine: BaseEnv) -> Tuple[List[FunctionCall], List[int], List[AlphaStarRawAction.Action]]:
        agent_action = copy.deepcopy(engine.agent_action)
        assert isinstance(agent_action, Sequence) and len(agent_action) == self._agent_num, agent_action
        ret = []
        for i in range(self._agent_num):
            action = agent_action[i]
            if not engine.due[i]:
                ret.append([FunctionCall.init_with_validation(0, [], raw=True), 0, None])
                continue
            action = self._core._from_agent_processor(action)
            legal = self._check_action(action)
            if not legal:
                # TODO(nyz) more fined solution for illegal action
                print('[WARNING], illegal raw action: {}'.format(action))
                ret.append([FunctionCall.init_with_validation(0, [], raw=True), 1, None])
                continue
            action_type, delay = action[:2]
            args = [v for v in action[2:6] if v is not None]  # queued, selected_units, target_units, target_location
            ret.append([FunctionCall.init_with_validation(action_type, args, raw=True), delay, action])
        return list(zip(*ret))

    # override
    def reset(self) -> None:
        pass

    def _check_action(self, action):
        action_attr = GENERAL_ACTION_INFO_MASK[action.action_type]
        if action_attr['selected_units']:
            if action.selected_units is None or len(action.selected_units) == 0:
                return False
        if action_attr['target_units']:
            if action.target_units is None or len(action.target_units) == 0:
                return False
        return True
