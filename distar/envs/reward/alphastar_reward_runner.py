import numpy as np
import torch
from ctools.envs.common import EnvElementRunner
from ctools.envs.env.base_env import BaseEnv
from .alphastar_reward import AlphaStarReward


class AlphaStarRewardRunner(EnvElementRunner):
    # override
    def _init(self, agent_num, *args, return_list=True, **kwargs):
        self._core = AlphaStarReward(agent_num, *args, **kwargs)
        self._agent_num = agent_num
        self._return_list = return_list

    # override
    def get(self, engine: BaseEnv) -> dict:
        assert isinstance(engine, BaseEnv)
        action = engine.action
        action_type = [0] * self._agent_num
        for i, a in enumerate(action):
            if a is not None:
                action_type[i] = a.action_type
        if self._agent_num == 1:
            battle_value = None
        else:
            now_battle_value = self._get_battle_value(engine.raw_obs)
            battle_value = AlphaStarReward.BattleValues(
                self._last_battle_value[0], now_battle_value[0], self._last_battle_value[1], now_battle_value[1]
            )
            self._last_battle_value = now_battle_value

        return self._core._to_agent_processor(
            engine.reward,
            action_type,
            engine.episode_stat,
            engine.loaded_eval_stat,
            engine.episode_steps,
            battle_value,
            return_list=self._return_list
        )

    # override
    def reset(self) -> None:
        self._last_battle_value = [0] * self._agent_num

    def _get_battle_value(self, raw_obs):
        minerals_ratio = 1.
        vespene_ratio = 1.
        return [
            int(
                np.sum(obs['score_by_category']['killed_minerals']) * minerals_ratio +
                np.sum(obs['score_by_category']['killed_vespene'] * vespene_ratio)
            ) for obs in raw_obs
        ]
