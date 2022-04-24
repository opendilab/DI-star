## Build a brand new agent
1. Implement the **step** function in distar/agent/template/agent.py.
This function should receive an observation and return a list of actions.

    the observation is a dict with 3 fields: 

    - **raw_obs**: contains observation in form of protobuf ([details](https://github.com/Blizzard/s2client-proto/blob/master/s2clientprotocol/sc2api.proto#L359)).

    - **opponent_obs**: contains opponent's observation similar to raw_obs.

    - **action_result**: action result of last action the agent made([details](https://github.com/Blizzard/s2client-proto/blob/master/s2clientprotocol/error.proto#L6)).

    the action is a dict with 6 fields:

    - **func_id**: ability id from pysc2([details](../distar/pysc2/lib/actions.py#L1183)).

    - **skip_steps**: how many frames in game the agent doesn't need an observation, count from current frame.

    - **queued**: whether to put this action into queue(like human use shift).

    - **unit_tags**: a list contains unit tags which are selected units in this action.

    -  **target_unit_tag**: a unit tag which is targetd unit in this action.

    -  **location**: the target location in this action.

2. Change the agent's name from `default` to `template` at [user_config.yaml](../distar/bin/user_config.yaml#L24)
3. `python -m distar.bin.play --game_type agent_vs_agent --model2 <agent you want to play against>`


## Modify our agent
1. Create a new branch named <dev>: `git checkout -b <dev>`
2. Modify any part of the default agent, here are some examples:
- Change the model size at [distar/agent/default/model/actor_critic_default_config.yaml](../distar/agent/default/model/actor_critic_default_config.yaml)
- Change input features at [distar/agent/default/lib/features.py](../distar/agent/default/lib/features.py)
- Overwrite the agent's action in any certain situation at [distar/agent/default/agent.py](../distar/agent/default/agent.py#L306)
3. Copy the directory of default agent from main branch and rename it main.
4. Change the agent's name from `default` to `main` at [user_config.yaml](../distar/bin/user_config.yaml#L24)
5. `python -m distar.bin.play --game_type agent_vs_agent --model1 <agent you want to play against> --model2 <your own agent>`
