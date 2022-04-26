# DI-star
This project is a reimplementation (with a few improvements) of Alphastar based on OpenDILab, which contains:

- [x] Play demo and test code (try and play with our agent!)

- [x] First version of pre-trained SL and RL agent (only Zerg vs Zerg)

- [x] Training code of Supervised Learning and Reinforcement Learning *(updated by 2022-01-31)*

- [x] Training baseline with limited resource(one PC) and training guidance [here](docs/guidance_to_small_scale_training.md) *(New! updated 2022-04-24)*

- [x] Agents fought with [Harstem (YouTube)](https://www.youtube.com/watch?v=fvQF-24IpXs&t=813s)  *(updated by 2022-04-01)*

- [ ] More stronger pre-trained RL agents *(WIP)*


## Usage

Please star us (click ![stars - di-star](https://img.shields.io/github/stars/opendilab/di-star?style=social) button in the top-right of this page) to help DI-star agents to grow up faster :)

### Installation
For users unfamiliar with programming, follow instructions [here](https://github.com/opendilab/DI-star/raw/main/docs/installation.pdf)

Environment requirement:

- Python: 3.6-3.8




#### 1.Install StarCraftII

- Download the retail version of StarCraftII: https://starcraft2.com

Note: There is no retail version on Linux, please follow the instruction [here](https://github.com/Blizzard/s2client-proto#downloads)

- Add SC2 installation path to environment variables ```SC2PATH``` (skip this if you use default installation path on MacOS or Windows, which is `C:\Program Files (x86)\StarCraft II` or `/Applications/StarCraft II`):

    - On MacOS or Linux, input this in terminal:

        ```shell
        export SC2PATH=<sc2/installation/path>
        ```

    - On Windows:
       1. Right-click the Computer icon and choose Properties, or in Windows Control Panel, choose System.
       2. Choose Advanced system settings.
        3. On the Advanced tab, click Environment Variables.
        4. Click New to create a new environment variable, set ```SC2PATH``` as the sc2 installation location.
        5. After creating or modifying the environment variable, click Apply and then OK to have the change take effect.


#### 2.Install distar:

```bash
git clone https://github.com/opendilab/DI-star.git
cd DI-star
pip install -e .
```

#### 3.Install pytorch:

Pytorch Version 1.7.1 and CUDA is recommended, Follow instructions from [pytorch official site](https://pytorch.org/get-started/previous-versions/)



**Note: GPU is neccessary for decent performance in realtime agent test, you can also use pytorch without cuda, but no performance guaranteed due to inference latency on cpu.
Make sure you set SC2 at lowest picture quality before testing.**

### Play with pretrained agent

#### 1. Download StarCraftII version 4.10.0
Double click the file [data/replays/replay_4.10.0.SC2Replay](data/replays/replay_4.10.0.SC2Replay), StarCraftII version 4.10.0 will be automatically downloaded.

Note: We trained our models with versions from 4.8.2 to 4.9.3. Patch 5.0.9 has came out in March 15, 2022, Some changes have huge impact on performance, so we fix our version at 4.10.0 in evaluation.

#### 2. Download models:
```bash
python -m distar.bin.download_model --name rl_model
```
Note: Specify `rl_model` or `sl_model` after `--name` to download reinforcement learning model or supervised model.

Model list:
- `sl_model`: training with human replays, skill is equal to diamond players.
- `rl_model`: used as default, training with reinforcement learning, skill is equal to master or grandmaster.
- `Abathur`: one of reinforcement learning models, likes playing mutalisk. 
- `Brakk`:  one of reinforcement learning models, likes lingbane rush.
- `Dehaka`: one of reinforcement learning models, likes playing roach ravager.

#### 3. Agent test
With the given model, we provide multiple tests with our agent.

##### Play against Agent
```bash
python -m distar.bin.play
```
It runs 2 StarCraftII instances. First one is controlled by our RL agent. Human player can play on the second one with full screen like normal game.

Note: 
- GPU and CUDA is required on default, add `--cpu` if you don't have these.
- Download RL model first or specify other models (like supervised model) with argument `--model1 <model_name>`
- In race cases, 2 StarCraftII instances may lose connection and agent won't issue any action. Please restart when this happens.

##### Agent vs Agent
```bash
python -m distar.bin.play --game_type agent_vs_agent
```
It runs 2 StarCraftII instances both controlled by our RL Agent, specify other model path with argument `--model1 <model_name> --model2 <model_name>`

##### Agent vs Bot
```bash
python -m distar.bin.play --game_type agent_vs_bot
```
RL agent plays against built-in elite bot.


## Building your own agent with our framework
It is necessary to build different agents within one code base and still be able to make them play against each other.
We implement this by making actor and environment as common components and putting everything related to the agent into one directory.
The agent called default under distar/agent is an example of this. Every script under default uses relative import, which 
makes them portable to anywhere as a whole part. 

If you want to create a new agent with/without our default agent, follow instructions [here](docs/agent.md)

If you want to train a new agent with our framework, follow instructions below and [here](docs/guidance_to_small_scale_training.md) is a guidance with more details of the whole training pipeline.
### Supervised Learning
StarCraftII client is required for replay decoding, follow instructions above.
```bash
python -m distar.bin.sl_train --data <path>
```
*path* could be either a directory with replays or a file includes a replay path at each line.

Optionally, separating replay decoding and model training could be more efficient, run the three scripts in different terminals:
```bash
python -m distar.bin.sl_train --type coordinator
python -m distar.bin.sl_train --type learner --remote
python -m distar.bin.sl_train --type replay_actor --data <path>
```

For distributed training:
```bash
python -m distar.bin.sl_train --init_method <init_method> --rank <rank> --world_size <world_size>
or
python -m distar.bin.sl_train --type coordinator
python -m distar.bin.sl_train --type learner --remote --init_method <init_method> --rank <rank> --world_size <world_size>
python -m distar.bin.sl_train --type replay_actor --data <path>
```
Here is an example of training on a machine with 4 GPUs in remote mode:
```bash
# Run the following scripts in different terminals (windows).
python -m distar.bin.sl_train --type coordinator
# Assume 4 GPUs are on the same machine. 
# If your GPUs are on different machines, you need to configure the init_mehod's IP for each machine.
python -m distar.bin.sl_train --type learner --remote --init_method tcp://127.0.0.1 --rank 0 --world_size 4
python -m distar.bin.sl_train --type learner --remote --init_method tcp://127.0.0.1 --rank 1 --world_size 4
python -m distar.bin.sl_train --type learner --remote --init_method tcp://127.0.0.1 --rank 2 --world_size 4
python -m distar.bin.sl_train --type learner --remote --init_method tcp://127.0.0.1 --rank 3 --world_size 4
python -m distar.bin.sl_train --type replay_actor --data <path>
```
### Reinforcement Learning
Reinforcement learning will use supervised model as initial model, please download it first, StarCraftII client is also required.

##### 1. Training against bots in StarCraftII: 
```bash
python -m disatr.bin.rl_train
```

##### 2. Training with self-play
```bash
python -m disatr.bin.rl_train --task selfplay
```

Four components are used for RL training, just like SL training, they can be executed through different process:
```bash
python -m distar.bin.rl_train --type league --task selfplay
python -m distar.bin.rl_train --type coordinator
python -m distar.bin.rl_train --type learner
python -m distar.bin.rl_train --type actor
```

Distributed training is also supported like SL training.

### Chat group
Slack: [link](https://join.slack.com/t/opendilab/shared_invite/zt-v9tmv4fp-nUBAQEH1_Kuyu_q4plBssQ)


Discord server: [link](https://discord.gg/FJHFUvJg)

## Citation
```latex
@misc{distar,
    title={DI-star: An Open-sourse Reinforcement Learning Framework for StarCraftII},
    author={DI-star Contributors},
    publisher = {GitHub},
    howpublished = {\url{https://github.com/opendilab/DI-star}},
    year={2021},
}
```

## License
DI-star released under the Apache 2.0 license.
