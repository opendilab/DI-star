# DI-star
This project is reimplementation of Alphastar (Only Zerg vs Zerg) based on OpenDILab, which contains:

- [x] Play demo and test code (try and play with our agent!)

- [x] First version of pre-trained SL and RL agent

- [x] Training code of Supervised Learning *(New! updated by 2022-01-31)*

- [x] Training code of Reinforcement Learning with self-play and league *(New! updated by 2022-01-31)*

- [ ] Professional version of pre-trained RL agent *(WIP)*


## Usage

### Installation
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

#### 1. Download models:
```bash
python -m distar.bin.download_model --rl
```
Note: Specify `--rl` or `--sl` to download reinforcement learning model or supervised model.

#### 2. Agent test
With the given model, we provide multiple tests with our agent.

##### Play against Agent
```bash
python -m distar.bin.play
```
It runs 2 StarCraftII instances. First one is controlled by our RL agent. Human player can play on the second one with full screen like normal game.

Note: 
- GPU and CUDA is required on default, add `--cpu` if you don't have these.
- Download RL model first or specify other models (like supervised model) with argument `--model1 <sl_model_path>`, pass either absolute path or relative path under distar/bin/
- In race cases, 2 StarCraftII instances may lose connection and agent won't issue any action. Please restart when this happens.

##### Agent vs Agent
```bash
python -m distar.bin.play --game_type agent_vs_agent
```
It runs 2 StarCraftII instances both controlled by our RL Agent, specify other model path with argument `--model1 <model1_path> --model2 <model2_path>`

##### Agent vs Bot
```bash
python -m distar.bin.play --game_type agent_vs_bot
```
RL agent plays against built-in elite bot.


## Training your own agent with our framework
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

### Training Tips
More configuration(e.g. batch size, learning rate, etc.)  could be found at `distar/bin/user_config.yaml`.

Training guide and baselines will be added soon. 

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
