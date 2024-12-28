# MetalStar: High-Performance Metal Performance Shaders MPS for StarCraft II AI

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](./LICENSE)

**AppleStar** is a fork of the [DI-star repository](https://github.com/opendilab/DI-star) by OpenDILab which is in turn inspired by DeepMind's AlphaStar. This fork adds support for Apple’s Metal Performance Shaders (MPS) on macOS.

## Versions
Python 3.10.0, torch 2.5.1, torchaudio 2.5.1

### Command Line Usage Examples  

Apple MPS (on Apple Silicon with MPS-capable PyTorch): Showing off different scenarios to use MPS (Metal) on Apple Silicon macOS; if MPS isn’t available, it falls back to CPU. And if you really want CPU only, you can pass --cpu.

#### Human vs Agent (Default) with MPS  
```bash
python play.py
```
A “human_vs_agent” match using rl_model.pth (since model1 is "default"), with MPS as long as your system supports it. If MPS isn’t there, it prints a warning and runs on CPU.

#### Human vs Agent with a Custom Model  
Suppose you have a file my_rl_model.pth in the same folder as play.py:  
```bash
python play.py --model1 my_rl_model
```  
This instructs the script to look for my_rl_model.pth. You’re still in “human_vs_agent” mode by default, so the AI uses your custom model, and you can go head-to-head as the human.

#### Agent vs Bot on MPS  
```bash
python play.py --game_type agent_vs_bot
```  
Here the AI model (model1) faces off against the built-in bot at difficulty bot10. If you want a lower-level bot, say bot7, do this:  
```bash
python play.py --model2 bot7 --game_type agent_vs_bot
```  
The script will interpret “bot7” as the bot difficulty rather than a model file on disk.  

#### Agent vs Agent  
Use two different models:  
```bash
python play.py --game_type agent_vs_agent --model1 rl_model --model2 sl_model
```  
Now you have a reinforcement-learning model against a supervised-learning model. Both will run on MPS (or CPU fallback) with no human players.  

##### Forcing CPU Mode
If you don’t want MPS for some reason (maybe you’re testing CPU performance), you can override:
```bash
python play.py --cpu
```
This forcibly uses the CPU, ignoring MPS even if it’s available.

#### Another Human vs Agent Example
Imagine you’ve trained some advanced RL named grandmaster_model.pth. You want to see if you can beat it:
```bash
python play.py --model1 grandmaster_model
```
It’ll attempt MPS first, default to “human_vs_agent,” and use grandmaster_model.pth for the AI side. Let the showdown begin!

That’s it! These examples should help you jump right into your preferred StarCraft II matchups, whether it’s a human player, a built-in bot, or a pair of AI models. Enjoy battling it out under Apple Metal (MPS) acceleration!

## macOS Installation

macOS prerequisites 
```
brew install python
brew install pip
brew install micromamba
micromamba create -n pytorch python=3.10
micromamba activate pytorch
micromamba install pytorch torchvision torchaudio -c pytorch -c conda-forge
```

DI-star prerequisites
Inside the Applestar directory, issue the following commands:
```
pip install -e .
```

## macOS Troubleshooting VideoCard related errors

On occasions that the game session did not terminate gracefully, there will be specific videocard related errors. In order to fix this, open battle.net and under Settings, choose the option to Restore In-Game Options.

## Rolling Updates
- [x] Added MPS support for model inference located at play.py *(updated on 2024-12-28)*
- [x] Upgraded to use the latest version of pytorch that supports MPS through simple updates upon importing torch._six such as import as inf->math.inf and string_classes->str  *(updated on 2024-12-28)*
- [x] Tested on Python 3.10.0, torch 2.5.1, torchaudio 2.5.1 *(updated on 2024-12-29)*
- [x] StarCraft version remains to be on 4.10.0 to maintain the game version with the rl_model's training *(updated on 2024-12-29)*  
- [ ] Add MPS support for MPS based distributed training

## License and Attribution
This project is licensed under the [Apache 2.0 License](./LICENSE). The original DI-star is (c) OpenDILab, and all work in this fork is (c) 2024 Jaymari Chua.

# DI-Star Overview
DI-star: A large-scale game AI distributed training platform specially developed for the StarCraft II. We've already trained grand-master AI！This project contains:
- [x] Play demo and test code (try and play with our agent!)
- [x] First version of pre-trained SL and RL agent (only Zerg vs Zerg)
- [x] Training code of Supervised Learning and Reinforcement Learning *(updated by 2022-01-31)*
- [x] Training baseline with limited resource(one PC) and training guidance [here](docs/guidance_to_small_scale_training.md) *(New! updated 2022-04-24)*
- [x] Agents fought with [Harstem (YouTube)](https://www.youtube.com/watch?v=fvQF-24IpXs&t=813s)  *(updated by 2022-04-01)*
- [ ] More stronger pre-trained RL agents *(WIP)*

## Usage

[Testing software on Windows](docs/installation.md) | [对战软件下载](docs/安装教程.md)

Please star us (click ![stars - di-star](https://img.shields.io/github/stars/opendilab/di-star?style=social) button in the top-right of this page) to help DI-star agents to grow up faster :)

### Installation

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
- `Zagara`: one of reinforcement learning models, likes roach rush.

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


Discord server: [link](https://discord.gg/dkZS2JF56X)

## Recommended Citation MetalStar and DI-star (upstream)
```latex
@misc{distar,
    title={DI-star: An Open-sourse Reinforcement Learning Framework for StarCraftII},
    author={DI-star Contributors},
    publisher = {GitHub},
    howpublished = {\url{https://github.com/opendilab/DI-star}},
    year={2021},
}

@misc{metalstar,
    title={MetalStar: High-Performance Metal Performance Shaders MPS for StarCraft II AI},
    author={Jaymari Chua},
    publisher={GitHub},
    howpublished={\url{https://github.com/jaymarichua/MetalStar}},
    year={2024},
}
```

## Information
This project is forked from opendilab/DI-star, created by OpenDILab, and Applestar is focused on adding macOS Metal (MPS) acceleration. Applestar is based on DI-star, which is released under the Apache 2.0 license at the time of this fork.

