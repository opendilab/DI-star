# DI-star
This project is a reimplementation of Alphastar (Only Zerg vs Zerg) based on OpenDILab, which contains:

- [x] Plays with trained agents

- [ ] Supervised Learning

- [ ] Reinforcement Learning


## Usage

### Installation
#### 1.Install StarCraftII

- Download the retail version of StarCraftII: https://starcraft2.com

Note: There is no retail version on Linux, please follow the instruction [here](https://github.com/Blizzard/s2client-proto#downloads)

- Add SC2 installation path to environment variables(skip this if you use default installation path on MacOS or Windows, which is `C:\Program Files (x86)\StarCraft II` or `/Applications/StarCraft II`):

    - On MacOS or Linux, input this in terminal:

        ```shell
        export SC2PATH=<sc2/installation/path>
        ```

    - On Windows:
       1. Right-click the Computer icon and choose Properties, or in Windows Control Panel, choose System.
       2. Choose Advanced system settings.
        3. On the Advanced tab, click Environment Variables.
        4. Click New to create a new environment variable, input  sc2 installation path.
        5. After creating or modifying the environment variable, click Apply and then OK to have the change take effect.


#### 2.Install distar:

```bash
git clone https://github.com/opendilab/DI-star.git
cd DI-star
pip install -e .
```

#### 3.Install pytorch:

Pytorch Version 1.7.1 and CUDA is recommended, Follow instructions from [pytorch official site](https://pytorch.org/get-started/previous-versions/)



**Note: GPU is neccessary for decent performance in realtime agent test, you can also use pytorch without cuda, but no performance guaranteed due to inference latency on cpu.**

### Play

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

## Citation
```latex
@misc{distar,
    title={{DI-star: OpenDILab} Decision Intelligence in StarCraftII},
    author={DI-star Contributors},
    publisher = {GitHub},
    howpublished = {\url{https://github.com/opendilab/DI-star}},
    year={2021},
}
```

## License
DI-star released under the Apache 2.0 license.
