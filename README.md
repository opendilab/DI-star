[![qq](https://img.shields.io/badge/qq群-700157520-green)](mqqapi://card/show_pslcard?src_type=internal&version=1&uin=700157520&card_type=group&source=qrcode) <!-- # TODO：Only works on mobile version -->

![wechat公众号](https://img.shields.io/badge/wechat公众号-OpenDILab20210708-green) <!-- # TODO： hyperlink jump (seems not possible) -->

[![知乎](https://img.shields.io/badge/知乎-OpenDILab-green)](https://www.zhihu.com/people/opendilab)

[![email](https://img.shields.io/badge/Email-opendilab.contact@gmail.com-informational)](mailto:opendilab.contact@gmail.com)

[![slack](https://img.shields.io/badge/slack-OpenDILab-informational)](https://join.slack.com/t/opendilab/shared_invite/zt-v9tmv4fp-nUBAQEH1_Kuyu_q4plBssQ)

# DI-star

This project is reimplementation of Alphastar (Only Zerg vs Zerg) based on OpenDILab, which contains:

- [x] Play with trained agent

- [ ] Supervised Learning

- [ ] Reinforcement Learning


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
