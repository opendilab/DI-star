import argparse
import os
import shutil

import torch
from distar.worker.actor.eval_actor import ASEvalActor
import warnings
warnings.filterwarnings("ignore", message="Setting attributes on ParameterList is not supported.")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model1', type=str,
                        default='rl_model.pth',
                        help='must specify model path',
                        )
    parser.add_argument('--model2', type=str,
                        default='rl_model.pth',
                        help='must specify model path'
                        )
    parser.add_argument('--cpu', action="store_true", help='use cpu inference')
    parser.add_argument('--game_type', type=str, default='human_vs_agent')
    return parser.parse_args()

if __name__ == '__main__':
    if os.path.exists(r'C:\Program Files (x86)\StarCraft II'):
        sc2path = r'C:\Program Files (x86)\StarCraft II'
    elif os.path.exists('/Applications/StarCraft II'):
        sc2path = '/Applications/StarCraft II'
    else:
        assert 'SC2PATH' in os.environ.keys(), 'please add StarCraft2 installation path to your environment variables!'
        sc2path = os.environ['SC2PATH']
        assert os.path.exists(sc2path), 'SC2PATH: {} does not exist!'.format(sc2path)
    if not os.path.exists(os.path.join(sc2path, 'Maps/Ladder2019Season2')):
        os.makedirs(os.path.join(sc2path, 'Maps/Ladder2019Season2'))
    if not os.path.exists(os.path.join(sc2path, 'Maps/Ladder2019Season2/NewRepugnancyLE.SC2Map')):
        shutil.copyfile(os.path.join(os.path.dirname(__file__), '../data/map/NewRepugnancyLE.SC2Map'),
                        os.path.join(sc2path, 'Maps/Ladder2019Season2/NewRepugnancyLE.SC2Map'))
    if not os.path.exists(os.path.join(sc2path, 'Maps/Ladder2019Season2/NewRepugnancyLE.SC2Map')):
        if not os.path.exists(os.path.join(sc2path, 'Maps')):
            os.mkdir(os.path.join(sc2path, 'Maps'))
        if not os.path.exists(os.path.join(sc2path, 'Maps/Ladder2019Season2')):
            os.mkdir(os.path.join(sc2path, 'Maps/Ladder2019Season2'))
        if not os.path.exists(os.path.join(sc2path, 'Maps/Ladder2019Season2')):
            os.mkdir(os.path.join(sc2path, 'Maps/Ladder2019Season2'))
    args = get_args()
    model1 = os.path.join(os.path.join(os.path.dirname(__file__), args.model1))
    model2 = os.path.join(os.path.join(os.path.dirname(__file__), args.model2))
    assert os.path.exists(model1), 'model1 file : {} does not exist, please download model first!'.format(model1)
    assert os.path.exists(model2), 'model2 file : {} does not exist, please download model first!'.format(model2)
    if not args.cpu:
        assert torch.cuda.is_available(), 'cuda is not available, please install cuda first!'
    else:
        'warning! cuda is not activate, this will cause significant agent performance degradation!'
    assert args.game_type in ['agent_vs_agent', 'agent_vs_bot', 'human_vs_agent'], 'game_type only support agent_vs_agent or agent_vs_bot or human_vs_agent!'
    actor = ASEvalActor(model1=model1, model2=model2, cuda=not args.cpu, game_type=args.game_type)
    actor.run()