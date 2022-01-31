import argparse
import os
import shutil
import torch

from distar.actor import Actor
import warnings
from distar.ctools.utils import read_config
warnings.filterwarnings("ignore", message="Setting attributes on ParameterList is not supported.")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model1', type=str,
                        default=None,
                        help='must specify model path',
                        )
    parser.add_argument('--model2', type=str,
                        default=None,
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
        shutil.copytree(os.path.join(os.path.dirname(__file__), '../envs/maps/Ladder2019Season2'), os.path.join(sc2path, 'Maps/Ladder2019Season2'))

    user_config = read_config(os.path.join(os.path.dirname(__file__), 'user_config.yaml'))
    user_config.actor.job_type = 'eval_test'
    user_config.env.realtime = True
    args = get_args()
    default_model_path = os.path.join(os.path.dirname(__file__), 'rl_model.pth')
    if args.model1 is not None:
        model1 = os.path.join(os.path.join(os.path.dirname(__file__), args.model1))
        user_config.actor.model_paths['model1'] = model1
    else:
        model1 = user_config.actor.model_paths['model1']
    if user_config.actor.model_paths['model1'] == 'default':
        user_config.actor.model_paths['model1'] = default_model_path
        model1 = default_model_path

    if args.model2 is not None:
        model2 = os.path.join(os.path.join(os.path.dirname(__file__), args.model2))
        user_config.actor.model_paths['model2'] = model2
    else:
        model2 = user_config.actor.model_paths['model2']
    if user_config.actor.model_paths['model2'] == 'default':
        user_config.actor.model_paths['model2'] = default_model_path
        model2 = default_model_path

    assert os.path.exists(model1), 'model1 file : {} does not exist, please download model first!'.format(model1)
    assert os.path.exists(model2), 'model2 file : {} does not exist, please download model first!'.format(model2)
    if not args.cpu:
        assert torch.cuda.is_available(), 'cuda is not available, please install cuda first!'
        user_config.actor.use_cuda = True
    else:
        user_config.actor.use_cuda = False
        print('warning! cuda is not activate, this will cause significant agent performance degradation!')
    assert args.game_type in ['agent_vs_agent', 'agent_vs_bot', 'human_vs_agent'], 'game_type only support agent_vs_agent or agent_vs_bot or human_vs_agent!'
    if args.game_type == 'agent_vs_agent':
        user_config.env.player_ids = [os.path.basename(model1).split('.')[0], os.path.basename(model1).split('.')[1]]
    elif args.game_type == 'agent_vs_bot':
        user_config.env.player_ids = [os.path.basename(model1).split('.')[0], 'bot7']
    elif args.game_type == 'human_vs_agent':
        user_config.actor.player_ids = ['model1']
        user_config.env.player_ids = [os.path.basename(model1).split('.')[0], 'human']

    actor = Actor(user_config)
    actor.run()
