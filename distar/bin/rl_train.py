import argparse
import os
import shutil
import torch
import platform
import torch.multiprocessing as multiprocessing

from distar.ctools.utils import read_config

from distar.actor import Actor
from distar.agent.import_helper import import_module

from distar.ctools.worker.coordinator.coordinator import Coordinator, create_coordinator_app
from distar.ctools.worker.league.league import League
from distar.ctools.worker.league.league_api import create_league_app
import time


def learner_run(config, args):
    import threading
    def start_app(learner):
        torch.set_num_threads(1)
        learner_app = learner.create_rl_learner_app(learner)
        learner_app.run(host=learner._ip, port=learner._port, debug=False, use_reloader=False)

    config.learner.player_id = args.player_id
    learner_config_path = os.path.join(os.getcwd(), 'experiments', config.common.experiment_name,
                                       config.learner.player_id, 'user_config.yaml')
    address_path = os.path.join(os.getcwd(), 'experiments', config.common.experiment_name, config.learner.player_id,
                                'address')
    cluster_address_path = os.path.join(os.getcwd(), 'experiments', config.common.experiment_name,
                                        config.learner.player_id, 'clusters')
    if os.path.exists(address_path):
        shutil.rmtree(address_path, ignore_errors=True)
    if os.path.exists(cluster_address_path):
        shutil.rmtree(cluster_address_path, ignore_errors=True)
    try:
        os.makedirs(os.path.dirname(learner_config_path))
        os.makedirs(cluster_address_path)
    except:
        pass
    shutil.copyfile(args.config, learner_config_path)
    RLLearner = import_module(config.learner.agent, 'RLLearner')
    if config.learner.use_distributed:
        learner = RLLearner(config, "torch", args.init_method, args.rank, args.world_size)
    else:
        learner = RLLearner(config, "single_node")

    thread = threading.Thread(target=start_app, args=(learner,), daemon=True)
    thread.start()
    learner.run()


def actor_run(config, args):
    if args.gpu_batch_inference == 'true':
        config.actor.gpu_batch_inference = True
    else:
        config.actor.gpu_batch_inference = False
    if config.actor.job_type == 'train':
        exp_name = config.common.experiment_name
        replay_path = os.path.abspath(config.env.replay_dir)
        new_replay_path = os.path.join(replay_path, exp_name)
        config.env.replay_dir = new_replay_path
    actor = Actor(config)
    actor.run()


def league_run(config, args):
    exp_config_path = os.path.join(os.getcwd(), 'experiments', config.common.experiment_name, 'user_config.yaml')
    try:
        os.makedirs(os.path.dirname(exp_config_path))
    except:
        pass
    shutil.copyfile(args.config, exp_config_path)

    time_label = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
    config_dir = os.path.join(os.getcwd(), 'experiments', config.common.experiment_name, 'config')
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    bk_exp_config_path = os.path.join(config_dir, f'user_config_{time_label}.yaml')
    shutil.copyfile(args.config, bk_exp_config_path)

    default_model_path = os.path.join(os.path.dirname(__file__), 'sl_model.pth')
    for idx, p in enumerate(config.league.active_players.checkpoint_path):
        if p == 'default':
            config.league.active_players.checkpoint_path[idx] = default_model_path
    for idx, p in enumerate(config.league.historical_players.checkpoint_path):
        if p == 'default':
            config.league.historical_players.checkpoint_path[idx] = default_model_path
    for idx, p in enumerate(config.league.active_players.teacher_path):
        if p == 'default':
            config.league.active_players.teacher_path[idx] = default_model_path
    league = League(config)
    league_app = create_league_app(league)
    league_app.run(host=config.communication.coordinator_ip, port=config.communication.league_port, debug=False,
                   use_reloader=False)


def coordinator_run(config, args):
    coordinator = Coordinator(config)
    coordinator_app = create_coordinator_app(coordinator)
    coordinator_app.run(host=config.communication.coordinator_ip, port=config.communication.coordinator_port,
                        debug=False, use_reloader=False)


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

    parser = argparse.ArgumentParser(description="rl_train")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), 'user_config.yaml'))
    parser.add_argument("--type", default=None)
    parser.add_argument("--task", default='bot')
    parser.add_argument("--player_id", default='MP0')
    parser.add_argument("--gpu_batch_inference", default='false')
    parser.add_argument("--init_method", type=str, default=None)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    args = parser.parse_args()
    config = read_config(args.config)
    config.common.type = 'rl'
    config.actor.traj_len = config.learner.data.trajectory_length
    config.learner.var_record_type = 'alphastar'
    if args.init_method is not None:
        config.learner.use_distributed = True
    if args.task == 'bot':
        config.league.vs_bot = True
    elif args.task == 'selfplay':
        config.league.vs_bot = False
    else:
        raise NotImplementedError
    context_str = 'spawn'
    mp_context = multiprocessing.get_context(context_str)
    
    if args.type is None:
        p_coordinator = mp_context.Process(target=coordinator_run, args=(config, args))
        p_coordinator.start()
        time.sleep(10)
        p_league = mp_context.Process(target=league_run, args=(config, args))
        p_league.start()
        time.sleep(10)
        p_learner = mp_context.Process(target=learner_run, args=(config, args, args.init_method, args.rank, args.world_size))
        p_learner.start()
        time.sleep(10)
        actor_run(config, args)
    elif args.type == 'coordinator':
        coordinator_run(config, args)
    elif args.type == 'learner':
        learner_run(config, args)
    elif args.type == 'league':
        league_run(config, args)
    elif args.type == 'actor':
        actor_run(config, args)
