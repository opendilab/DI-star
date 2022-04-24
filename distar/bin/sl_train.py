import os
import shutil
from distar.agent.import_helper import import_module
import argparse
from distar.ctools.utils import read_config
from distar.ctools.worker.actor.replay_actor import ReplayActor
from distar.ctools.worker.coordinator.coordinator import Coordinator, create_coordinator_app

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="sl_train")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), 'sl_user_config.yaml'))
    parser.add_argument("--type", default='learner')
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--init_method", type=str, default=None)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    args = parser.parse_args()
    config = read_config(args.config)
    config.common.type = 'sl'
    config.learner.data.remote = args.remote

    if args.init_method is not None:
        config.learner.use_distributed = True

    if args.data is not None:
        config.learner.data.train_data_file = args.data
    if args.type == 'learner':
        exp_config_path = os.path.join(os.getcwd(), 'experiments', config.common.experiment_name, 'sl_user_config.yaml')
        try:
            os.makedirs(os.path.dirname(exp_config_path))
        except:
            pass
        shutil.copyfile(args.config, exp_config_path)
        SLLearner = import_module(config.learner.agent, 'SLLearner')
        if config.learner.use_distributed:
            learner = SLLearner(config, "torch", args.init_method, args.rank, args.world_size)
        else:
            config.learner.use_warmup = True
            learner = SLLearner(config, "single_node")
        learner.run()
    elif args.type == 'coordinator':
        coordinator = Coordinator(config)
        coordinator_app = create_coordinator_app(coordinator)
        coordinator_app.run(host=config.communication.coordinator_ip, port=config.communication.coordinator_port, debug=False, use_reloader=False)
    elif args.type == 'replay_actor':
        ReplayDecoder = import_module(config.learner.agent, 'ReplayDecoder')
        replay_decoder = ReplayDecoder(config)
        replay_actor = ReplayActor(config, replay_decoder)
        replay_actor.run()
