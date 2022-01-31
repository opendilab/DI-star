import importlib

MODULE_PATHS = {
    'RLLearner': '.rl_learner',
    'SLLearner': '.sl_learner',
    'Agent': '.agent',
    'ReplayDecoder': '.replay_decoder'
}


def import_module(pipeline: str, name: str):
    abs_path = 'distar.agent.' + pipeline + MODULE_PATHS[name]
    module = getattr(importlib.import_module(abs_path), name)
    return module


if __name__ == '__main__':
    Agent = import_module('default', 'Agent')



