class Agent:
    HAS_MODEL = False
    HAS_TEACHER_MODEL = False
    HAS_SUCCESSIVE_MODEL = False
    def __init__(self, cfg=None, env_id=0):
        pass

    def reset(self, map_name, race, game_info, obs):
        pass

    def step(self, obs):
        action = {'func_id': 0, 'skip_steps': 1, 
        'queued': False, 'unit_tags': [0], 
        'target_unit_tag': 0, 'location': [0, 0]}
        return [action]
