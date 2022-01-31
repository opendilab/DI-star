import torch
import os
import time
import random

import multiprocessing as mp
from distar.ctools.worker.coordinator.adapter import Adapter


def decode_loop(cfg, paths, decoder):
    torch.set_num_threads(1)
    first_data_flag = True  # make sure first batch has more data diversity
    adapter = Adapter(cfg)
    data_idx = 0
    player_idx = 0
    while True:
        path = paths[data_idx]
        if (data_idx + 1) % 100 == 0 and player_idx == 0:
            print('pid: {}, replays left: {}'.format(os.getpid(), len(paths) - data_idx))
        data = decoder.run(path, player_idx)
        player_idx = (player_idx + 1) % 2
        if player_idx == 0:
            data_idx += 1
        if data_idx == len(paths):
            print('replay actor job done')
            return
        if data is not None:
            if first_data_flag and len(data) < 600:
                time.sleep(random.randint(20, 50))
            first_data_flag = False
            while adapter.full():
                time.sleep(0.1)
            adapter.push(data, fs_type='pyarrow')
        

class ReplayActor(object):
    def __init__(self, cfg, decoder):
        self.whole_cfg = cfg
        self.cfg = cfg.learner.data
        ntasks, proc_id = 1, 0
        if 'SLURM_NTASKS' in os.environ:
            ntasks = int(os.environ['SLURM_NTASKS'])
        if 'SLURM_PROCID' in os.environ:
            proc_id = int(os.environ['SLURM_PROCID'])
        replay_paths = []
        if os.path.isfile(self.cfg.train_data_file):
            with open(self.cfg.train_data_file, 'r') as f:
                for l in f.readlines():
                    replay_paths.append(l.strip())
        elif os.path.isdir(self.cfg.train_data_file):
            for p in os.listdir(self.cfg.train_data_file):
                replay_paths.append(os.path.join(self.cfg.train_data_file, p))
        random.seed(233)
        expand_paths = []
        for i in range(self.cfg.epochs):
            random.shuffle(replay_paths)
            expand_paths += replay_paths
        replay_paths = expand_paths
        total_len = len(replay_paths) 
        per_len = total_len // ntasks
        replay_paths = replay_paths[proc_id * per_len: (proc_id + 1) * per_len]
        per_len = len(replay_paths) // self.cfg.replay_actor_num_workers
        print(f'replay actor start, totoal replay number: {total_len}, task id: {proc_id}, ntasks: {ntasks}, task len: {len(replay_paths)}, per proc len: {per_len}')
        self.procs = []
        for i in range(self.cfg.replay_actor_num_workers):
            per_proc_paths = replay_paths[i * per_len: (i + 1) * per_len]
            p = mp.Process(target=decode_loop, args=(self.whole_cfg, per_proc_paths, decoder), daemon=True)
            p.start()
            self.procs.append(p)
    
    def run(self):
        for p in self.procs:
            p.join()


    
    
