import torch
import random
import time
import threading
import os

from collections import deque
from easydict import EasyDict
from torch.utils.data._utils.collate import default_collate
import torch.multiprocessing as mp

from distar.ctools.torch_utils import to_device
from distar.ctools.utils.dist_helper import get_rank, get_world_size
from distar.agent.default.lib.features import fake_step_data
from distar.agent.default.replay_decoder import ReplayDecoder
from distar.ctools.worker.coordinator.adapter import Adapter


def send_data(worker_queue, main_traj_pipes_c, worker_index, data, shared_step_data, trajectory_length):
    worker_queue.put(worker_index)
    start = 0
    replay_length = len(data)
    
    while True:
        if main_traj_pipes_c.poll():
            batch_index = main_traj_pipes_c.recv()
            end = min(start + trajectory_length, replay_length)
            
            for i in range(start, end):
                step_data = data[i]
                data_idx = batch_index * trajectory_length + i - start
                entity_num = step_data['entity_num']
                selected_units_num = step_data['selected_units_num']
                for k, v in step_data.items():
                    if isinstance(v, torch.Tensor):
                        shared_step_data[k][data_idx].copy_(step_data[k])
                    elif isinstance(v, dict):
                        for _k, _v in v.items():
                            if _k in shared_step_data[k].keys():
                                if k == 'action_info' and _k == 'selected_units':
                                    if selected_units_num > 0:
                                        shared_step_data[k][_k][data_idx, :selected_units_num].copy_(step_data[k][_k])
                                elif k == 'entity_info':
                                    shared_step_data[k][_k][data_idx, :entity_num].copy_(step_data[k][_k])
                                elif k == 'spatial_info':
                                    if 'effect' in _k:
                                        shared_step_data[k][_k][data_idx].copy_(step_data[k][_k])
                                    else:
                                        h, w = step_data[k][_k].shape
                                        shared_step_data[k][_k][data_idx] *= 0
                                        shared_step_data[k][_k][data_idx, :h, :w].copy_(step_data[k][_k])
                                else:
                                    shared_step_data[k][_k][data_idx].copy_(step_data[k][_k])

            # mask
            for i in range(end, start + trajectory_length):
                data_idx = batch_index * trajectory_length + i - start
                for k in shared_step_data['action_mask'].keys():
                    shared_step_data['action_mask'][k][data_idx].copy_(torch.tensor(0).bool())

            new_episode = True if start == 0 else False
            end_episode = True if end == replay_length else False
            main_traj_pipes_c.send((new_episode, end - start, end_episode))
            start = end
            if end_episode:
                return
        else:
            time.sleep(0.002)


def worker_loop(cfg, paths, main_traj_pipes_c, shared_step_data, worker_queue, worker_index):
    torch.set_num_threads(1)
    if cfg.learner.data.remote:
        adapter = Adapter(cfg)
        print('remote data init, waiting for data...')
    else:
        replay_decoder = ReplayDecoder(cfg)
        data_idx = 0
        player_idx = 0
    while True:
        while True:
            if cfg.learner.data.remote:
                data = adapter.pull(fs_type='pyarrow', sleep_time=0.2)
            else:
                data = replay_decoder.run(paths[data_idx], player_idx)
                player_idx = (player_idx + 1) % 2
                if player_idx == 0:
                    data_idx += 1
                if data_idx == len(paths):
                    print('ran out of data, training is done!')
                    return
            if data is not None:
                break
        send_data(worker_queue, main_traj_pipes_c, worker_index, data, shared_step_data, cfg.learner.data.trajectory_length)


class SLDataloader:
    def __init__(self, cfg):
        torch.set_num_threads(1)
        self.use_cuda = cfg.learner.use_cuda
        self.device = torch.cuda.current_device() if self.use_cuda else None
        self.cfg = cfg.learner.data
        self.batch_size = self.cfg.batch_size
        self.trajectory_length = self.cfg.trajectory_length
        data_path = []
        if os.path.isfile(self.cfg.train_data_file):
            with open(self.cfg.train_data_file, 'r') as f:
                for l in f.readlines():
                    data_path.append(l.strip())
        elif os.path.isdir(self.cfg.train_data_file):
            for p in os.listdir(self.cfg.train_data_file):
                data_path.append(os.path.join(self.cfg.train_data_file, p))
        data_paths = data_path * self.cfg.get('epochs', 100)
        random.seed(233)
        random.shuffle(data_paths)
        if cfg.learner.use_distributed:
            rank, world_size = get_rank(), get_world_size()
        else:
            rank, world_size = 0, 1
        self.rank = rank
        per_rank_size = len(data_paths) // world_size

        self.cur_cache_idx = 0
        self.shared_step_data = fake_step_data(share_memory=True, batch_size=self.trajectory_length * self.batch_size)
        data_paths = data_paths[rank * per_rank_size: (rank + 1) * per_rank_size]

        self.worker_queue = mp.Queue()
        main_traj_pipes = [mp.Pipe() for _ in range(self.cfg.num_workers)]
        self.main_traj_pipes_p = [main_traj_pipes[i][0] for i in range(self.cfg.num_workers)]
        main_traj_pipes_c = [main_traj_pipes[i][1] for i in range(self.cfg.num_workers)]

        per_worker_size = len(data_paths) // self.cfg.num_workers
        for i in range(self.cfg.num_workers):
            worker_paths = data_paths[i * per_worker_size: (i + 1) * per_worker_size]
            mp.Process(target=worker_loop,
                       args=(cfg, worker_paths, main_traj_pipes_c[i], self.shared_step_data, self.worker_queue, i), daemon=True).start()

        self.worker_indices = [None] * self.batch_size
        for i in range(self.batch_size):
            self.worker_indices[i] = self.worker_queue.get()
            self.main_traj_pipes_p[self.worker_indices[i]].send(i)

    def __iter__(self):
        return self

    def __next__(self):
        new_episodes = []
        traj_lens = []
        for i in range(self.batch_size):
            new_episode, traj_len, end_episode = self.main_traj_pipes_p[self.worker_indices[i]].recv()
            new_episodes.append(new_episode)
            traj_lens.append(traj_len)
            if end_episode:
                self.worker_indices[i] = self.worker_queue.get()
                print(f'rank: {self.rank}, left data: {self.worker_queue.qsize()}')

        batch_data = self.shared_step_data
        if self.use_cuda:
            batch_data = to_device(batch_data, self.device)
        for i in range(self.batch_size):
            self.main_traj_pipes_p[self.worker_indices[i]].send(i)
        batch_data['traj_lens'] = traj_lens
        batch_data['new_episodes'] = new_episodes
        return batch_data


class FakeDataloader:
    def __init__(self, cfg):
        self.use_cuda = cfg.learner.use_cuda
        self.device = torch.cuda.current_device() if self.use_cuda else None
        self.batch_size = cfg.learner.data.batch_size
        self.traj_len = cfg.learner.data.trajectory_length

    def __iter__(self):
        return self

    def __next__(self):
        new_episodes = [random.randint(0, 1)] * self.batch_size
        traj_lens = [self.traj_len] * self.batch_size
        data = []
        for _ in range(self.batch_size * self.traj_len):
            data.append(fake_step_data())
        data = default_collate(data)

        if self.use_cuda:
            data = to_device(data, self.device)
        data['traj_lens'] = traj_lens
        data['new_episodes'] = new_episodes
        return data
