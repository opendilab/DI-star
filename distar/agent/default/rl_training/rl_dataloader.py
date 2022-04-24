import os.path as osp
import pickle
import platform
import queue
import pickle
import platform
import queue
import re
import threading
import traceback
import time
import random
from typing import Iterable, Callable, Any, Sequence
import traceback
import numpy as np
import requests
import torch
import torch.multiprocessing as tm
import torch.nn.functional as F

from ..lib.features import SPATIAL_SIZE, MAX_ENTITY_NUM, MAX_SELECTED_UNITS_NUM


from distar.ctools.data.collate_fn import default_collate_with_dim
from distar.ctools.torch_utils import sequence_mask
from distar.ctools.torch_utils import to_device
from distar.ctools.worker.coordinator.adapter import Adapter


def flat(data):
    if isinstance(data, torch.Tensor):
        return torch.flatten(data, start_dim=0, end_dim=1)  # (1, (T+1) * B)
    elif isinstance(data, dict):
        new_data = {}
        for k, val in data.items():
            new_data[k] = flat(val)
        return new_data
    elif isinstance(data, Sequence):
        new_data = [flat(v) for v in data]
        return new_data
    else:
        print(type(data))


def collate_fn(traj_batch):
    # data list of list, with shape batch_size, unroll_len
    # find max_entity_num in data_batch
    max_entity_num = max(
        [len(traj_data['entity_info']['x']) for traj_data_list in traj_batch for traj_data in traj_data_list])
    # max_entity_num = MAX_ENTITY_NUM

    # padding entity_info in observatin, target_unit, selected_units, mask
    traj_batch = [[padding_entity_info(traj_data,max_entity_num) for traj_data in traj_data_list] for traj_data_list in
                  traj_batch]

    data = [default_collate_with_dim(traj_data_list) for traj_data_list in traj_batch]

    batch_size = len(data)
    unroll_len = len(data[0]['step'])
    data = default_collate_with_dim(data, dim=1)

    new_data = {}
    for k, val in data.items():
        if k in ['spatial_info',
                 'entity_info',
                 'scalar_info',
                 'entity_num',
                 'entity_location',
                 'hidden_state', 'value_feature']:
            new_data[k] = flat(val)
        else:
            new_data[k] = val
    new_data['aux_type'] = batch_size
    new_data['batch_size'] = batch_size
    new_data['unroll_len'] = unroll_len
    return new_data


def worker_loop(cfg, data_queue, collate_fn, batch_size) -> None:
    player_id = cfg.learner.player_id
    adapter = Adapter(cfg=cfg)
    torch.set_num_threads(1)
    worker_num = cfg.communication.adapter_traj_worker_num

    buffer_size = cfg.learner.data.get('buffer_size', batch_size)
    buffer_size = max(buffer_size, cfg.learner.data.batch_size)
    data = adapter.pull(token=player_id + 'traj', fs_type='nppickle', sleep_time=0.5, size=buffer_size, worker_num=worker_num)

    data = data + data[:(batch_size // 2 + 1)]
    while True:
        if data_queue.full():
            time.sleep(0.1)
            continue
        try:
            t = time.time()
            batch_data = collate_fn(data[:batch_size])
            # print('collate time', time.time() - t)
            t = time.time()
            data_queue.put(batch_data)
            # print('put time', time.time() - t)
        except Exception as e:
            print(f"can't collate, Error:{e}", flush=True)
            print(''.join(traceback.format_tb(e.__traceback__)), flush=True)
            # torch.save(data, 'wrong_data_{}.pth'.format(time.time()))
        data = data[batch_size:]
        left_num = buffer_size - len(data)
        random.shuffle(data)
        if left_num > 0:
            new_data = adapter.pull(player_id + 'traj', fs_type='nppickle', sleep_time=0.2, size=left_num, worker_num=worker_num)
            data = new_data + data + new_data


def _cuda_loop(cuda_queue, data_queue, device) -> None:
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        while True:
            while data_queue.empty():
                time.sleep(0.05)
            t = time.time()
            data = data_queue.get()
            # print('get time', time.time() - t)
            while cuda_queue.full():
                time.sleep(0.1)
            t = time.time()
            data = to_device(data, device)
            # print('cuda time', time.time() - t)
            cuda_queue.put(data)


class RLDataLoader(object):
    def __init__(
            self,
            cfg,
    ) -> None:
        self._whole_cfg = cfg
        self.cfg = self._whole_cfg.learner.data
        self.use_cuda = self._whole_cfg.learner.use_cuda and torch.cuda.is_available()
        self.use_async_cuda = self.cfg.get('use_async_cuda', True)
        self.num_workers = self.cfg.get('num_workers', 1)
        self._data_path_queue_size = self.cfg.get('data_path_queue_size', 3)
        self.batch_size = self.cfg.get('batch_size', 2)
        self.fs_type = self.cfg.get('fs_type', 'pickle')
        self.device = torch.cuda.current_device() if self.use_cuda else 'cpu'
        self.cache_data = None
        self.collate_fn = collate_fn
        if self.num_workers < 0:
            raise ValueError(
                'num_workers should be non-negative; '
                'use num_workers = 0 or 1 to disable multiprocessing.'
            )

        context_str = 'spawn' if platform.system().lower() == 'windows' else 'forkserver'
        mp_context = tm.get_context(context_str)
        self.data_queue = mp_context.Queue(maxsize=self.num_workers)
        self.workers = [mp_context.Process(target=worker_loop, args=(self._whole_cfg, self.data_queue, self.collate_fn, self.batch_size),
                                           daemon=True) for _ in range(self.num_workers)]
        for w in self.workers:
            w.start()

        # cuda thread
        if self.use_async_cuda and self.use_cuda:
            # the queue to store processed cuda data, user will get data from it if use cuda
            self.cuda_queue = mp_context.Queue(maxsize=1)
            self.cuda_process = mp_context.Process(target=_cuda_loop, args=(
                self.cuda_queue, self.data_queue, self.device,),
                                                  daemon=True)
            self.cuda_process.start()
        elif self.use_cuda:
            self.stream = torch.cuda.Stream(device=self.device)

    def __iter__(self) -> Iterable:
        return self

    def __next__(self) -> Any:
        if self.use_cuda:
            if self.use_async_cuda:
                while self.cuda_queue.empty():
                    time.sleep(0.005)
                del self.cache_data
                # start = time.time()
                self.cache_data = self.cuda_queue.get()
                # print(f'get data from cuda queue time:{time.time() - start}')
                return self.cache_data
            else:
                with torch.cuda.stream(self.stream):
                    del self.cache_data
                    self.cache_data = self.data_queue.get()
                    self.cache_data = to_device(self.cache_data, self.device)
                return self.cache_data
        else:
            return self.data_queue.get()

    def close(self):
        processes =  [self.ask_data_loop] + self.workers
        if self.use_async_cuda and self.use_cuda:
            processes += [self.cuda_process]
        for p in processes:
            p.terminate()
        for p in processes:
            p.join()
        time.sleep(1)
        print('has already close all subprocess in RLdataloader')
        return True


def padding_entity_info(traj_data, max_entity_num):
    traj_data.pop('map_name', None)
    entity_padding_num = max_entity_num - len(traj_data['entity_info']['x'])
    if 'entity_embeddings' in traj_data.keys():
        traj_data['entity_embeddings']= torch.nn.functional.pad(traj_data['entity_embeddings'], (0,0,0, entity_padding_num),'constant', 0)

    for k in traj_data['entity_info'].keys():
        traj_data['entity_info'][k] = torch.nn.functional.pad(traj_data['entity_info'][k], (0, entity_padding_num),
                                                              'constant', 0)
    if 'action_info' in traj_data:
        su_padding_num = MAX_SELECTED_UNITS_NUM - traj_data['teacher_logit']['selected_units'].shape[0]

        traj_data['mask']['selected_units_mask'] = sequence_mask(traj_data['selected_units_num'].unsqueeze(dim=0),
                                                                 max_len=MAX_SELECTED_UNITS_NUM).squeeze(dim=0)
        traj_data['action_info']['selected_units'] = torch.nn.functional.pad(traj_data['action_info']['selected_units'],
                                                                             (0,MAX_SELECTED_UNITS_NUM -
                                                                              traj_data['action_info'][
                                                                                  'selected_units'].shape[-1]),
                                                                             'constant', 0)
                          
        traj_data['behaviour_logp']['selected_units'] = torch.nn.functional.pad(
            traj_data['behaviour_logp']['selected_units'], (0,su_padding_num,), 'constant',
            -1e9)

        traj_data['teacher_logit']['selected_units'] = torch.nn.functional.pad(
            traj_data['teacher_logit']['selected_units'], (0, entity_padding_num, 0, su_padding_num,), 'constant', -1e9)
        traj_data['teacher_logit']['target_unit'] = torch.nn.functional.pad(traj_data['teacher_logit']['target_unit'],
                                                                            (0, entity_padding_num), 'constant', -1e9) 
        #successive data
        #traj_data['successive_logit']['selected_units'] = torch.nn.functional.pad(
        #    traj_data['successive_logit']['selected_units'], (0, entity_padding_num, 0, su_padding_num,), 'constant', -1e9)
        #traj_data['successive_logit']['target_unit'] = torch.nn.functional.pad(traj_data['successive_logit']['target_unit'],
        #                                                                   (0, entity_padding_num), 'constant', -1e9) 

        traj_data['mask']['selected_units_logits_mask'] = sequence_mask(traj_data['entity_num'].unsqueeze(dim=0) + 1,
                                                                        max_len=max_entity_num + 1).squeeze(dim=0)
        traj_data['mask']['target_units_logits_mask'] = sequence_mask(traj_data['entity_num'].unsqueeze(dim=0),
                                                                      max_len=max_entity_num).squeeze(dim=0)

    return traj_data
