import os
import queue
import sys
import threading
import time
import traceback
import logging
from typing import Iterable, Callable, Optional, Any, Union

import requests
import torch
import torch.multiprocessing as tm
from ctools.torch_utils import to_device
from ctools.utils.compression_helper import get_data_decompressor
from ctools.utils.file_helper import remove_file
from ctools.utils.log_helper import TextLogger

from .collate_fn import default_collate


def _read_data_loop(worker_queue, url_prefix, batch_size, learner_uid=0) -> None:
    logger = logging.getLogger('default_logger')
    api = 'coordinator/ask_for_metadata'
    while True:
        while worker_queue.full():  # make sure only last queue has cache data
            time.sleep(0.1)
        d = {'learner_uid': learner_uid, 'batch_size': batch_size}
        while True:
            t = time.time()
            try:
                result = requests.post(url_prefix + api, json=d).json()
                if result is not None and result['code'] == 0:
                    metadata = result['info']
                    logger.info('ask for data cost time: {}'.format(time.time() - t))
                    if metadata is not None:
                        assert isinstance(metadata, list)
                        data = [
                            m['traj_id']
                            for m in metadata
                        ]
                        break
            except Exception as e:
                logger.error(''.join(traceback.format_tb(e.__traceback__)))
                logger.error("[error] api({}): {}".format(api, sys.exc_info()))
                time.sleep(0.05)
        worker_queue.put(data)


def _worker_loop(data_queue, collate_fn, path_traj, url_prefix, learner_uid, batch_size,num_workers,cur_batch,
                 decompress_type='none', ) -> None:
    logger = TextLogger(path='.log', name='data_loader_read_loop')
    torch.set_num_threads(4)
    worker_queue = queue.Queue(maxsize=3)
    read_data_thread = threading.Thread(
        target=_read_data_loop, args=(worker_queue, url_prefix, batch_size, learner_uid), daemon=True)
    read_data_thread.start()
    print('start read loop')

    decompressor = get_data_decompressor(decompress_type)

    while True:
        if worker_queue.empty() or data_queue.full():
            time.sleep(0.1)
            continue
        try:
            data = worker_queue.get()
            load_t = time.time()
            for i in range(len(data)):
                filename = data[i]
                filepath = os.path.join(path_traj, filename)
                data[i] = torch.load(filepath, map_location='cpu')
                remove_file(filepath)

        except Exception as e:
            print(e)
            continue
        
        data = collate_fn(data)
        while data_queue.full():
            time.sleep(0.01)
        data_queue.put(data)
        with cur_batch.get_lock():
            cur_batch.value = (cur_batch.value + 1) % num_workers


def _cuda_loop(cuda_queue, data_queue, device) -> None:
    """
            Overview:
                Only when using cuda, would this be run as a thread through ``self.cuda_thread``.
                Get data from ``self.async_train_queue``, change its device and put it into ``self.cuda_queue``
    """
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        while True:
            
            while data_queue.empty():
                time.sleep(0.05)
            data = data_queue.get()
            while cuda_queue.full():
                time.sleep(0.05)
            t = time.time()
            data = to_device(data, device)
            cuda_queue.put(data)


class AsyncDataLoader(object):
    def __init__(
            self,
            data_source: Union[Callable, dict],
            batch_size: int,
            device: str,
            learner_uid: int,
            url_prefix: str,
            path_traj: str,
            collate_fn: Optional[Callable] = None,
            num_workers: int = 0,
            use_async=True,
            use_async_cuda=True,  # using aysnc cuda costs extra GPU memory
            max_reuse=0,
            decompress_type='none',
    ) -> None:
        self.url_prefix = url_prefix
        self.path_traj = path_traj
        self.use_async = use_async
        self.use_async_cuda = use_async_cuda
        self.data_source = data_source
        self.batch_size = batch_size
        self.device = device
        self.use_cuda = isinstance(self.device, int)
        self.logger = TextLogger(path='.log', name='data_loader')
        self.learner_uid = learner_uid
        self.cache_data = None
        self.decompress_type = decompress_type

        if collate_fn is None:
            self.collate_fn = default_collate
        else:
            self.collate_fn = collate_fn
        self.num_workers = num_workers
        if self.num_workers < 0:
            raise ValueError(
                'num_workers should be non-negative; '
                'use num_workers = 0 or 1 to disable multiprocessing.'
            )

        self.reuse_count = max_reuse
        self.max_reuse = max_reuse
        context_str = 'spawn'  # if platform.system().lower() == 'windows' else 'forkserver'
        mp_context = tm.get_context(context_str)
        mp_context_fork = tm.get_context('forkserver')
        self.cur_batch = mp_context_fork.Value('i', 0)
        if self.use_async:
            # self.queue_maxsize = self.num_workers
            self.data_queue = tm.Queue(maxsize=self.num_workers * 2)

            self.workers = [mp_context_fork.Process(target=_worker_loop,
                                                    args=(self.data_queue,
                                                          self.collate_fn,
                                                          self.path_traj,
                                                          self.url_prefix,
                                                          self.learner_uid,
                                                          self.batch_size,
                                                          self.num_workers,
                                                          self.cur_batch,
                                                          self.decompress_type,
                                                          ),
                                                    daemon=True) for _ in
                            range(self.num_workers)]
            for w in self.workers:
                w.start()

            # cuda thread
            if self.use_async_cuda:
                # the queue to store processed cuda data, user will get data from it if use cuda
                self.cuda_queue = queue.Queue(maxsize=1)
                self.cuda_thread = threading.Thread(target=_cuda_loop, args=(
                    self.cuda_queue, self.data_queue, self.device,),
                                                      daemon=True)
                self.cuda_thread.start()
            elif self.use_cuda:
                self.stream = torch.cuda.Stream(device=self.device)

    def __iter__(self) -> Iterable:
        """
        Overview:
            Return the iterable self as an iterator
        Returns:
            - self (:obj:`Iterable`): self as an iterator
        """
        return self

    def sync_loop(self):
        while True:
            try:
                data = self.data_source(self.batch_size)
                for i in range(len(data)):
                    data[i] = data[i]()
                break
            except:
                pass
        data = self.collate_fn(data)
        if self.use_cuda:
            data = to_device(data, self.device)
        return data

    def __next__(self) -> Any:
        """
        Overview:
            Return next data in the iterator. If use cuda, get from ``self.cuda_queue``;
            Otherwise, get from ``self.async_train_queue``.
        Returns:
            - data (:obj:`torch.Tensor`): next data in the iterator
        """
        if self.use_async:
            if self.use_cuda:
                if self.use_async_cuda:
                    if self.reuse_count == self.max_reuse:
                        while self.cuda_queue.empty():
                            time.sleep(0.01)
                        self.cache_data = self.cuda_queue.get()
                        self.reuse_count = 0
                    else:
                        self.reuse_count += 1
                    return self.cache_data
                else:
                    if self.reuse_count == self.max_reuse:
                        with torch.cuda.stream(self.stream):
                            del self.cache_data
                            self.cache_data = self.data_queue.get()
                            self.cache_data = to_device(self.cache_data, self.device)
                        self.reuse_count = 0
                    else:
                        self.reuse_count += 1
                    return self.cache_data
            else:
                return self.data_queue.get()
        else:
            return self.sync_loop()

