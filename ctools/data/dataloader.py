import threading
import queue
from typing import Iterable, Callable, Optional, Any, Union

import time
import torch
from copy import deepcopy
from ctools.torch_utils import to_device
from .collate_fn import default_collate


class AsyncDataLoader(object):
    def __init__(
            self,
            data_source: Union[Callable, dict],
            batch_size: int,
            device: str,
            chunk_size: Optional[int] = None,
            collate_fn: Optional[Callable] = None,
            num_workers: int = 0,
            use_async=True,
            use_async_cuda=True,  # using aysnc cuda costs extra GPU memory
            max_reuse=0,
    ) -> None:
        self.use_async = use_async
        self.use_async_cuda = use_async_cuda
        self.data_source = data_source
        self.batch_size = batch_size
        self.device = device
        self.use_cuda = isinstance(self.device, int)
        if self.use_cuda:
            self.stream = torch.cuda.Stream()

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
        self.cache_data = None
        torch.set_num_threads(torch.get_num_threads() // 4)
        if self.use_async:
            self.queue_maxsize = 1
            self.worker_queue = queue.Queue(maxsize=1)
            self.data_queue = queue.Queue(maxsize=self.queue_maxsize)
            self.read_data_thread = threading.Thread(target=self.read_data_loop, args=(), daemon=True)
            self.read_data_thread.start()
            self.workers = [threading.Thread(target=self._worker_loop, args=(), daemon=True) for _ in range(self.num_workers)]
            for w in self.workers:
                w.start()

            # cuda thread
            if self.use_async_cuda:
                # the queue to store processed cuda data, user will get data from it if use cuda
                self.cuda_queue = queue.Queue(maxsize=self.queue_maxsize)
                self.cuda_thread = threading.Thread(target=self._cuda_loop, args=())
                self.cuda_thread.daemon = True
                self.cuda_thread.start()


    def __iter__(self) -> Iterable:
        """
        Overview:
            Return the iterable self as an iterator
        Returns:
            - self (:obj:`Iterable`): self as an iterator
        """
        return self

    def read_data_loop(self) -> None:
        while True:
            data = self.data_source(self.batch_size)
            while self.worker_queue.qsize() > 0:  # make sure only last queue has cache data
                time.sleep(0.1)
            self.worker_queue.put(data)

    def _worker_loop(self) -> None:
        print('dataloader worker start, threads:{}!!!!!!!!!!!!!!!!!!'.format(torch.get_num_threads()))
        while True:
            try:
                data = self.worker_queue.get()
                for i in range(len(data)):
                    data[i] = data[i]()
            except Exception as e:
                print(e)
                continue
            data = self.collate_fn(data)
            while self.data_queue.qsize() > 0:
                time.sleep(0.1)
            self.data_queue.put(data)

    def _cuda_loop(self) -> None:
        """
        Overview:
            Only when using cuda, would this be run as a thread through ``self.cuda_thread``.
            Get data from ``self.async_train_queue``, change its device and put it into ``self.cuda_queue``
        """
        with torch.cuda.stream(self.stream):
            while True:
                while self.cuda_queue.qsize() > 0:
                    time.sleep(0.1)
                data = self.data_queue.get()
                data = to_device(data, self.device)
                self.cuda_queue.put(data)

    def sync_loop(self):
        while True:
            try:
                data = self.data_source(self.batch_size, paths)
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
                if self.reuse_count == self.max_reuse:
                    if self.use_async_cuda:
                        self.cache_data = self.cuda_queue.get()
                    else:
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


