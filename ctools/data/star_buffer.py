from collections import deque
import time
import os
from ctools.utils import remove_file
import queue
import threading

class StarBuffer(object):
    def __init__(self, cfg, name):
        self.name = name
        self.meta_maxlen = cfg.meta_maxlen
        self.min_sample_ratio = cfg.min_sample_ratio
        self.data = deque(maxlen=self.meta_maxlen)
        self.total_data_count = 0
        self.path_traj = cfg.path_traj
        self.delete_deque = deque()
        self.delete_thread = threading.Thread(target=self.delete_func, daemon=True)
        self.delete_thread.start()

    def push_data(self, data):
        if len(self.data) == self.meta_maxlen:
            metadata = self.data.popleft()
            file_path = os.path.join(self.path_traj, metadata['traj_id'])
            self.delete_deque.append(file_path)

        self.data.append(data)
        if self.total_data_count < self.min_sample_ratio:
            self.total_data_count += 1

    def sample(self, batch_size):
        if self.total_data_count < self.min_sample_ratio:
            print(f'not enough data, required {self.min_sample_ratio} to begin, now has {self.total_data_count}!')
            return None
        data = []
        for i in range(batch_size):
            while True:
                try:
                    data.append(self.data.popleft())
                    break
                except IndexError:
                    time.sleep(0.1)
        return data

    def delete_func(self):
        while True:
            if len(self.delete_deque) > 0:
                path = self.delete_deque.pop()
                os.remove(path)
                print(self.name, 'data too many, delete file:', path)
            else:
                time.sleep(1)
