import copy
import os.path as osp
from threading import Thread
from typing import Union

from ctools.data.structure import PrioritizedBuffer, Cache
from ctools.utils import LockContext, LockContextType, read_config, deep_merge_dicts

default_config = read_config(osp.join(osp.dirname(__file__), 'replay_buffer_default_config.yaml')).replay_buffer


class ReplayBuffer:
    """
    Overview: reinforcement learning replay buffer, with priority sampling, data cache
    Interface: __init__, push_data, sample, update, run, close
    """

    def __init__(self, cfg: dict):
        """
        Overview: initialize replay buffer
        Arguments:
            - cfg (:obj:`dict`): config dict
        """
        self.cfg = deep_merge_dicts(default_config, cfg)
        max_reuse = self.cfg.max_reuse if 'max_reuse' in self.cfg.keys() else None
        delete_cache_length = cfg.get('delete_cache_length', 50)
        self.traj_len = cfg.get('traj_len', None)
        self.unroll_len = cfg.get('unroll_len', None)
        self._meta_buffer = PrioritizedBuffer(
            maxlen=self.cfg.meta_maxlen,
            max_reuse=max_reuse,
            min_sample_ratio=self.cfg.min_sample_ratio,
            alpha=self.cfg.alpha,
            beta=self.cfg.beta,
            enable_track_used_data=self.cfg.enable_track_used_data,
            delete_cache_length=delete_cache_length,
            path_traj=cfg.path_traj,
        )

    def push_data(self, data: Union[list, dict]) -> None:
        """
        Overview: push data into replay buffer
        Arguments:
            - data (:obj:`list` or `dict`): data list or data item
        Note: thread-safe
        """
        assert (isinstance(data, list) or isinstance(data, dict))

        def push(item: dict) -> None:
            if 'data_push_length' not in item.keys():
                self._meta_buffer.append(item)
                return
            data_push_length = item['data_push_length']
            traj_len = self.traj_len if self.traj_len is not None else data_push_length
            unroll_len = self.unroll_len if self.unroll_len is not None else data_push_length
            assert data_push_length == traj_len
            split_num = traj_len // unroll_len
            split_item = [copy.deepcopy(item) for _ in range(split_num)]
            for i in range(split_num):
                split_item[i]['unroll_split_begin'] = i * unroll_len
                split_item[i]['unroll_len'] = unroll_len
                self._meta_buffer.append(split_item[i])

        if isinstance(data, list):
            for d in data:
                push(d)
        elif isinstance(data, dict):
            push(data)

    def sample(self, batch_size: int, recycle_paths) -> list:
        """
        Overview: sample data from replay buffer
        Arguments:
            - batch_size (:obj:`int`): the batch size of the data will be sampled
        Returns:
            - data (:obj:`list` ): sampled data
        Note: thread-safe
        """
        return self._meta_buffer.sample(batch_size, recycle_paths)

    def update(self, info: dict):
        """
        Overview: update meta buffer with outside info
        Arguments:
            - info (:obj:`dict`): info dict
        Note: thread-safe
        """
        self._meta_buffer.update(info)


    @property
    def count(self):
        """
        Overview: return current buffer data count
        """
        return self._meta_buffer.validlen