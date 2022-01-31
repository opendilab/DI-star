from typing import Iterable, Any, Optional
from collections.abc import Sequence
import numbers
import time
from threading import Thread
from queue import Queue

import numpy as np
import torch


def to_tensor(
        item: Any,
        dtype: Optional[torch.dtype] = None,
        ignore_keys: list = [],
        transform_scalar: bool = True
) -> torch.Tensor:
    r"""
    Overview:
        Change `numpy.ndarray`, sequence of scalars to torch.Tensor, and keep other data types unchanged.
    Arguments:
        - item (:obj:`Any`): the item to be changed
        - dtype (:obj:`type`): the type of wanted tensor
    Returns:
        - item (:obj:`torch.Tensor`): the change tensor
    .. note:
        Now supports item type: :obj:`dict`, :obj:`list`, :obj:`tuple` and :obj:`None`
    """

    def transform(d):
        if dtype is None:
            return torch.as_tensor(d)
        else:
            return torch.tensor(d, dtype=dtype)

    if isinstance(item, dict):
        new_data = {}
        for k, v in item.items():
            if k in ignore_keys:
                new_data[k] = v
            else:
                new_data[k] = to_tensor(v, dtype, ignore_keys, transform_scalar)
        return new_data
    elif isinstance(item, list) or isinstance(item, tuple):
        if len(item) == 0:
            return None
        elif isinstance(item[0], numbers.Integral) or isinstance(item[0], numbers.Real):
            return transform(item)
        elif hasattr(item, '_fields'):  # namedtuple
            return type(item)(*[to_tensor(t, dtype) for t in item])
        else:
            new_data = []
            for t in item:
                new_data.append(to_tensor(t, dtype, ignore_keys, transform_scalar))
            return new_data
    elif isinstance(item, np.ndarray):
        if dtype is None:
            if item.dtype == np.float64:
                return torch.FloatTensor(item)
            else:
                return torch.from_numpy(item)
        else:
            return torch.from_numpy(item).to(dtype)
    elif isinstance(item, bool) or isinstance(item, str):
        return item
    elif np.isscalar(item):
        if transform_scalar:
            if dtype is None:
                return torch.as_tensor(item)
            else:
                return torch.as_tensor(item).to(dtype)
        else:
            return item
    elif item is None:
        return None
    elif isinstance(item, torch.Tensor):
        if dtype is None:
            return item
        else:
            return item.to(dtype)
    else:
        raise TypeError("not support item type: {}".format(type(item)))


def to_ndarray(item: Any, dtype: np.dtype = None) -> np.ndarray:
    r"""
    Overview:
        Change `torch.Tensor`, sequence of scalars to ndarray, and keep other data types unchanged.
    Arguments:
        - item (:obj:`object`): the item to be changed
        - dtype (:obj:`type`): the type of wanted ndarray
    Returns:
        - item (:obj:`object`): the changed ndarray
    .. note:
        Now supports item type: :obj:`torch.Tensor`,  :obj:`dict`, :obj:`list`, :obj:`tuple` and :obj:`None`
    """

    def transform(d):
        if dtype is None:
            return np.array(d)
        else:
            return np.array(d, dtype=dtype)

    if isinstance(item, dict):
        new_data = {}
        for k, v in item.items():
            new_data[k] = to_ndarray(v, dtype)
        return new_data
    elif isinstance(item, list) or isinstance(item, tuple):
        if len(item) == 0:
            return None
        elif isinstance(item[0], numbers.Integral) or isinstance(item[0], numbers.Real):
            return transform(item)
        elif hasattr(item, '_fields'):  # namedtuple
            return type(item)(*[to_ndarray(t, dtype) for t in item])
        else:
            new_data = []
            for t in item:
                new_data.append(to_ndarray(t, dtype))
            return new_data
    elif isinstance(item, torch.Tensor):
        if dtype is None:
            return item.cpu().numpy()
        else:
            return item.cpu().numpy().astype(dtype)
    elif isinstance(item, np.ndarray):
        if dtype is None:
            return item
        else:
            return item.astype(dtype)
    elif isinstance(item, bool) or isinstance(item, str):
        return item
    elif np.isscalar(item):
        return np.array(item)
    elif item is None:
        return None
    else:
        raise TypeError("not support item type: {}".format(type(item)))
