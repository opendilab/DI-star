from typing import Sequence, Mapping
import torch

def detach_grad(data):
    if isinstance(data, Sequence):
        for i in range(len(data)):
            data[i] = detach_grad(data[i])
    elif isinstance(data, Mapping):
        for k in data.keys():
            data[k] = detach_grad(data[k])
    elif isinstance(data, torch.Tensor):
        data = data.detach()
    return data