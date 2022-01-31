import logging
import os
import pickle
import time
from typing import NoReturn, Union

try:
    import pyarrow
    USE_PYARROW = True
except ImportError:
    USE_PYARROW = False

import torch
import io
import pickle

from .import_helper import try_import_ceph
from .import_helper import try_import_mc
from .import_helper import try_import_redis
from .data_helper import to_tensor,to_ndarray
import lz4.frame
import _pickle as cPickle
import warnings
warnings.filterwarnings("ignore")


global mclient
mclient = None

ceph = try_import_ceph()
mc = try_import_mc()
redis,StrictRedis = try_import_redis()

def read_from_ceph(path: str) -> object:
    """
    Overview:
        read file from ceph
    Arguments:
        - path (:obj:`str`): file path in ceph, start with "s3://"
    Returns:
        - (:obj`data`): deserialized data
    """
    value = ceph.Get(path)
    if not value:
        raise FileNotFoundError("File({}) doesn't exist in ceph".format(path))

    return pickle.loads(value)


def read_from_file(path: str) -> object:
    """
    Overview:
        read file from local file system
    Arguments:
        - path (:obj:`str`): file path in local file system
    Returns:
        - (:obj`data`): deserialized data
    """
    with open(path, "rb") as f:
        value = pickle.load(f)

    return value


def _ensure_memcached():
    global mclient
    if mclient is None:
        server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf"
        client_config_file = "/mnt/lustre/share/memcached_client/client.conf"
        mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
    return


def read_from_mc(path: str, flush=False) -> object:
    """
    Overview:
        read file from memcache, file must be saved by `torch.save()`
    Arguments:
        - path (:obj:`str`): file path in local system
    Returns:
        - (:obj`data`): deserialized data
    """
    global mclient
    _ensure_memcached()
    value = mc.pyvector()
    if flush:
        mclient.Get(path, value, mc.MC_READ_THROUGH)
        return
    else:
        mclient.Get(path, value)
    value_buf = mc.ConvertBuffer(value)
    value_str = io.BytesIO(value_buf)
    value_str = torch.load(value_str, map_location='cpu')

    return value_str



def read_from_path(path: str):
    """
    Overview:
        read file from ceph
    Arguments:
        - path (:obj:`str`): file path in ceph, start with "s3://", or use local file system
    Returns:
        - (:obj`data`): deserialized data
    """
    if ceph is None:
        logging.info(
            "You do not have ceph installed! Loading local file!"
            " If you are not testing locally, something is wrong!"
        )
        return read_from_file(path)
    else:
        return read_from_ceph(path)


def save_file_ceph(path, data):
    """
    Overview:
        save pickle dumped data file to ceph
    Arguments:
        - path (:obj:`str`): file path in ceph, start with "s3://", use file system when not
        - data (:obj:`anything`): could be dict, list or tensor etc.
    """
    data = pickle.dumps(data)
    save_path = os.path.dirname(path)
    file_name = os.path.basename(path)
    if ceph is not None:
        if hasattr(ceph, 'save_from_string'):
            ceph.save_from_string(save_path, file_name, data)
        elif hasattr(ceph, 'put'):
            ceph.put(os.path.join(save_path, file_name), data)
        else:
            raise RuntimeError('ceph can not save file, check your ceph installation')
    else:
        import logging
        size = len(data)
        if save_path == 'do_not_save':
            logging.info(
                "You do not have ceph installed! ignored file {} of size {}!".format(file_name, size) +
                " If you are not testing locally, something is wrong!"
            )
            return
        p = os.path.join(save_path, file_name)
        with open(p, 'wb') as f:
            logging.info(
                "You do not have ceph installed! Saving as local file at {} of size {}!".format(p, size) +
                " If you are not testing locally, something is wrong!"
            )
            f.write(data)


def read_file(path: str, fs_type: Union[None, str] = None) -> object:
    r"""
    Overview:
        read file from path
    Arguments:
        - path (:obj:`str`): the path of file to read
        - fs_type (:obj:`str` or :obj:`None`): the file system type, support 'normal' and 'ceph'
    """
    if fs_type is None:
        if path.lower().startswith('s3'):
            fs_type = 'ceph'
        elif mc is not None:
            fs_type = 'mc'
        else:
            fs_type = 'normal'
    assert fs_type in ['normal', 'ceph', 'mc']
    if fs_type == 'ceph':
        data = read_from_path(path)
    elif fs_type == 'normal':
        data = torch.load(path, map_location='cpu')
    elif fs_type == 'mc':
        data = read_from_mc(path)
    return data


def save_file(path: str, data: object, fs_type: Union[None, str] = None) -> NoReturn:
    r"""
    Overview:
        save data to file of path
    Arguments:
        - path (:obj:`str`): the path of file to save to
        - data (:obj:`object`): the data to save
        - fs_type (:obj:`str` or :obj:`None`): the file system type, support 'normal' and 'ceph'
    """
    if fs_type is None:
        if path.lower().startswith('s3'):
            fs_type = 'ceph'
        elif mc is not None:
            fs_type = 'mc'
        else:
            fs_type = 'normal'
    assert fs_type in ['normal', 'ceph', 'mc']
    if fs_type == 'ceph':
        save_file_ceph(path, data)
    elif fs_type == 'normal':
        torch.save(data, path)
    elif fs_type == 'mc':
        torch.save(data, path)
        read_from_mc(path, flush=True)



def remove_file(path: str, fs_type: Union[None, str] = None) -> NoReturn:
    r"""
    Overview:
        remove file
    Arguments:
        - path (:obj:`str`): the path of file you want to remove
        - fs_type (:obj:`str` or :obj:`None`): the file system type, support 'normal' and 'ceph'
    """
    if fs_type is None:
        fs_type = 'ceph' if path.lower().startswith('s3') else 'normal'
    assert fs_type in ['normal', 'ceph']
    if fs_type == 'ceph':
        pass
        os.popen("aws s3 rm --recursive {}".format(path))
    elif fs_type == 'normal':
        os.popen("rm -rf {}".format(path))


def save_traj_file(data: object, path: str,fs_type: Union[None, str] ='pickle') -> NoReturn:
    if fs_type == 'torch':
        torch.save(data,path)
    elif fs_type == 'torchnp':
        data = to_ndarray(data)
        torch.save(data,path)
    elif fs_type == 'pyarrow' and USE_PYARROW:
        data = to_ndarray(data)
        with open(path,'wb',buffering=0) as f:
            f.write(pyarrow.serialize(data).to_buffer())
    else:
        data = to_ndarray(data)
        with open(path,'wb',buffering=0) as f:
            pickle.dump(data,f)

def load_traj_file(path: str, fs_type: Union[None, str] = 'pickle') -> object:
    if fs_type == 'torch':
        data = torch.load(path, map_location='cpu')
    elif fs_type == 'torchnp':
        data = torch.load(path, map_location='cpu')
        data = to_tensor(data)
    elif fs_type == 'pyarrow' and USE_PYARROW:
        with open(path, "rb") as f:
            data = pyarrow.deserialize(f.read())
        data = to_tensor(data)
    else:
        with open(path, "rb") as f:
            data = pickle.load(f)
        data = to_tensor(data)
    return data

def dumps(data,fs_type: Union[None, str] = 'cPickle',compress=True):
    # return cPickle.dumps(data)
    if fs_type == 'pyarrow' and USE_PYARROW:
        data = pyarrow.serialize(to_ndarray(data)).to_buffer()
    elif fs_type == 'torch':
        b = io.BytesIO()
        torch.save(data, b)
        data = b.getvalue()
    elif fs_type == 'pyarrow' and not USE_PYARROW:
        data = cPickle.dumps(data)
    elif fs_type == 'cPickle':
        data = cPickle.dumps(data)
    elif fs_type =='npcPickle':
        data = cPickle.dumps(to_ndarray(data))
    elif fs_type == 'pickle':
        data = pickle.dumps(data)
    elif fs_type =='nppickle':
        data = pickle.dumps(to_ndarray(data))
    else:
        print( f'not support fs_type:{fs_type}')
        raise NotImplementedError
    if compress:
        data = lz4.frame.compress(data)
    return data

def loads(data,fs_type: Union[None, str] = 'cPickle',compress=True):
    if compress:
        data = lz4.frame.decompress(data)
    if fs_type == 'pyarrow' and USE_PYARROW:
        data = to_tensor(pyarrow.deserialize(data))
    elif fs_type == 'torch':
        data = io.BytesIO(data)
        data = torch.load(data, 'cpu')
    elif fs_type == 'pyarrow' and not USE_PYARROW:
        data = cPickle.loads(data)
    elif fs_type == 'cPickle':
        data = cPickle.loads(data)
    elif fs_type == 'npcPickle' :
        data =to_tensor( cPickle.loads(data))
    elif fs_type == 'pickle':
        data = pickle.loads(data)
    elif fs_type =='nppickle':
        data =to_tensor( pickle.loads(data))

    else:
        print( f'not support fs_type:{fs_type}')
        raise NotImplementedError
    return data

def get_redis_address(exp_name,player_id):
    workdir = os.getcwd()
    redis_path = os.path.join(workdir,f'experiments/{exp_name}/{player_id}/redis_address')
    files = os.listdir(redis_path)
    redis_address = []
    for f in files:
        redis_addr = f.split(':')
        redis_address.append({'host':redis_addr[0],"port":redis_addr[1]})
    return redis_address



