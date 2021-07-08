from multiprocessing import connection, get_context, Array, Value
from torch.multiprocessing import Process, Pipe
from collections import namedtuple
import enum
import platform
import time
import math
import copy
import traceback
import threading
import numpy as np
import torch
import ctypes
import pickle
import cloudpickle
from functools import partial
from types import MethodType
from typing import Any, Union, List, Tuple, Iterable, Dict, Callable, Optional
from numpy import dtype
from ctools.utils import PropagatingThread
from ctools.worker.actor.env_manager.base_env_manager import BaseEnvManager
from distar.envs.other.alphastar_map import MAPS
torch.multiprocessing.set_start_method('spawn', force=True)

_NTYPE_TO_CTYPE = {
    np.bool: ctypes.c_bool,
    np.bool_: ctypes.c_bool,
    np.uint8: ctypes.c_uint8,
    np.uint16: ctypes.c_uint16,
    np.uint32: ctypes.c_uint32,
    np.uint64: ctypes.c_uint64,
    np.int8: ctypes.c_int8,
    np.int16: ctypes.c_int16,
    np.int32: ctypes.c_int32,
    np.int64: ctypes.c_int64,
    np.float32: ctypes.c_float,
    np.float64: ctypes.c_double,
}


class EnvState(enum.IntEnum):
    INIT = 1
    RUN = 2
    RESET = 3
    DONE = 4


class ShmBuffer():

    def __init__(self, dtype: np.generic, shape: Tuple[int], key) -> None:
        self.buffer = Array(_NTYPE_TO_CTYPE[dtype.type], int(np.prod(shape)))
        self.dtype = dtype
        self.shape = shape
        self.key = key
        self.entity_num = Value(ctypes.c_uint64, 0)

    def fill(self, src_arr: Union[np.ndarray]) -> None:
        assert isinstance(src_arr, np.ndarray), type(src_arr)
        dst_arr = np.frombuffer(self.buffer.get_obj(), dtype=self.dtype).reshape(self.shape)
        if self.key in ['entity_info', 'id', 'location', 'type']:
            self.entity_num.value = src_arr.shape[0]
            dst_arr = dst_arr[:self.entity_num.value]
        with self.buffer.get_lock():
            np.copyto(dst_arr, src_arr)

    def get(self) -> np.ndarray:
        """
        return a copy of the data
        """
        arr = np.frombuffer(self.buffer.get_obj(), dtype=self.dtype).reshape(self.shape)
        if self.key in ['entity_info', 'id', 'location', 'type']:
            arr = arr[:self.entity_num.value]
        return arr.copy()


class ShmBufferContainer(object):

    def __init__(self, dtype: np.generic, shape: Union[dict, tuple]) -> None:
        if isinstance(shape, dict):
            self._data = {k: ShmBufferContainer(dtype, v) for k, v in shape.items()}
        elif isinstance(shape, (tuple, list)):
            self._data = ShmBuffer(dtype, shape)
        else:
            raise RuntimeError("not support shape: {}".format(shape))
        self._shape = shape

    def fill(self, src_arr: Union[dict, np.ndarray]) -> None:
        if isinstance(self._shape, dict):
            for k in self._shape.keys():
                self._data[k].fill(src_arr[k])
        elif isinstance(self._shape, (tuple, list)):
            self._data.fill(src_arr)

    def get(self) -> Union[dict, np.ndarray]:
        if isinstance(self._shape, dict):
            return {k: self._data[k].get() for k in self._shape.keys()}
        elif isinstance(self._shape, (tuple, list)):
            return self._data.get()


single_shape = {'entity_info': (dtype('float32'), [512, 1340]),
              'entity_raw': {'id': (dtype('int64'), [512]),
                             'location': (dtype('int64'), [512, 2]),
                             'type': (dtype('int64'), [512])},
              'map_size': (dtype('int64'), [2]),
              'scalar_info': {'agent_statistics': (dtype('float32'), [10]),
                              'available_actions': (dtype('float32'), [327]),
                              'beginning_build_order': (dtype('float32'), [20, 194]),
                              'cumulative_stat': {
                                 'effect': (dtype('float32'), [83]),
                                 'research': (dtype('float32'), [60]),
                                 'unit_build': (dtype('float32'), [120])},
                              'immediate_beginning_build_order': (dtype('float32'), [20, 194]),
                              'immediate_cumulative_stat': {
                                 'effect': (dtype('float32'), [83]),
                                 'research': (dtype('float32'), [60]),
                                 'unit_build': (dtype('float32'), [120])},
                              'enemy_race': (dtype('float32'), [5]),
                              'enemy_upgrades': (dtype('float32'), [48]),
                              'last_action_type': (dtype('float32'), [327]),
                              'last_delay': (dtype('float32'), [128]),
                              'last_queued': (dtype('float32'), [20]),
                              'mmr': (dtype('float32'), [7]),
                              'race': (dtype('float32'), [5]),
                              'score_cumulative': (dtype('float32'), [13]),
                              'time': (dtype('float32'), [64]),
                              'unit_counts_bow': (dtype('float32'), [259]),
                              'upgrades': (dtype('float32'), [90])
                             },
              'spatial_info': (dtype('float32'), [20, 148, 152])}
star_shape = [single_shape, copy.deepcopy(single_shape)]


def init_shmbuffer(data, key=None):
    if isinstance(data, list):
        for i in range(len(data)):
            data[i] = init_shmbuffer(data[i])
    if isinstance(data, dict):
        for k, v in data.items():
            data[k] = init_shmbuffer(data[k], k)
    if isinstance(data, tuple):
        return ShmBuffer(data[0], tuple(data[1]), key)
    return data


def transfer(buffer, data, type='none', key=None):
    if key in ['id', 'type', 'map_size']:
        if type == 'fill':
            data = np.array(data, dtype=np.int64)
            buffer.fill(data)
            return
        elif type == 'get':
            data = buffer.get()
            data = data.tolist()
            return data
    if isinstance(buffer, list):
        for i in range(len(buffer)):
            if type == 'fill':
                transfer(buffer[i], data[i], type=type)
            elif type == 'get':
                data[i] = transfer(buffer[i], data[i], type=type)
    if isinstance(buffer, dict):
        for k, v in buffer.items():
            if type == 'fill':
                transfer(buffer[k], data[k], type=type, key=k)
            elif type == 'get':
                data[k] = transfer(buffer[k], data[k], type=type, key=k)
    if isinstance(buffer, ShmBuffer):
        if type == 'fill':
            data = data.numpy()
            buffer.fill(data)
        elif type == 'get':
            data = buffer.get()
            data = torch.from_numpy(data)
            return data
    if type == 'get':
        return data


class StarContainer(object):
    def __init__(self, player_num, map_name):
        shape = copy.deepcopy(star_shape[:player_num])
        map_shape = MAPS[map_name][2]
        for s in shape:
            s['spatial_info'] = (dtype('float32'), [20, map_shape[1], map_shape[0]])
        self._shape = copy.deepcopy(shape)
        self._data = init_shmbuffer(shape)

    def fill(self, data):
        transfer(self._data, data, type='fill')

    def get(self):
        data = copy.deepcopy(self._shape)
        return transfer(self._data, data, type='get')


class CloudpickleWrapper(object):
    """
    Overview:
        CloudpickleWrapper can be able to pickle more python object(e.g: an object with lambda expression)
    """

    def __init__(self, data: Any) -> None:
        self.data = data

    def __getstate__(self) -> bytes:
        if isinstance(self.data, (tuple, list, np.ndarray)):  # pickle is faster
            return pickle.dumps(self.data)
        else:
            return cloudpickle.dumps(self.data)

    def __setstate__(self, data: bytes) -> None:
        if isinstance(data, (tuple, list, np.ndarray)):  # pickle is faster
            self.data = pickle.loads(data)
        else:
            self.data = cloudpickle.loads(data)


def retry_wrapper(fn: Callable, max_retry: int = 5) -> Callable:

    def wrapper(*args, **kwargs):
        exceptions = []
        for _ in range(max_retry):
            try:
                ret = fn(*args, **kwargs)
                return ret
            except Exception as e:
                exceptions.append(e)
                time.sleep(5)
        e_info = ''.join(
            [
                'Retry {} failed from:\n {}\n'.format(i, ''.join(traceback.format_tb(e.__traceback__)) + str(e))
                for i, e in enumerate(exceptions)
            ]
        )
        fn_exception = Exception("Function {} runtime error:\n{}".format(fn, e_info))
        raise RuntimeError("Function {} has exceeded max retries({})".format(fn, max_retry)) from fn_exception

    return wrapper


class SubprocessEnvManager(BaseEnvManager):

    def __init__(
            self,
            env_fn: Callable,
            env_cfg: Iterable,
            env_num: int,
            episode_num: Optional[int] = 'inf',
            timeout: Optional[float] = 0.5,
            wait_num: Optional[int] = 2,
            shared_memory: Optional[bool] = True,
            done_after_episodes=False,
            player_num=2,
            map_name='KingsCove'
    ) -> None:
        super().__init__(env_fn, env_cfg, env_num, episode_num)
        self.shared_memory = shared_memory
        self.timeout = timeout
        self.wait_num = wait_num
        self.done_after_episodes = done_after_episodes
        self.player_num = player_num
        self.map_name = map_name

    def _create_state(self) -> None:
        r"""
        Overview:
            Fork/spawn sub-processes and create pipes to convey the data.
        """
        self._closed = False
        self._env_episode_count = {env_id: 0 for env_id in range(self.env_num)}
        self._env_done = {env_id: False for env_id in range(self.env_num)}
        self._next_obs = {env_id: None for env_id in range(self.env_num)}
        self._env_ref = self._env_fn(self._env_cfg[0])
        if self.shared_memory:
            self._obs_buffers = {env_id: StarContainer(self.player_num, self.map_name) for env_id in range(self.env_num)}
        else:
            self._obs_buffers = {env_id: None for env_id in range(self.env_num)}
        self._parent_remote, self._child_remote = zip(*[Pipe() for _ in range(self.env_num)])
        #context_str = 'spawn' if platform.system().lower() == 'windows' else 'fork'
        context_str = 'spawn'
        ctx = get_context(context_str)
        # due to the runtime delay of lambda expression, we use partial for the generation of different envs,
        # otherwise, it will only use the last item cfg.
        env_fn = [partial(self._env_fn, cfg=self._env_cfg[env_id]) for env_id in range(self.env_num)]
        self._processes = [
            ctx.Process(
                target=self.worker_fn,
                args=(parent, child, CloudpickleWrapper(fn), obs_buffer, self.method_name_list),
                daemon=True
            ) for parent, child, fn, obs_buffer in
            zip(self._parent_remote, self._child_remote, env_fn, self._obs_buffers.values())
        ]
        for p in self._processes:
            p.start()
        for c in self._child_remote:
            c.close()
        self._env_state = {env_id: EnvState.INIT for env_id in range(self.env_num)}
        self._waiting_env = {'step': set()}
        self._setup_async_args()

    def _setup_async_args(self) -> None:
        r"""
        Overview:
            set up the async arguments utilized in the step().
            wait_num: for each time the minimum number of env return to gather
            timeout: for each time the minimum number of env return to gather
        """
        self._async_args = {
            'step': {
                'wait_num': self.wait_num,
                'timeout': self.timeout
            },
        }

    @property
    def active_env(self) -> List[int]:
        return [i for i, s in self._env_state.items() if s == EnvState.RUN]

    @property
    def ready_env(self) -> List[int]:
        return [i for i in self.active_env if i not in self._waiting_env['step']]

    @property
    def next_obs(self) -> Dict[int, Any]:
        no_done_env_idx = [i for i, s in self._env_state.items() if s != EnvState.DONE]
        sleep_count = 0
        while all([self._env_state[i] == EnvState.RESET for i in no_done_env_idx]):
            print('VEC_ENV_MANAGER: all the not done envs are resetting, sleep {} times'.format(sleep_count))
            if sleep_count >= 50:
                raise RuntimeError('Env crashed in reset!!!!')
            time.sleep(sleep_count)
            sleep_count += 1
            
        ret = {i: self._next_obs[i] for i in self.ready_env}
        for i in self.ready_env:
            self._next_obs[i] = None
        return ret

    @property
    def done(self) -> bool:
        return all([self._env_episode_count[env_id] >= self._episode_num for env_id in range(self.env_num)])

    def launch(self, reset_param: Union[None, List[dict]] = None) -> None:
        assert self._closed, "please first close the env manager"
        self._create_state()
        self.reset(reset_param)

    def episode_reset(self, agent_names):
        # reset episode count and env but don't make new environments
        self._env_episode_count = {env_id: 0 for env_id in range(self.env_num)}
        self._waiting_env = {'step': set()}
        self._next_obs = {env_id: None for env_id in range(self.env_num)}
        sleep_count = 0
        resetting_env = [self._env_state[i] == EnvState.RESET for i in range(self.env_num)]
        while any(resetting_env):
            print('VEC_ENV_MANAGER: env:{} are resetting, sleep {} times'.format(resetting_env, sleep_count))
            time.sleep(sleep_count)
            sleep_count += 1
            resetting_env = [self._env_state[i] == EnvState.RESET for i in range(self.env_num)]

        self.reset({'agent_names': agent_names})

    def reset(self, reset_param: Union[None, List[dict], dict] = None) -> None:
        if reset_param is None:
            reset_param = [{} for _ in range(self.env_num)]
        else:
            reset_param = [reset_param for _ in range(self.env_num)]
        self._reset_param = reset_param
        # set seed
        # if hasattr(self, '_env_seed'):
        #     for i in range(self.env_num):
        #         self._parent_remote[i].send(CloudpickleWrapper(['seed', [self._env_seed[i]], {}]))
        #     ret = [p.recv().data for p in self._parent_remote]
        #     self._check_data(ret)

        # reset env
        lock = threading.Lock()
        reset_thread_list = []
        for env_id in range(self.env_num):
            reset_thread = PropagatingThread(target=self._reset, args=(env_id, lock))
            reset_thread.daemon = True
            reset_thread_list.append(reset_thread)
        for t in reset_thread_list:
            t.start()
            # time.sleep(5)  # start all SC2 Processes simultaneously cause crashing
        for t in reset_thread_list:
            t.join()


    def _reset(self, env_id: int, lock: Any) -> None:

        def reset_fn():
            self._parent_remote[env_id].send(CloudpickleWrapper(['reset', [], self._reset_param[env_id]]))
            obs = self._parent_remote[env_id].recv().data
            self._check_data([obs], close=False)
            if self.shared_memory:
                obs = self._obs_buffers[env_id].get()
            with lock:
                self._env_state[env_id] = EnvState.RUN
                self._next_obs[env_id] = obs

        try:
            reset_fn()
        except Exception as e:
            with lock:
                self._env_episode_count[env_id] = self._episode_num # make this env DONE
                self._env_state[env_id] = EnvState.DONE

    def step(self, action: Dict[int, Any]) -> Dict[int, namedtuple]:
        self._check_closed()
        env_ids = list(action.keys())
        assert all([self._env_state[env_id] == EnvState.RUN for env_id in env_ids]
                   ), 'current env state are: {}, please check whether the requested env is in reset or done'.format(
                       {env_id: self._env_state[env_id]
                        for env_id in env_ids}
                   )

        for env_id, act in action.items():
            act = act
            self._parent_remote[env_id].send(CloudpickleWrapper(['step', [act], {}]))

        handle = self._async_args['step']
        wait_num, timeout = min(handle['wait_num'], len(env_ids)), handle['timeout']
        rest_env_ids = list(set(env_ids).union(self._waiting_env['step']))

        ready_env_ids = []
        ret = {}
        cur_rest_env_ids = copy.deepcopy(rest_env_ids)
        while True:
            rest_conn = [self._parent_remote[env_id] for env_id in cur_rest_env_ids]
            ready_conn, ready_ids = SubprocessEnvManager.wait(rest_conn, min(wait_num, len(rest_conn)), timeout)
            cur_ready_env_ids = [cur_rest_env_ids[env_id] for env_id in ready_ids]
            assert len(cur_ready_env_ids) == len(ready_conn)
            ret.update({env_id: p.recv().data for env_id, p in zip(cur_ready_env_ids, ready_conn)})
            self._check_data(ret.values())
            ready_env_ids += cur_ready_env_ids
            cur_rest_env_ids = list(set(cur_rest_env_ids).difference(set(cur_ready_env_ids)))
            # at least one not done timestep or all the connection is ready
            if any([not t.done for t in ret.values()]) or len(ready_conn) == len(rest_conn):
                break

        self._waiting_env['step']: set
        for env_id in rest_env_ids:
            if env_id in ready_env_ids:
                if env_id in self._waiting_env['step']:
                    self._waiting_env['step'].remove(env_id)
            else:
                self._waiting_env['step'].add(env_id)

        lock = threading.Lock()
        for env_id, timestep in ret.items():
            if self.shared_memory:
                timestep = timestep._replace(obs=self._obs_buffers[env_id].get())
            ret[env_id] = timestep
            if timestep.done:
                self._env_episode_count[env_id] += 1
                if self._env_episode_count[env_id] >= self._episode_num and self.done_after_episodes:
                    self._env_state[env_id] = EnvState.DONE
                else:
                    self._env_state[env_id] = EnvState.RESET
                    reset_thread = PropagatingThread(target=self._reset, args=(env_id, lock))
                    reset_thread.daemon = True
                    reset_thread.start()
            else:
                self._next_obs[env_id] = timestep.obs

        return ret

    # this method must be staticmethod, otherwise there will be some resource conflicts(e.g. port or file)
    # env must be created in worker, which is a trick of avoiding env pickle errors.
    @staticmethod
    def worker_fn(p, c, env_fn_wrapper, obs_buffer, method_name_list) -> None:
        env_fn = env_fn_wrapper.data
        torch.set_num_threads(1)
        env = env_fn()
        p.close()
        try:
            while True:
                try:
                    cmd, args, kwargs = c.recv().data
                except EOFError:  # for the case when the pipe has been closed
                    c.close()
                    break
                try:
                    if cmd == 'getattr':
                        ret = getattr(env, args[0])
                    elif cmd in method_name_list:
                        if cmd == 'step':
                            timestep = env.step(*args, **kwargs)
                            if obs_buffer is not None:
                                obs_buffer.fill(timestep.obs)
                                timestep = timestep._replace(obs=None)
                            ret = timestep
                        elif cmd == 'reset':
                            ret = env.reset(*args, **kwargs)  # obs
                            if obs_buffer is not None:
                                obs_buffer.fill(ret)
                                ret = None
                        elif args is None and kwargs is None:
                            ret = getattr(env, cmd)()
                        else:
                            ret = getattr(env, cmd)(*args, **kwargs)
                    else:
                        raise KeyError("not support env cmd: {}".format(cmd))
                    c.send(CloudpickleWrapper(ret))
                except Exception as e:
                    # when there are some errors in env, worker_fn will send the errors to env manager
                    # directly send error to another process will lose the stack trace, so we create a new Exception
                    c.send(
                        CloudpickleWrapper(
                            e.__class__(
                                '\nEnv Process Exception:\n' + ''.join(traceback.format_tb(e.__traceback__)) + repr(e)
                            )
                        )
                    )
                if cmd == 'close':
                    c.close()
                    break
        except KeyboardInterrupt:
            c.close()

    def _check_data(self, data: Iterable, close: bool = True) -> None:
        for d in data:
            if isinstance(d, Exception):
                # when receiving env Exception, env manager will safely close and raise this Exception to caller
                print(d)
                if close:
                    self.close()
                raise d

    # override
    def __getattr__(self, key: str) -> Any:
        self._check_closed()
        # we suppose that all the envs has the same attributes, if you need different envs, please
        # create different env managers.
        if not hasattr(self._env_ref, key):
            raise AttributeError("env `{}` doesn't have the attribute `{}`".format(type(self._env_ref), key))
        if isinstance(getattr(self._env_ref, key), MethodType) and key not in self.method_name_list:
            raise RuntimeError("env getattr doesn't supports method({}), please override method_name_list".format(key))
        for p in self._parent_remote:
            p.send(CloudpickleWrapper(['getattr', [key], {}]))
        ret = [p.recv().data for p in self._parent_remote]
        self._check_data(ret)
        return ret

    # override
    def close(self) -> None:
        if self._closed:
            return
        self._env_ref.close()
        self._closed = True
        for p in self._parent_remote:
            p.send(CloudpickleWrapper(['close', None, None]))
        for p in self._processes:
            p.join()
        for p in self._processes:
            p.terminate()
        for p in self._parent_remote:
            p.close()

    @staticmethod
    def wait(rest_conn: list, wait_num: int, timeout: Union[None, float] = None) -> Tuple[list, list]:
        """
        Overview:
            wait at least enough(len(ready_conn) >= wait_num) num connection within timeout constraint
            if timeout is None, wait_num == len(ready_conn), means sync mode;
            if timeout is not None, len(ready_conn) >= wait_num when returns;
        """
        assert 1 <= wait_num <= len(rest_conn
                                    ), 'please indicate proper wait_num: <wait_num: {}, rest_conn_num: {}>'.format(
                                        wait_num, len(rest_conn)
                                    )
        rest_conn_set = set(rest_conn)
        ready_conn = set()
        start_time = time.time()
        rest_time = timeout
        while len(rest_conn_set) > 0:
            finish_conn = set(connection.wait(rest_conn_set, timeout=timeout))
            ready_conn = ready_conn.union(finish_conn)
            rest_conn_set = rest_conn_set.difference(finish_conn)
            if len(ready_conn) >= wait_num and timeout:
                rest_time = timeout - (time.time() - start_time)
                if rest_time <= 0.0:
                    break
        ready_ids = [rest_conn.index(c) for c in ready_conn]
        return list(ready_conn), ready_ids


class SyncSubprocessEnvManager(SubprocessEnvManager):

    def _setup_async_args(self) -> None:
        self._async_args = {
            'step': {
                'wait_num': math.inf,
                'timeout': None,
            },
        }


if __name__ == '__main__':
    def func(d1, d2, key=None):
        if key in ['id', 'type', 'map_size']:
            print(key, d1 == d2)
            return
        if isinstance(d1, dict):
            for k in d1.keys():
                func(d1[k], d2[k], k)
        if isinstance(d1, torch.Tensor):
            print(key, d1.equal(d2))

    data = torch.load('C:/work\data/d.d')
    d = copy.deepcopy(data)
    buffer = StarContainer()
    buffer.fill(data)
    data = buffer.get()
    for i in range(2):
        func(d[i], data[i])
