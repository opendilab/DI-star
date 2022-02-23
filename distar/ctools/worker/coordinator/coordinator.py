import os
import logging
import multiprocessing

from collections import defaultdict, deque
from functools import partial
import re

from distar.agent.default.rl_training.rl_dataloader import worker_loop
from flask import Flask, request
import portpicker

from distar.ctools.utils.log_helper import TextLogger
from distar.ctools.utils import LockContextType, LockContext

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


class Worker(object):
    def __init__(self, worker_index) -> None:
        self._lock = LockContext(type_=LockContextType.THREAD_LOCK)
        self._buffer = defaultdict(partial(deque, maxlen=256))
        self.worker_index = worker_index

    def deal_with_push(self, request_info):
        token = request_info.pop('token')
        self._buffer[token].append(request_info)
        print(f'worker index: {self.worker_index}, push to token: {token} buffer size: {len(self._buffer[token])}, {request_info}')
        return True
    
    def deal_with_pull(self, request_info):
        token = request_info['token']
        size = request_info['size']
        num = size if size is not None else 1
        data = []
        with self._lock:
            for _ in range(num):
                try:
                    data.append(self._buffer[token].pop())
                except IndexError:
                    break
        if size is None:
            if len(data):
                print(f'worker index: {self.worker_index}, pull from token: {token} buffer size: {len(self._buffer[token])}, {request_info}')
                return data[0]      
            else:
                return False
        else:
            if len(data):
                print(f'worker index: {self.worker_index}, pull from token: {token} buffer size: {len(self._buffer[token])}, {request_info}')
            return data
    

def run_worker(token, ip, port, worker_index):
    worker = Worker(worker_index)
    worker_app = create_worker_app(worker)
    print(f'run worker for token: {token} at {ip}: {port}')
    worker_app.run(host=ip, port=port, debug=False, use_reloader=False)
    

class Coordinator(object):
    def __init__(self, cfg: dict) -> None:
        self._ip = cfg.communication.coordinator_ip
        self._lock = LockContext(type_=LockContextType.THREAD_LOCK)
        self._whole_cfg = cfg
        self._buffer = defaultdict(partial(deque, maxlen=1000))
        self._put_server_list = defaultdict(list)
        self._get_server_list = defaultdict(list)
        self._remove_count = defaultdict(int)
        self._workers = defaultdict(list)
        self._logger = TextLogger(
            path=os.path.join(os.getcwd(), 'experiments', self._whole_cfg.common.experiment_name, 'log'),
            name='coordinator')
        self.traj_metadata_buffer = {}

    def deal_with_register(self, request_info):
        register_type = request_info.pop('type')
        if register_type == 'get':
            return self.register_get(request_info)
        else:
            return self.register_put(request_info)

    def register_put(self, request_info):
        token = request_info.pop('token')
        server = request_info.get('server', False)
        with self._lock:
            if server:
                self._put_server_list[token].append(request_info)
                print(self._put_server_list)
                return True
            else:
                if not len(self._get_server_list[token]):
                    return False
                else:
                    ret = self._get_server_list[token]
                    return ret

    def register_get(self, request_info):
        token = request_info['token']
        server = request_info.get('server', True)
        with self._lock:
            if server:
                self._get_server_list[token].append(request_info)
                print(self._get_server_list)
                return True
            else:
                if not len(self._put_server_list[token]):
                    return False
                else:
                    ret = self._put_server_list[token]
                    return ret
    
    def deal_with_remove_server(self, request_info):
        token, type, ip, port = request_info['token'], request_info['type'], request_info['ip'], request_info['port']
        if type == 'get':
            server_list = self._put_server_list[token]
        else:
            server_list = self._get_server_list[token]
        for s in server_list:
            if s['ip'] == ip and s['port'] == port:
                with self._lock:
                    key = str(ip) + str(port)
                    self._remove_count[key] += 1
                    if self._remove_count[key] > 5:
                        print('remove server: {}'.format(request_info))
                        server_list.remove(s)
                        self._remove_count.pop(key)

    def deal_with_push(self, request_info, request_ip):
        token = request_info.pop('token')
        request_info['user_ip'] = request_ip
        self._buffer[token].append(request_info)
        print(f'push to token: {token}, buffer size: {len(self._buffer[token])}, request info: {request_info}')
        return True

    def deal_with_pull(self, request_info):
        token = request_info['token']
        size = request_info.pop('size')
        num = size if size is not None else 1
        data = []
        with self._lock:
            for _ in range(num):
                try:
                    data.append(self._buffer[token].pop())
                except IndexError:
                    break
        if len(data) == 0:
            return False
        print(f'pull from token: {token}, buffer size: {len(self._buffer[token])}, request info: {request_info}')
        if size is None:
            return data[0]
        else:
            return data

    def deal_with_start_worker(self, request_info):
        token = request_info['token']
        worker_num = request_info['worker_num']
        with self._lock:
            if len(self._workers[token]) == 0:
                for i in range(worker_num):
                    port = portpicker.pick_unused_port()
                    multiprocessing.Process(target=run_worker, args=(token, self._ip, port, i,), daemon=True).start()
                    self._workers[token].append({'ip': self._ip, 'port': port})
        return self._workers[token]


def create_coordinator_app(coordinator: Coordinator):
    app = Flask(__name__)

    def build_ret(code, info=''):
        return {'code': code, 'info': info}

    @app.route('/coordinator/remove_server', methods=['POST'])
    def remove_server():
        ret_info = coordinator.deal_with_remove_server(request.json)
        if ret_info:
            return build_ret(0)
        else:
            return build_ret(1)

    @app.route('/coordinator/register', methods=['POST'])
    def register():
        ret_info = coordinator.deal_with_register(request.json)
        if ret_info:
            return build_ret(0, ret_info)
        else:
            return build_ret(1)

    @app.route('/coordinator/push', methods=['POST'])
    def push():
        ret_info = coordinator.deal_with_push(request.json, request.remote_addr)
        if ret_info:
            return build_ret(0)
        else:
            return build_ret(1)

    @app.route('/coordinator/pull', methods=['POST'])
    def pull():
        ret_info = coordinator.deal_with_pull(request.json)
        if ret_info:
            return build_ret(0, ret_info)
        else:
            return build_ret(1)

    @app.route('/coordinator/start_worker', methods=['POST'])
    def start_worker():
        ret_info = coordinator.deal_with_start_worker(request.json)
        if ret_info:
            return build_ret(0, ret_info)
        else:
            return build_ret(1)

    # ************************** learner *********************************
    @app.route('/coordinator/learner_ask_for_metadata', methods=['POST'])
    def learner_ask_for_metadata():
        ret_info = coordinator.deal_with_learner_ask_for_metadata(request.json)
        if ret_info:
            return build_ret(0, ret_info)
        else:
            return build_ret(1)

    # ************************** actor *********************************
    @app.route('/coordinator/actor_send_metadata', methods=['POST'])
    def actor_send_metadata():
        ret_info = coordinator.deal_with_actor_send_metadata(request.json)
        if ret_info:
            return build_ret(0)
        else:
            return build_ret(1)

    return app

def create_worker_app(worker: Worker):
    app = Flask(__name__)

    def build_ret(code, info=''):
        return {'code': code, 'info': info}

    @app.route('/worker/push', methods=['POST'])
    def push():
        ret_info = worker.deal_with_push(request.json)
        if ret_info:
            return build_ret(0)
        else:
            return build_ret(1)

    @app.route('/worker/pull', methods=['POST'])
    def pull():
        ret_info = worker.deal_with_pull(request.json)
        if ret_info:
            return build_ret(0, ret_info)
        else:
            return build_ret(1)

    return app
