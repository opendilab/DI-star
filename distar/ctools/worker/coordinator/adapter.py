import pickle
import socket
import requests
import threading
import portpicker
import time
import uuid
import io
import torch
import os
import gc
import copy
import random

from collections import deque, defaultdict
from functools import partial

from distar.ctools.utils.file_helper import redis, dumps, loads
from .protocol import encode, decode


BYTES_LENGTH = 5
START = b"\n\r"
END = b"\r\n"


class Adapter(object):
    def __init__(self, cfg=None, maxlen=None, server=None, token=None, type=None):
        self._whole_cfg = cfg
        self._maxlen = maxlen
        self._cache_data = defaultdict(partial(deque, maxlen=maxlen))
        self._worker_addr = defaultdict(list)
        self._coordinator_ip = self._whole_cfg.communication.coordinator_ip
        self._coordinator_port = self._whole_cfg.communication.coordinator_port

        # =======ignore these code=======
        self._server = server
        self._token = token
        self._type = type  # 'put' or 'get' 
        self._last_update_request_addr_time = time.time()
        if server is not None:
            self._register()
        # =======ignore these code=======

    def full(self, token='default'):
        if self._maxlen is not None and len(self._cache_data[token]) == self._maxlen:
            return True
        else:
            return False

    def length(self, token='default'):
        return len(self._cache_data[token])

    def request_worker(self, token, worker_num):
        meta_data = {'token': token, 'worker_num': worker_num}
        while True:
            try: 
                response = requests.post('http://{}:{}/'.format(self._coordinator_ip, self._coordinator_port) + 'coordinator/start_worker', json=meta_data).json()
                assert response['code'] == 0
                break
            except Exception as e:
                print(f'[start worker ERROR], can not connect to coordinator, ip: {self._coordinator_ip}, port: {self._coordinator_port}, please restart coordinator with same address')
                time.sleep(random.randint(30, 50))
        self._worker_addr[token] = response['info']

    def push(self, data, token='default', fs_type='torch', compress=True, worker_num=None):
        if not hasattr(self, '_push_thread'):
            def get_host_ip(): 
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    s.connect((self._coordinator_ip, self._coordinator_port))
                    ip = s.getsockname()[0]
                finally:
                    s.close()
                return ip
            self._ip = get_host_ip()
            self._push_thread = threading.Thread(target=self._push_loop, daemon=True)
            self._push_thread.start()

        if worker_num is not None and len(self._worker_addr[token]) == 0:
            self.request_worker(token, worker_num)

        if not isinstance(data, bytes):
            data = dumps(data, fs_type=fs_type, compress=compress)
        self._cache_data[token].append(data)

    def _push_loop(self):
        torch.set_num_threads(1)
        meta_data = {'user_ip': self._ip}
        while True:
            flag = False
            for token, v in self._cache_data.items():
                if len(v):
                    flag = True 
                    break
            if not flag:
                time.sleep(0.01)
                continue
            while True:
                try:
                    port = portpicker.pick_unused_port()
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.bind((self._ip, port))
                    s.listen(5)
                    break
                except Exception as e:
                    s.close()
            meta_data['token'] = token
            meta_data.update({'token': token, 'user_port': port})
            if len(self._worker_addr[token]) == 0:
                http_addr =  'http://{}:{}/'.format(self._coordinator_ip, self._coordinator_port) + 'coordinator/push'
            else:
                worker_index = random.randint(0, len(self._worker_addr[token]) - 1)
                worker_addr = self._worker_addr[token][worker_index]
                http_addr = 'http://{}:{}/'.format(worker_addr['ip'], worker_addr['port']) + 'worker/push'
            try: 
                response = requests.post(http_addr, json=meta_data).json()
                assert response['code'] == 0
            except requests.exceptions.ConnectionError as e:
                worker_num = len(self._worker_addr[token])
                if worker_num:
                    self.request_worker(token, worker_num)
                print(f'[push ERROR], can not connect to {http_addr}, retrying!')
                time.sleep(1)
                continue
            except Exception as e:
                print(f'[push ERROR], can not connect to {http_addr} retrying!', e)
                time.sleep(10)
                continue

            try:
                s.settimeout(20)
                c, addr = s.accept()
            except socket.timeout as e:
                # print('[push ERROR] wait accept time out', e)
                s.close()
                continue
            try:
                # t = time.time()
                data = self._cache_data[token].popleft()
                data_len = len(data)
                send_len = c.send(data_len.to_bytes(length=BYTES_LENGTH, byteorder='big', signed=False))
                if send_len == 0:
                    raise RuntimeError("socket connection broken")
                c.sendall(data)
                # print(f'size: {data_len/ 1000000:.2f}M, push comm time: {time.time() - t:.2f}', flush=True)
            except Exception as e:
                print('[push socket ERROR jump to next data]', e, flush=True)
            finally:
                c.close()
                s.close()  

    def pull(self, token='default', fs_type='torch', compress=True, sleep_time=1, timeout=None, size=None, worker_num=None):
        if worker_num is not None and len(self._worker_addr[token]) == 0:
            self.request_worker(token, worker_num)
        require_size = size
        meta_data = {'token': token, 'size': size}
        start_time = time.time()
        sleep_count = 0
        if len(self._worker_addr[token]) == 0:
            http_addr = 'http://{}:{}/'.format(self._coordinator_ip, self._coordinator_port) + 'coordinator/pull'
        else:
            worker_index = random.randint(0, len(self._worker_addr[token]) - 1)
            worker_addr = self._worker_addr[token][worker_index]
            http_addr = 'http://{}:{}/'.format(worker_addr['ip'], worker_addr['port']) + 'worker/pull'

        ret_data = []
        while True:
            while True:
                try:
                    t = time.time()
                    response = requests.post(http_addr, json=meta_data).json()
                    # print('coordinator pull time', time.time() - t)
                    if response['code'] == 0:
                        results = response['info']
                        break
                    else:
                        if timeout is not None and time.time() - start_time > timeout:
                            return None
                        if len(self._worker_addr[token]):
                            worker_index = (worker_index + 1) % len(self._worker_addr[token])
                            worker_addr = self._worker_addr[token][worker_index]
                            http_addr = 'http://{}:{}/'.format(worker_addr['ip'], worker_addr['port']) + 'worker/pull'
                        time.sleep(sleep_time + sleep_count * sleep_time)
                        sleep_count = min(sleep_count + 1, 10)
                except requests.exceptions.ConnectionError as e:
                    if worker_num is not None:
                        self.request_worker(token, worker_num)
                        worker_index = random.randint(0, len(self._worker_addr[token]) - 1)
                        worker_addr = self._worker_addr[token][worker_index]
                        http_addr = 'http://{}:{}/'.format(worker_addr['ip'], worker_addr['port']) + 'worker/pull'
                    print(f'[pull ERROR], can not connect to {http_addr}, retrying!')
                    time.sleep(1)
                except Exception as e:
                    print(f'[pull ERROR], can not connect to {http_addr}, retrying!', e)
                    time.sleep(10)
                    
            if size is None:
                results = [results]
            for result in results:
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(5)
                    try:
                        s.connect((result['user_ip'], result['user_port']))
                    except Exception as e:
                        s.close()
                        print('[pull ERROR] socket connect time out, if this keeps happening, check firewall or close proxy!', e)
                        continue
                        
                    data = io.BytesIO()
                    data_len = 0
                    bytes_length = s.recv(BYTES_LENGTH)
                    if len(bytes_length) == 0:
                        raise RuntimeError("socket connection broken")
                    data_size = int.from_bytes(bytes_length, byteorder='big', signed=False)
                    while data_len < data_size:
                        data.seek(data_len)
                        d = s.recv(data_size - data_len)
                        if len(d) == 0:
                            raise RuntimeError("socket connection broken")
                        data.write(d)
                        data_len += len(d)
                    s.close()
                    t = time.time()
                    new_data = loads(data.getvalue(), fs_type, compress=compress)
                    data.close()
                    ret_data.append(new_data)
                    # print(f'size: {data_len/ 1000000:.2f}M, pull unpickle time: {time.time() - t:.2f}', flush=True)
                except Exception as e:
                    data.close()
                    s.close()
                    print('[pull socket ERROR jump to next data]', e, flush=True)
                    if timeout is not None and time.time() - start_time > timeout:
                        return None
                    time.sleep(sleep_time + sleep_count * sleep_time)
                    sleep_count = min(sleep_count + 1, 10)
            if size is None and len(ret_data):
                return ret_data[0]

            if size is not None:
                left_num = require_size - len(ret_data)
                if left_num > 0:
                    meta_data['size'] = left_num
                else:
                    return ret_data
            
    def _register(self):
        self._ip = socket.gethostbyname(socket.gethostname())
        self._port = portpicker.pick_unused_port()
        self.meta_data = {'token': self._token, 'type': self._type, 'server': self._server}
        if self._server:
            self._socket = socket.socket()
            self._socket.bind((self._ip, self._port))
            self.meta_data.update({'ip': self._ip, 'port': self._port})
            while True:
                try:
                    response = requests.post(self._coordinator_url_prefix + 'coordinator/register', json=self.meta_data).json()
                    if response['code'] == 0:
                        print('register adapter successfully, token: {}, type: {}, ip: {}, port: {}'.format(self._token, self._type, self._ip, self._port), flush=True)
                        break
                    else:
                        time.sleep(5)
                except Exception as e:
                    print('[register adapter server ERROR] coordinator error', e, flush=True)
        else:
            self._update_request_addr(show_log=True)
        
    def _update_request_addr(self, show_log=False):
        while True:
            try: 
                response = requests.post(self._coordinator_url_prefix + 'coordinator/register', json=self.meta_data).json()
                if response['code'] == 0:
                    if show_log:
                        print('register adapter successfully, token: {}, type: {}, ip: {}, port: {}'.format(self._token, self._type, self._ip, self._port), flush=True)
                    result = response['info']
                    break
                else:
                    if show_log:
                        print('register adapter failed, token: {} type: {}'.format(self._token, self._type))
                    time.sleep(5)
            except Exception as e:
                print('[update request address ERROR] coordinator error', e, flush=True)
        self._request_ips = []
        self._request_ports = []
        for i in range(len(result)):
            self._request_ips.append(result[i]['ip'])
            self._request_ports.append(result[i]['port'])
            
    def send_data(self, c, data):
        data_len = len(data) + 2
        bytes_len = data_len.to_bytes(length=BYTES_LENGTH, byteorder='big', signed=False)
        c.sendall(bytes_len)
        data = b"".join((data, END))
        c.sendall(data)

    def recv_data(self, c):
        t = time.time()
        data = io.BytesIO()
        data_len = 0
        bytes_length = c.recv(BYTES_LENGTH)
        if len(bytes_length) == 0:
            raise RuntimeError("socket connection broken")
        data_size = int.from_bytes(bytes_length, byteorder='big', signed=False)
        while data_len < data_size:
            data.seek(data_len)
            d = c.recv(data_size - data_len)
            if len(d) == 0:
                raise RuntimeError("socket connection broken")
            data.write(d)
            data_len += len(d)
        c.close()
        ret_data = data.getvalue()
        assert ret_data.endswith(END), 'received data does not end with END!!'
        data.close()
        return ret_data[:-2]

    def get(self, fs_type='torch', compress=True):
        if self._server:
            while True:
                try:
                    self._socket.listen(5)
                    c, addr = self._socket.accept()
                    data = self.recv_data(c)
                    data = loads(data, fs_type=fs_type, compress=compress)
                    return data
                except Exception as e:
                    c.close()
                    time.sleep(1)
                    print('[get ERROR]', e, flush=True)
        else:
            while True:
                try:
                    s = socket.socket()
                    s.settimeout(0.1)
                    if time.time() - self._last_update_request_addr_time > 120:
                        self._update_request_addr()
                        self._last_update_request_addr_time = time.time()
                    idx = random.randint(0, len(self._request_ips) - 1)
                    ip, port = self._request_ips[idx], self._request_ports[idx]
                    s.connect((ip, port))
                    s.settimeout(None)
                    data = self.recv_data(s)
                    data = loads(data, fs_type=fs_type, compress=compress)
                    return data
                except socket.timeout as e:
                    pass
                except ConnectionRefusedError as e:
                    print('get connection refused, remove server list')
                    self._remove_server('get', ip, port)
                    time.sleep(5)
                except Exception as e:
                    print('[get ERROR]', e, flush=True)
                    time.sleep(0.1)
                finally:
                    s.close()

    def put(self, data, fs_type='pyarrow', compress=False):
        if not hasattr(self, '_put_thread'):
            self._put_thread = threading.Thread(target=self._put_loop, daemon=True)
            self._put_thread.start()
        if not isinstance(data, bytes):
            data = dumps(data, fs_type=fs_type, compress=compress)
        self._cache_data[self._token].append(data)

    def _put_loop(self):
        while True:
            if len(self._cache_data[self._token]):
                data = self._cache_data[self._token].popleft()
            else:
                time.sleep(0.1)
                continue
            if self._server:
                while True:
                    try:
                        self._socket.listen(5)
                        c, addr = self._socket.accept()
                        self.send_data(c, data)
                        break
                    except Exception as e:
                        c.close()
                        time.sleep(1)
                        print('[put ERROR]', e, flush=True)
            else:
                while True:
                    try:
                        s = socket.socket()
                        s.settimeout(0.1)
                        if time.time() - self._last_update_request_addr_time > 120:
                            self._update_request_addr()
                            self._last_update_request_addr_time = time.time()
                        idx = random.randint(0, len(self._request_ips) - 1)
                        ip, port = self._request_ips[idx], self._request_ports[idx]
                        s.connect((ip, port))
                        s.settimeout(None)
                        self.send_data(s, data)
                        break
                    except socket.timeout as e:
                        pass
                    except ConnectionRefusedError as e:
                        print('put connection refused, remove server list')
                        self._remove_server('put', ip, port)
                        time.sleep(5)
                    except Exception as e:
                        print('[put ERROR]', e, flush=True)
                        time.sleep(0.1)
                    finally:
                        s.close()
    
    def _remove_server(self, type, ip, port):
        meta_data = {'token': self._token, 'type': type, 'ip': ip, 'port': port}
        requests.post(self._coordinator_url_prefix + 'coordinator/remove_server', json=meta_data).json()
