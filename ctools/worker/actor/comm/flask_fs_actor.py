import os
import sys
import time
import traceback

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ctools.utils import read_file, save_file
from .base_comm_actor import BaseCommActor


class FlaskFileSystemActor(BaseCommActor):

    def __init__(self, cfg: dict) -> None:
        super(FlaskFileSystemActor, self).__init__(cfg)
        self._url_prefix = 'http://{}:{}/'.format(cfg.upstream_ip, cfg.upstream_port)
        self._requests_session = requests.session()
        retries = Retry(total=20, backoff_factor=1)
        self._requests_session.mount('http://', HTTPAdapter(max_retries=retries))
        self._job_request_id = 0

        self._path_agent = cfg.path_agent
        self._path_traj = cfg.path_traj
        self._heartbeats_freq = cfg.heartbeats_freq

    # override
    def get_job(self) -> dict:
        d = {'request_id': self._job_request_id, 'actor_uid': self._actor_uid}
        while self._active_flag:
            result = self._flask_send(d, 'manager/ask_for_job')
            if result is not None and result['code'] == 0:
                job = result['info']
                break
            else:
                time.sleep(3)
        self._job_request_id += 1
        return job

    def get_init_agent_info(self, model_name):
        path = os.path.join(self._path_agent, model_name)
        return read_file(path)

    # override
    def get_agent_update_info(self, learner_uid) -> dict:
        d = {'learner_uid': learner_uid}
        while True:
            result = self._flask_send(d, 'manager/ask_for_model_path')
            if result is not None and result['code'] == 0:
                path = result['info']
                path = os.path.join(self._path_agent, path)
                return read_file(path), path
            else:
                time.sleep(1)

    # override
    def send_traj_stepdata(self, path: str, stepdata: list) -> None:
        name = os.path.join(self._path_traj, path)
        save_file(name, stepdata)

    # override
    def send_traj_metadata(self, metadata: dict) -> None:
        assert self._actor_uid == metadata['actor_uid']
        d = {'actor_uid': metadata['actor_uid'], 'job_id': metadata['job_id'], 'metadata': metadata}
        api = 'manager/get_metadata'
        self._flask_send(d, api)

    # override
    def send_result(self, result_info: dict) -> None:
        assert self._actor_uid == result_info['actor_uid']
        d = {'actor_uid': result_info['actor_uid'], 'job_id': result_info['job_id'], 'result': result_info}
        api = 'manager/send_result'
        self._flask_send(d, api)

    # override
    def register_actor(self) -> None:
        d = {'actor_uid': self._actor_uid}
        while True:  # only registeration succeeded `_active_flag` can be True
            result = self._flask_send(d, 'manager/register')
            if result is not None and result['code'] == 0:
                return
            else:
                time.sleep(1)

    # override
    def _send_actor_heartbeats(self) -> None:
        while self._active_flag:
            d = {'actor_uid': self._actor_uid}
            self._flask_send(d, 'manager/get_heartbeats')
            for _ in range(self._heartbeats_freq):
                if not self._active_flag:
                    break
                time.sleep(1)

    def _flask_send(self, data, api):
        response = None
        t = time.time()
        try:
            response = self._requests_session.post(self._url_prefix + api, json=data).json()
            name = self._actor_uid
            if 'job_id' in data.keys():
                name += '_{}'.format(data['job_id'])
            if response['code'] == 0:
                self._logger.info("{} succeed sending result: {}, cost_time: {}".format(api, name, time.time() - t))
            else:
                self._logger.error("{} failed to send result: {}, cost_time: {}".format(api, name, time.time() - t))
        except Exception as e:
            self._logger.error(''.join(traceback.format_tb(e.__traceback__)))
            self._logger.error("[error] api({}): {}".format(api, sys.exc_info()))
        return response
