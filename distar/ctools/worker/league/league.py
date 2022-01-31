import itertools
import json
import os
import random
import shutil
import sys

import threading
import time
from abc import ABC
from collections import defaultdict, Counter, OrderedDict
from copy import deepcopy
from pprint import pprint, pformat
from shutil import copyfile, rmtree

import numpy as np
import torch
import torch.multiprocessing as tm
from easydict import EasyDict
from tensorboardX import SummaryWriter

from distar.ctools.utils import LockContextType, read_config, LockContext
from distar.ctools.utils.log_helper import TextLogger
from distar.ctools.worker.ladder.elo import ELORating
from distar.ctools.worker.league.player import ActivePlayer, HistoricalPlayer, MainPlayer, MainExploiterPlayer, ExploiterPlayer, ExpertPlayer, AdaptiveEvolutionaryExploiterPlayer, ExpertExploiterPlayer
import dill as pickle
import lz4.frame


class League(ABC):
    def __init__(self, cfg: EasyDict) -> None:
        self._whole_cfg = cfg

        self.api_info = defaultdict(list)
        torch.set_num_threads(1)

        self._lock = LockContext(type_=LockContextType.THREAD_LOCK)
        self._set_logger()

        # dir realated
        self._path_model = os.path.abspath(
            os.path.join('experiments', self._whole_cfg.common.experiment_name, 'league_models'))
        if not os.path.exists(self._path_model):
            os.makedirs(self._path_model)
        self.resume_dir = os.path.join('experiments', self._whole_cfg.common.experiment_name, 'league_resume')
        if not os.path.exists(self.resume_dir):
            os.makedirs(self.resume_dir)
        # setup players
        self._init_league()

        # thread to save resume
        self.save_resume_freq = self.cfg.get('save_resume_freq', 3600)
        save_resume_thread = threading.Thread(target=self._save_resume_thread, daemon=True)
        save_resume_thread.start()

        # thread for deal_with_actor_send_result
        self._result_queue = tm.Queue()
        self._send_result_thread = threading.Thread(target=self._send_result_loop, daemon=True)
        self._send_result_thread.start()

    def _set_logger(self):
        # pay off related
        self._stat_decay = self.cfg.get('stat_decay', 0.999)
        self._stat_warm_up_size = self.cfg.get('stat_warm_up_size', 1000)
        self._payoff_min_win_rate_games = self.cfg.get('payoff_min_win_rate_games', 200)

        # logging related
        self._print_freq = self.cfg.get('print_freq', 100)
        self._logger = TextLogger(
            path=os.path.join(os.getcwd(), 'experiments', self._whole_cfg.common.experiment_name, 'log'),
            name='league')

        # tb_logger related
        # we have different tb_log for different player
        self._tb_log_dir = os.path.join(os.getcwd(), 'experiments', self._whole_cfg.common.experiment_name, 'tb_log')
        self._tb_log_dict = {}

        # elo related
        self.elo = ELORating()

    def set_player_tblogger(self, player_id):
        player_tb_log_path = os.path.join(self._tb_log_dir, player_id)
        if not os.path.exists(player_tb_log_path):
            try:
                os.mkdir(player_tb_log_path)
            except:
                pass
        self._tb_log_dict[player_id] = SummaryWriter(player_tb_log_path)

    def _init_league(self) -> None:
        if self.cfg.resume_path and os.path.isfile(
                self.cfg.resume_path):
            self.logger.info('load league, path: {}'.format(self.cfg.resume_path))
            self._load_resume(self.cfg.resume_path)
        else:
            self.active_players = {}
            self.historical_players = {}
            # init active players
            active_ckpt_paths = self.cfg.active_players.checkpoint_path
            active_pipelines = self.cfg.active_players.pipeline
            active_frac_ids = self.cfg.active_players.frac_id
            active_z_paths = self.cfg.active_players.z_path
            active_teacher_ckpt_paths = self.cfg.active_players.teacher_path
            active_teacher_ids = self.cfg.active_players.teacher_id
            active_player_ids = self.cfg.active_players.player_id
            one_phase_step_list = self.cfg.active_players.one_phase_step
            chosen_weight_list = self.cfg.active_players.chosen_weight
            z_probs_list = self.cfg.active_players.z_prob
            for (ckpt_path, pipeline, frac_id, z_path, teacher_id, teacher_ckpt, active_player_id,
                 one_phase_step, chosen_weight, z_prob) in zip(active_ckpt_paths, active_pipelines, active_frac_ids,
                                        active_z_paths, active_teacher_ids, active_teacher_ckpt_paths,
                                        active_player_ids,
                                        one_phase_step_list, chosen_weight_list, z_probs_list):
                self.add_active_player(ckpt_path, pipeline, frac_id, z_path, teacher_id, teacher_ckpt, active_player_id,
                                       one_phase_step,chosen_weight, z_prob)

            # add pretrain player as the initial HistoricalPlayer
            if self.cfg.use_historical_players:
                history_player_ids = self.cfg.historical_players.get('player_id', None)
                history_ckpt_paths = self.cfg.historical_players.checkpoint_path
                history_pipelines = self.cfg.historical_players.pipeline
                history_frac_ids = self.cfg.historical_players.frac_id
                history_z_paths = self.cfg.historical_players.z_path
                history_z_probs = self.cfg.historical_players.z_prob
                for idx, (ckpt_path, pipeline, frac_id, z_path, z_prob) in enumerate(
                        zip(history_ckpt_paths, history_pipelines, history_frac_ids, history_z_paths, history_z_probs)):
                    if history_player_ids is None:
                        player_id = f'HP{idx}'
                    else:
                        player_id = history_player_ids[idx]
                    hp = HistoricalPlayer(
                        checkpoint_path=ckpt_path,
                        player_id=player_id,
                        pipeline=pipeline,
                        frac_id=frac_id,
                        z_path=z_path,
                        z_prob=z_prob,
                        total_agent_step=0,
                        decay=self._stat_decay,
                        warm_up_size=self._stat_warm_up_size,
                        min_win_rate_games=self._payoff_min_win_rate_games,
                        parent_id='none',
                    )
                    self.historical_players[player_id] = hp
        # save active players' player_id
        self.logger.info('init league with active players:')
        self.logger.info(pformat(self.active_players.keys()))
        self.logger.info('init league with historical players:')
        self.logger.info(pformat(self.historical_players.keys()))
        self.add_hist_player_count = 0
        for player in self.active_players.values():
            self.save_successive_model(player)

    def add_active_player(self, ckpt_path, pipeline, frac_id, z_path, teacher_id, teacher_ckpt, player_id,
                          one_phase_step, chosen_weight, z_prob, *args, **kargs):
        # Note: when we load league resume, we will not use this funciton add_active_player
        # We will use player_id to determine activer player type
        # MP: mainplayer, ME: main exploiter, EP: League exploiter
        player_type = self.active_player_type(player_id)
        if player_type is None:
            return False
        player_ckpt_path = os.path.join(self._path_model, '{}_ckpt.pth.tar'.format(player_id))
        player = player_type(checkpoint_path=player_ckpt_path,
                             player_id=player_id,
                             pipeline=pipeline,
                             frac_id=frac_id,
                             z_path=z_path,
                             z_prob=z_prob,
                             teacher_id=teacher_id,
                             teacher_checkpoint_path=teacher_ckpt,
                             chosen_weight=chosen_weight,
                             one_phase_step=int(float(one_phase_step)),
                             decay=self._stat_decay,
                             warm_up_size=self._stat_warm_up_size,
                             min_win_rate_games=self._payoff_min_win_rate_games,
                             )
        self.set_player_tblogger(player_id)
        self.setup_tmp_dir(player_id=player_id)
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu')
            # this part is actually for sl pretirained model
            if 'last_iter' in ckpt.keys():
                ckpt['last_iter'] = 0
            torch.save(ckpt, player.checkpoint_path)
        else:
            assert self.cfg.get('fake_model', True), "Can't use fake_model if it is set to be False"
        self.logger.info(
            '[league manager] copy ckpt from {} to {}.'.format(
                ckpt_path, player.checkpoint_path))
        with self._lock:
            self.active_players[player_id] = player
        # Notice: we only initial snapshot for main player
        if isinstance(player, MainPlayer) and self.cfg.get('save_initial_snapshot', False):
            self.save_snapshot(player)
        return True

    def save_snapshot(self, player):
        hp = player.snapshot()
        hp.checkpoint_path = os.path.join(self._path_model,
                                          hp.player_id + '_' + os.path.basename(player.checkpoint_path))
        copyfile(player.checkpoint_path, hp.checkpoint_path)
        self.logger.info(
            '[league manager] copy {} to {}.'.format(
                player.checkpoint_path, hp.checkpoint_path))
        self.set_hist_player(hp)
        return hp.player_id

    def save_successive_model(self, player):
        if not self._whole_cfg.learner.use_dapo:
            return
        model_name = os.path.basename(player.checkpoint_path)
        tmp_path = os.path.abspath(
            os.path.join(os.getcwd(), 'experiments', self._whole_cfg.common.experiment_name,
                         'successive_model/{}/'.format(player.player_id)))
        successive_model_path = os.path.join(tmp_path,model_name)
        if os.path.exists(successive_model_path+'.bak'):
            os.rename(successive_model_path+'.bak',successive_model_path)
        tmp_files = os.listdir(tmp_path) 
        copyfile(player.checkpoint_path,successive_model_path+'.bak')
        for filename in tmp_files:
            os.remove(os.path.join(tmp_path,filename))
        player.successive_model_path = successive_model_path
        player.last_successive_step = player.total_agent_step
        self.logger.info(
            '[league manager] copy {} to {}.'.format(
                player.checkpoint_path, successive_model_path))
    
    def set_hist_player(self, hp):
        self.logger.info(f'add history player:{hp.player_id}')
        with self._lock:
            self.historical_players[hp.player_id] = hp

    @staticmethod
    def child_frac_ids(frac_id):
        if frac_id == 0:
            return [1, 2, 3, -1, -2, -3]
        elif frac_id == -1:
            return [2, 3]
        elif frac_id == -2:
            return [1, 3]
        elif frac_id == -3:
            return [1, 2]
        else:
            return []

    # ************************** learner *********************************
    def deal_with_register_learner(self, request_info):
        player_id = request_info['player_id']
        ip = request_info['ip']
        port = request_info['port']
        rank = request_info['rank']
        world_size = request_info['world_size']
        self.api_info[player_id].append((ip, port, rank, world_size))
        self._logger.info((ip, port, rank, world_size))
        assert player_id in self.active_players.keys(), f'{player_id} not in active players{self.active_players.keys()}'
        self._logger.info('register learner: {}'.format(player_id))
        return {'ckpt_path': self.active_players[player_id].checkpoint_path}

    def deal_with_learner_send_train_info(
            self, request_info: dict):
        """
        Overview:
            Update an active player's info
        Arguments:
            - player_info (:obj:`dict`): an info dict of the player which is to be updated
        """
        player_id = request_info['player_id']
        train_steps = request_info['train_steps']
        ckpt_path = request_info['checkpoint_path']

        player = self.active_players[player_id]
        with self._lock:
            player.total_agent_step += train_steps
            player.checkpoint_path = ckpt_path
        if self.cfg.get('pfsp_train_bot', False) == False:
            historical_players = {player_id: player for player_id, player in self.historical_players.items() if
                                  player.pipeline != 'bot'}
        else:
            historical_players = self.historical_players
        reset_flag = player.reset_flag
        new_hp_id = None
        if player.is_save_successive_model():
            self.save_successive_model(player)
        if player.is_trained_enough(historical_players, self.active_players, pfsp_train_bot=self.cfg.pfsp_train_bot):
            new_hp_id = self.save_snapshot(player)
            reset_flag |= player.is_reset()
        if reset_flag:
            player.reset_flag = False
            print(f"reset {player.player_id} param")
            with self._lock:
                player.reset_stats()
                new_checkpoint_path = player.reset_checkpoint(self.active_players, self.historical_players, new_hp_id)
                player.checkpoint_path = os.path.join(self._path_model, '{}_ckpt.pth.tar'.format(player.player_id))
                copyfile(new_checkpoint_path, player.checkpoint_path)
            self.save_successive_model(player)
            return {'reset_checkpoint_path': player.checkpoint_path}
        return {'reset_checkpoint_path': 'none'}

    # ************************** actor *********************************

    def deal_with_actor_send_result(
            self, request_info: dict):
        """
        Overview:
            Finish current job. Update active players' ``launch_count`` to release job space,
            and shared payoff to record the game result.
        Arguments:
            - job_result (:obj:`dict`): a dict containing job result information
        """
        self._result_queue.put(request_info)
        return True

    def _send_result_loop(self):
        torch.set_num_threads(1)
        while True:
            if self._result_queue.empty():
                time.sleep(0.01)
            else:
                request_info = self._result_queue.get()
                game_steps = request_info.pop('game_steps')
                game_iters = request_info.pop('game_iters')
                game_duration = request_info.pop('game_duration')
                for side_id in request_info.keys():
                    player_id = request_info[side_id]['player_id']
                    player = self.all_players[player_id]
                    winloss_info = {'winrate': (1 + request_info[side_id]['winloss']) / 2,
                                    'game_steps': game_steps,
                                    'game_iters': game_iters,
                                    'game_duration': game_duration,
                                    }
                    with self._lock:
                        if player_id!=request_info[side_id]['opponent_id']:
                            player.payoff.update(opponent_id=request_info[side_id]['opponent_id'], stat_info=winloss_info)
                        player.total_game_count += 1
                with self._lock:
                    self.elo.update(request_info['0']['player_id'], request_info['0']['opponent_id'], request_info['0']['winloss'])
                if self.elo.game_count % 100 == 0:
                    print('result queue size', self._result_queue.qsize())
                    self._logger.info(self.elo.elo_text())
                for side_id in request_info.keys():
                    player_id = request_info[side_id]['player_id']
                    if player_id in self.active_players.keys():
                        frac_id = request_info[side_id]['race_id']
                        # success_stat = request_info[side_id]['success_stat']
                        player = self.all_players[player_id]
                        with self._lock:
                            player.dist_stat.update(frac_id, request_info[side_id])
                            # player.unit_num_stat.update(frac_id, side_id, request_info[side_id])
                            player.cum_stat.update(frac_id, request_info[side_id])

                player_ids_set = {val['player_id'] for val in request_info.values()}
                for player_id in player_ids_set:
                    player = self.all_players[player_id]
                    if isinstance(player,ActivePlayer):
                        if player.total_game_count % self.cfg.print_freq == 0:
                            for opponent_id, info in player.payoff.stat_info_dict.items():
                                for k, val in info.items():
                                    self.tb_log_dict(player_id).add_scalar(
                                        tag=f'{k}/{opponent_id}',
                                        scalar_value=val,
                                        global_step=player.total_game_count)

                            for frac_id, info in player.dist_stat.stat_info_dict.items():
                                for k, val in info.items():
                                    self.tb_log_dict(player_id).add_scalar(
                                        tag=f'{frac_id}/{k}',
                                        scalar_value=val,
                                        global_step=player.dist_stat.game_count[frac_id])
                            for frac_id, info in player.cum_stat.stat_info_dict.items():
                                for k, val in info.items():
                                    for i in range(4):
                                        self.tb_log_dict(player_id).add_scalar(
                                            tag=f'cum_{frac_id}_{i}/{k}',
                                            scalar_value=val[i],
                                            global_step=player.cum_stat.game_count[frac_id])
                            if (self.cfg.get('active_payoff_log', True) and isinstance(player, ActivePlayer)) or \
                                    (self.cfg.get('hist_payoff_log', False) and isinstance(player, HistoricalPlayer)):
                                self._logger.info('=' * 30 + f'{player.player_id}' + '=' * 30)
                                self._logger.info(player.payoff.get_text())
                            if (self.cfg.get('active_opponent_payoff_log', False) and isinstance(player, ActivePlayer)) or \
                                    (self.cfg.get('hist_opponent_payoff_log', False) and isinstance(player,
                                                                                                    HistoricalPlayer)):
                                self._logger.info('=' * 30 + f'{player.player_id}' + '=' * 30)
                                self._logger.info(player.opponent_payoff.get_text())

    def choose_active_player(self):
        active_player_ids = list(self.active_players.keys())
        active_player_weights = [self.active_players[player_id].chosen_weight  for player_id in
                                 active_player_ids]
        chosen_player_id = random.choices(active_player_ids, weights=active_player_weights, k=1)[0]
        chosen_player = self.active_players[chosen_player_id]
        return chosen_player

    def deal_with_actor_ask_for_job(self, request_info: dict):
        job_type = request_info['job_type']
        print(job_type, '!!!!!!!!!!!!!!!!!!!', flush=True)
        if job_type == 'train':
            player = self.choose_active_player()
            if self.cfg.get('vs_bot', False):
                branch, job_info = self._get_vs_bot_job_info(player)
            else:
                branch, job_info = self._get_train_job_info(player)
        elif job_type == 'ladder':
            branch, job_info = self._get_ladder_job_info()
        
        map_ids = self.cfg.get('map_names', ['KairosJunction'])
        map_id_weights = self.cfg.get('map_id_weights', [1])
        map_id = random.choices(map_ids, weights=map_id_weights, k=1)[0]
        job_info['env_info']['map_name'] = map_id
        return job_info

    def _get_vs_bot_job_info(self, player):
        assert isinstance(player, ActivePlayer)
        bot_probs = self.cfg.get('bot_probs', [1] * 10)
        bot_level_num = len(bot_probs)
        bot_level = random.choices(range(bot_level_num), weights=bot_probs, k=1)[0]
        bot_race = self.cfg.get('frac_id',1)
        if isinstance(player,MainPlayer):
            successive_ids = [player.player_id]
        else:
            successive_ids = ['none']
        job_info = {'player_ids': [player.player_id],
                    'side_ids': [0],
                    'checkpoint_paths': [player.checkpoint_path],
                    'successive_ids': successive_ids,
                    'pipelines': [player.pipeline],
                    'z_path': [player.z_path],
                    'z_prob': [player.z_prob],
                    'teacher_player_ids': [player.teacher_id],
                    'teacher_checkpoint_paths': [player.teacher_checkpoint_path],
                    'send_data_players': [player.player_id],
                    'update_players': [player.player_id],
                    'frac_ids':[player.frac_id,bot_race],
                    'bot_id':f"bot{bot_level}",
                    'env_info': {
                         'player_ids':[player.player_id,f'bot{bot_level}'],
                         'side_id':[0,1],
                        }
                    }
        branch = 'train_bot'
        return branch, job_info

    def _get_train_job_info(self, player, ):
        assert isinstance(player, ActivePlayer)
        branch, home_team, opponent_team = player.get_branch_opponent(self.historical_players, self.active_players,
                                                                      self.cfg.branch_probs,
                                                                      self.cfg.get('pfsp_train_bot', False))
        players = list(itertools.chain.from_iterable(zip(opponent_team, home_team)))

        successive_ids =[]
        for p in players:
            if isinstance(p,MainPlayer):
                successive_ids.append(p.player_id)
            else:
                successive_ids.append('none')
        job_info = {
            'player_ids': [p.player_id for p in players],
            'side_ids': [idx for idx in range(len(players))],
            'pipelines': [p.pipeline for p in players],
            'checkpoint_paths': [p.checkpoint_path for p in players],
            'successive_ids': successive_ids,
            'z_path': [p.z_path for p in players],
            'z_prob': [p.z_prob for p in players],
            'teacher_player_ids': [p.teacher_id for p in players],
            'teacher_checkpoint_paths': [p.teacher_checkpoint_path for p in players],
            'send_data_players': list({p.player_id for p in players if isinstance(p, ActivePlayer)}),
            'update_players': list({p.player_id for p in players if isinstance(p, ActivePlayer)}),
            'frac_ids':[p.frac_id for p in players],
            'env_info': {
                'player_ids': [p.player_id for p in players],
                'side_id':[0,1],
            }
        }
        if branch == 'vs_main':
            for idx, p in enumerate(players):
                if isinstance(p, MainPlayer):
                    job_info['teacher_player_ids'][idx] = job_info['teacher_checkpoint_paths'][idx] = 'none'
            job_info['send_data_players'] = \
                list({p.player_id for p in players if
                      isinstance(p, ActivePlayer) and (not isinstance(p, MainPlayer))})
        elif 'eval' in branch:
            job_info['teacher_player_ids'] = job_info['teacher_checkpoint_paths'] = ['none'] * 2
            job_info['send_data_players'] = []
        return branch, job_info

    def _get_ladder_job_info(self):
        less_player_pairs = []
        enough_player_pairs = []
        historical_players = list(self.historical_players.values())
        ladder_bots = self.cfg.get('ladder_bots', ['bot7', 'bot10'])
        if ladder_bots:
            historical_players += ladder_bots
        for hist_player in historical_players:
            for oppo_hist_player in historical_players:
                home_id = hist_player if isinstance(hist_player, str) else hist_player.player_id
                oppo_id = oppo_hist_player if isinstance(oppo_hist_player, str) else oppo_hist_player.player_id
                if 'bot' in home_id or home_id == oppo_id:
                    continue
                elif self.elo.games[home_id][oppo_id] < self.cfg.get('ladder_min_games', 100):
                    less_player_pairs.append([hist_player, oppo_hist_player, home_id, oppo_id])
                else:
                    enough_player_pairs.append([hist_player, oppo_hist_player, home_id, oppo_id])
                    
        if len(less_player_pairs):
            players = random.choice(less_player_pairs)
        else:
            players = random.choice(enough_player_pairs)
        if 'bot' in players[2] or 'bot' in players[3]:
            player_num = 1
            pipelines = [players[0].pipeline, players[1]]
        else:
            pipelines = [p.pipeline for p in players[:2]]
            player_num = 2
        
        job_info = {
            'player_ids': players[2:],
            'side_ids': [0,1],
            'pipelines': pipelines,
            'checkpoint_paths': [p.checkpoint_path for p in players[:player_num]],
            'successive_ids': ['none'] * player_num,
            'z_path': [p.z_path for p in players[:player_num]],
            'z_prob': [p.z_prob for p in players[:player_num]],
            'teacher_player_ids': ['none']*player_num,
            'teacher_checkpoint_paths': ['none'] *player_num,
            'send_data_players': [],
            'update_players': [],
            'frac_ids': [1, 1],
            'env_info': {
                'player_ids': players[2:],
                'side_id':[0,1]
            }
        }
        return 'ladder', job_info

    def _save_resume_thread(self):
        self.lasttime = int(time.time())
        self.logger.info("[UP] check resume thread start")
        while True:
            nowtime = int(time.time())
            if nowtime - self.lasttime >= self.save_resume_freq:
                self.save_resume()
                self.lasttime = nowtime
            time.sleep(self.save_resume_freq)

    def save_resume(self, ):
        resume_data = {}
        resume_data['active_players'] = self.active_players
        resume_data['historical_players'] = self.historical_players
        resume_data['elo_ratings'] = self.elo
        resume_label = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
        resume_data_path = os.path.join(self.resume_dir, f'cmp_league.resume.' + resume_label)
        with open(resume_data_path, "wb") as f:
            compress_data = lz4.frame.compress(pickle.dumps(resume_data))
            f.write(compress_data)
        self.logger.info('[resume] save to {}'.format(resume_data_path))
        return resume_data_path

    def add_hist_player(self, request_info):
        player_id = request_info['player_id']
        ckpt_path = request_info['checkpoint_path']
        pipeline = request_info['pipeline']
        frac_id = request_info['frac_id']
        z_path = request_info['z_path']
        if ckpt_path == 'none' or not os.path.exists(ckpt_path):
            return False
        self.add_hist_player_count += 1
        if player_id == 'none' or not player_id:
            player_id = f'HP_ADD{self.add_hist_player_count}'
        new_ckpt_path = os.path.join(self._path_model,
                                     player_id + '_' + os.path.basename(ckpt_path))
        copyfile(ckpt_path, new_ckpt_path)
        hp = HistoricalPlayer(
            checkpoint_path=new_ckpt_path,
            player_id=player_id,
            pipeline=pipeline,
            frac_id=frac_id,
            z_path=z_path,
            total_agent_step=0,
            decay=self._stat_decay,
            warm_up_size=self._stat_warm_up_size,
            min_win_rate_games=self._payoff_min_win_rate_games,
            parent_id='none',
        )
        self.set_player_tblogger(player_id)
        with self._lock:
            self.historical_players[player_id] = hp
        self.logger.info(f'league_current_hist_players:{list(self.historical_players.keys())}')
        return True

    def show_elo(self):
        text = self.elo.elo_text()
        self.logger.info(text)
        return True

    def reset_player_stat(self, request_info):
        player_id = request_info['player_id']
        stat_types = request_info.get('stat_types', None)
        player_ids = self.get_correspondent_player_ids(player_id)
        for player_id in player_ids:
            with self._lock:
                self.all_players[player_id].reset_stats(stat_types)
        return True

    def display_player(self, request_info):
        player_id = request_info['player_id']
        stat_types = request_info.get('stat_types', [])
        player_ids = self.get_correspondent_player_ids(player_id)
        for player_id in player_ids:
            print(player_id)
            player = self.all_players[player_id]
            print(player)
            for stat_type in stat_types:
                if hasattr(player, stat_type):
                    self.logger.info(getattr(player, stat_type).get_text())
        return True

    def update_player(self, request_info):
        player_id = request_info['player_id']
        if player_id not in self.all_players:
            return False
        player = self.all_players[player_id]
        with self._lock:
            for attr_type in [
                'checkpoint_path', 'pipeline', 'frac_id', 'z_path',
                'teacher_id', 'teacher_checkpoint_path', 'chosen_weight',
                'total_agent_step', 'decay', 'warm_up_size', 'total_game_count',
                'parent_id',
                'one_phase_step', 'last_enough_step', 'snapshot_times', 'strong_win_rate', 'snapshot_flag',
                'reset_flag','z_prob',
            ]:
                if hasattr(player, attr_type) and attr_type in request_info:
                    if attr_type == 'one_phase_step' and isinstance(request_info[attr_type], str):
                        request_info[attr_type] = int(float(request_info[attr_type]))
                    setattr(player, attr_type, request_info[attr_type])
        return True

    def refresh_active_player(self, ):
        for player_id, old_player in self.active_players.items():
            new_player_info = {
                attr_type: getattr(old_player, attr_type, None)
                for attr_type in [
                    'checkpoint_path', 'player_id', 'pipeline', 'frac_id', 'z_path',
                    'teacher_id', 'teacher_checkpoint_path', 'chosen_weight', 
                    'total_agent_step', 'decay', 'warm_up_size', 'min_win_rate_games', 'total_game_count',
                    'one_phase_step', 'last_enough_step', 'snapshot_times', 'strong_win_rate',
                    'payoff', 'teammate_payoff', 'opponent_payoff', 'dist_stat', 'cum_stat', 'unit_num_stat','z_prob','init_players'
                ]}
            player_type = self.active_player_type(player_id)
            new_player_info['min_win_rate_games'] = self._payoff_min_win_rate_games
            new_player = player_type(**new_player_info)
            with self._lock:
                self.active_players[player_id] = new_player
            self.logger.info(f'update active player:{player_id}')
        return True

    def refresh_hist_player(self, ):
        for player_id, old_player in self.historical_players.items():
            new_player_info = {
                attr_type: getattr(old_player, attr_type, None)
                for attr_type in [
                    'checkpoint_path', 'player_id', 'pipeline', 'frac_id', 'z_path',
                    'total_agent_step', 'decay', 'warm_up_size', 'min_win_rate_games', 'total_game_count',
                    'parent_id',
                    'payoff', 'teammate_payoff', 'opponent_payoff', 'dist_stat', 'cum_stat', 'unit_num_stat','z_prob','init_players'
                ]}
            if new_player_info['min_win_rate_games'] is None:
                new_player_info['min_win_rate_games'] = self._payoff_min_win_rate_games
            new_player = HistoricalPlayer(**new_player_info)
            with self._lock:
                self.historical_players[player_id] = new_player
            self.logger.info(f'update historical player:{player_id}')
        return True

    def _load_resume(self, resume_path: str):
        if 'cmp' not in os.path.basename(resume_path):
            resume_data = torch.load(resume_path)
        else:
            with open(resume_path, "rb") as f:
                resume_data = pickle.loads(lz4.frame.decompress(f.read()))
        if isinstance(resume_data, dict):
            self.active_players = resume_data['active_players']
            self.historical_players = resume_data['historical_players']
            self.elo = resume_data['elo_ratings']
            self.elo.init_elo = 1000
            self.show_elo()
            for player_id, player in self.all_players.items():
                if isinstance(player, ActivePlayer):
                    self.set_player_tblogger(player_id)
                if not self.cfg.get('copy_model', True):
                    continue
                    
                old_ckpt_path = player.checkpoint_path
                new_ckpt_path = os.path.join(self._path_model, os.path.basename(old_ckpt_path))
                if not os.path.exists(old_ckpt_path):
                    if os.path.exists(new_ckpt_path):
                        player.checkpoint_path = new_ckpt_path
                    else:
                        print(f"cant find ckpt path:{old_ckpt_path}")
                        raise FileNotFoundError
                else:
                    if old_ckpt_path != new_ckpt_path:
                        player.checkpoint_path = new_ckpt_path
                        copyfile(old_ckpt_path, new_ckpt_path)
                        print(f"copy {player_id} model to path:{new_ckpt_path}")
        if self.cfg.get('extra_resume_path', ''):
            extra_resume_path = self.cfg.extra_resume_path
            if 'cmp' not in os.path.basename(extra_resume_path):
                resume_data = torch.load(extra_resume_path)
            else:
                with open(extra_resume_path, "rb") as f:
                    resume_data = pickle.loads(lz4.frame.decompress(f.read()))
            for p in resume_data['historical_players'].keys():
                if p not in self.historical_players:
                    self.historical_players[p] = resume_data['historical_players'][p]
        if self.cfg.get('0Z_history_player', False):
            historical_players = list(self.historical_players.items())
            for player_id, player in historical_players:
                if player_id + '_0Z' not in self.historical_players and player.z_prob != 1. and '0Z' not in player_id:
                    player_0Z = deepcopy(player)
                    player_0Z.player_id = player_id + '_0Z'
                    player_0Z.z_prob = 0.
                    self.historical_players[player_id + '_0Z'] = player_0Z

        for active_player_id in self.active_players:
            self.setup_tmp_dir(player_id=active_player_id)
        for player in self.active_players.values(): 
            self.save_successive_model(player)

    def load_resume(self, request_info):
        resume_path = request_info['path']
        if resume_path and os.path.exists(resume_path):
            self.logger.info(
                'load league, path: {}'.format(resume_path))
            self._load_resume(resume_path)
            return True
        else:
            return False

    def remove_hist_player(self, request_info):
        player_id = request_info['player_id']
        player_ids = self.get_correspondent_player_ids(player_id)
        for del_player_id in player_ids:
            if del_player_id not in self.historical_players:
                print(f'{del_player_id} not in historical players')
                continue
            with self._lock:
                self.historical_players.pop(del_player_id, None)
                for player in self.all_players.values():
                    if del_player_id in player.payoff._stat_info_record.keys():
                        del player.payoff._stat_info_record[del_player_id]
                    if del_player_id in player.teammate_payoff._stat_info_record.keys():
                        del player.teammate_payoff._stat_info_record[del_player_id]
                    if del_player_id in player.opponent_payoff._stat_info_record.keys():
                        del player.opponent_payoff._stat_info_record[del_player_id]
                self.logger.info(f'remove player{del_player_id}')
        return True

    def setup_tmp_dir(self, player_id):
        tmp_model_path = os.path.abspath(
            os.path.join(os.getcwd(), 'experiments', self._whole_cfg.common.experiment_name,
                         'tmp/{}/model'.format(player_id)))
        tmp_successive_model_path = os.path.abspath(
            os.path.join(os.getcwd(), 'experiments', self._whole_cfg.common.experiment_name,
                         'successive_model/{}/'.format(player_id))) 
        tmp_traj_path = os.path.abspath(os.path.join(os.getcwd(), 'experiments', self._whole_cfg.common.experiment_name,
                                                     'tmp/{}/traj'.format(player_id)))
        if os.path.exists(tmp_traj_path):
            try:
                rmtree(tmp_traj_path)
            except:
                pass
        if os.path.exists(tmp_successive_model_path):
            try:
                rmtree(tmp_successive_model_path)
            except:
                pass
        
        try:
            os.makedirs(tmp_traj_path)
        except:
            pass

        if not os.path.exists(tmp_model_path):
            try:
                os.makedirs(tmp_model_path)
            except:
                pass
        try:
            os.makedirs(tmp_successive_model_path)
        except:
            pass

    @staticmethod
    def active_player_type(player_id):
        if 'MP' in player_id:
            return MainPlayer
        elif 'ME' in player_id:
            return MainExploiterPlayer
        elif 'EP' in player_id:
            return ExploiterPlayer
        elif 'EX' in player_id:
            return ExpertPlayer
        elif 'AE' in player_id:
            return AdaptiveEvolutionaryExploiterPlayer
        elif 'EE' in player_id:
            return ExpertExploiterPlayer
        else:
            print(f"not support{player_id} for active players, must include one of ['MP','ME','EP','EX','AE']")
            return None

    @staticmethod
    def active_player_type_str(player_id):
        if 'MP' in player_id:
            return 'MainPlayer'
        elif 'ME' in player_id:
            return 'MainExploiterPlayer'
        elif 'EP' in player_id:
            return 'ExploiterPlayer'
        elif 'EX' in player_id:
            return 'ExpertPlayer'
        elif 'AE' in player_id:
            return 'AdaptiveEvolutionaryExploiterPlayer'
        else:
            print(f"not support{player_id} for active players, must include one of ['MP','ME','EP','EX','AE']")
            return None

    @property
    def logger(self):
        return self._logger

    @property
    def all_players(self):
        return {**self.historical_players, **self.active_players}

    @property
    def cfg(self):
        with self._lock:
            return self._whole_cfg.league

    def tb_log_dict(self, player_id):
        if player_id in self._tb_log_dict:
            return self._tb_log_dict[player_id]
        else:
            self.set_player_tblogger(player_id)
            return self._tb_log_dict[player_id]

    def get_correspondent_player_ids(self, player_id):
        if player_id == 'all':
            player_ids = self.all_players.keys()
        elif player_id == 'active':
            player_ids = self.active_players.keys()
        elif player_id == 'hist':
            player_ids = self.historical_players.keys()
        elif isinstance(player_id, list):
            player_ids = [p for p in player_id if p in self.all_players]
        elif player_id not in self.all_players:
            print(f'{player_id} not in league pool')
            player_ids = []
        else:
            player_ids = [player_id]
        return player_ids


def default_elo_rating(num):
    return num


if __name__ == '__main__':
    import os.path as osp

    cfg = read_config(
        osp.join(
            osp.dirname(__file__),
            "league_default_config.yaml"))
    league = League(cfg)
