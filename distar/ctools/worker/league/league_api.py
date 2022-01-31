import logging
import os
import time

from flask import Flask, request

from distar.ctools.utils.config_helper import save_config
from distar.ctools.worker.league.league import League

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


def create_league_app(league: League):
    app = Flask(__name__)

    def build_ret(code, info=''):
        return {'code': code, 'info': info}

    # ************************** learner *********************************
    @app.route('/league/register_learner', methods=['POST'])
    def register_learner():
        ret_info = league.deal_with_register_learner(request.json)
        if ret_info:
            return build_ret(0, ret_info)
        else:
            return build_ret(1)


    @app.route('/league/learner_send_train_info', methods=['POST'])
    def send_train_info():
        ret_info = league.deal_with_learner_send_train_info(request.json)
        if ret_info:
            return build_ret(0, ret_info)
        else:
            return build_ret(1)

    # ************************** actor *********************************
    @app.route('/league/actor_ask_for_job', methods=['POST'])
    def ask_for_job():
        ret_info = league.deal_with_actor_ask_for_job(request.json)
        if ret_info:
            return build_ret(0, ret_info)
        else:
            return build_ret(1)

    @app.route('/league/actor_send_result', methods=['POST'])
    def send_result():
        ret_info = league.deal_with_actor_send_result(request.json)
        if ret_info:
            return build_ret(0)
        else:
            return build_ret(1)

    # ************************** for debug use *********************************
    @app.route('/league/show_payoff', methods=['GET'])
    def show_payoff():
        for player_id in league.active_players:
            league.logger.info('=' * 20 + player_id + '=' * 20)
            league.logger.info(league.active_players[player_id].payoff.get_text())
        return {'Done': 'successfully show_pay_off'}

    @app.route('/league/show_dist_stat', methods=['GET'])
    def show_dist_stat():
        for player_id in league.active_players:
            league.logger.info('=' * 20 + player_id + '=' * 20)
            league.logger.info(league.active_players[player_id].dist_stat.get_text())
        return {'Done': 'successfully show_dist_stat'}

    @app.route('/league/show_cum_stat', methods=['GET'])
    def show_cum_stat():
        for player_id in league.active_players:
            league.logger.info('=' * 20 + player_id + '=' * 20)
            league.logger.info(league.active_players[player_id].cum_stat.get_text())
        return {'Done': 'successfully show_cum_stat'}

    @app.route('/league/show_unit_num_stat', methods=['GET'])
    def show_unit_num_stat():
        for player_id in league.active_players:
            league.logger.info('=' * 20 + player_id + '=' * 20)
            league.logger.info(league.active_players[player_id].unit_num_stat.get_text())
        return {'Done': 'successfully show_unit_num_stat'}

    @app.route('/league/show_opponent_payoff', methods=['GET'])
    def show_opponent_payoff():
        for player_id in league.active_players:
            league.logger.info('=' * 20 + player_id + '=' * 20)
            league.logger.info(league.active_players[player_id].opponent_payoff.get_text())
        return {'Done': 'successfully show_pay_off'}

    @app.route('/league/show_teammate_payoff', methods=['GET'])
    def show_teammate_payoff():
        for player_id in league.active_players:
            league.logger.info('=' * 20 + player_id + '=' * 20)
            league.logger.info(league.active_players[player_id].teammate_payoff.get_text())
        return {'Done': 'successfully show_pay_off'}

    @app.route('/league/show_hist_payoff', methods=['GET'])
    def show_hist_payoff():
        for player_id in league.historical_players:
            league.logger.info('=' * 20 + player_id + '=' * 20)
            league.logger.info(league.historical_players[player_id].payoff.get_text())
        return {'Done': 'successfully show_pay_off'}

    @app.route('/league/show_hist_dist_stat', methods=['GET'])
    def show_hist_dist_stat():
        for player_id in league.historical_players:
            league.logger.info('=' * 20 + player_id + '=' * 20)
            league.logger.info(league.historical_players[player_id].dist_stat.get_text())
        return {'Done': 'successfully show_dist_stat'}

    @app.route('/league/show_hist_cum_stat', methods=['GET'])
    def show_hist_cum_stat():
        for player_id in league.historical_players:
            league.logger.info('=' * 20 + player_id + '=' * 20)
            league.logger.info(league.historical_players[player_id].cum_stat.get_text())
        return {'Done': 'successfully show_hist_cum_stat'}

    @app.route('/league/show_hist_unit_num_stat', methods=['GET'])
    def show_hist_unit_num_stat():
        for player_id in league.historical_players:
            league.logger.info('=' * 20 + player_id + '=' * 20)
            league.logger.info(league.historical_players[player_id].unit_num_stat.get_text())
        return {'Done': 'successfully show_unit_num_stat'}

    @app.route('/league/show_hist_opponent_payoff', methods=['GET'])
    def show_hist_opponent_payoff():
        for player_id in league.historical_players:
            league.logger.info('=' * 20 + player_id + '=' * 20)
            league.logger.info(league.historical_players[player_id].opponent_payoff.get_text())
        return {'Done': 'successfully show_pay_off'}

    @app.route('/league/show_hist_teammate_payoff', methods=['GET'])
    def show_hist_teammate_payoff():
        for player_id in league.historical_players:
            league.logger.info('=' * 20 + player_id + '=' * 20)
            league.logger.info(league.historical_players[player_id].teammate_payoff.get_text())
        return {'Done': 'successfully show_pay_off'}

    @app.route('/league/save_resume', methods=['GET'])
    def save_resume():
        try:
            resume_path = league.save_resume()
            return {'resume_path':resume_path}
        except Exception as e:
            print(e)
            return e

    @app.route('/league/show_config', methods=['GET'])
    def show_cfg():
        time_label = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))

        config_dir = os.path.join(os.getcwd(), 'experiments', league._whole_cfg.common.experiment_name, 'config')
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
        config_path = os.path.join(config_dir, f'user_config_{time_label}.yaml')
        save_config(league._whole_cfg, config_path)

        print(f'save config to config_path:{config_path}')

        return league._whole_cfg

    @app.route('/league/update_config', methods=['GET'])
    def update_cfg():
        import os
        import time
        from distar.ctools.utils.config_helper import read_config, deep_merge_dicts, save_config

        resume_path = league.save_resume()
        print(f'save resume to resume_path:{resume_path}')
        time_label = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))

        config_dir = os.path.join(os.getcwd(), 'experiments', league._whole_cfg.common.experiment_name, 'config')
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
        config_path = os.path.join(config_dir, f'user_config_{time_label}.yaml')
        save_config(league._whole_cfg, config_path)
        print(f'save config to config_path:{config_path}')
        load_config_path = os.path.join(os.getcwd(), 'experiments', league._whole_cfg.common.experiment_name,
                                        f'user_config.yaml')
        load_config = read_config(load_config_path)
        league._whole_cfg = deep_merge_dicts(league._whole_cfg, load_config)

        print(f'update config from config_path:{load_config_path}')

        return league._whole_cfg

    @app.route('/league/add_hist_player', methods=['POST'])
    def add_hist_player():
        ret_info = league.add_hist_player(request.json)
        if ret_info:
            return build_ret(0, )
        else:
            return build_ret(1)

    @app.route('/league/add_active_player', methods=['POST'])
    def add_active_player():
        ret_info = league.add_active_player(**request.json)
        league.register_learner()
        if ret_info:
            return build_ret(0, )
        else:
            return build_ret(1)

    @app.route('/league/show_elo', methods=['GET'])
    def show_elo():
        league.show_elo()
        return {'Done': 'successfully show elo ratings'}

    @app.route('/league/save_elo', methods=['GET'])
    def save_elo():
        league.show_elo()
        league.save_elo_ratings(zero_min=False)
        return {'Done': 'successfully save_elo ratings'}

    @app.route('/league/save_zero_elo', methods=['GET'])
    def save_zero_elo():
        league.show_elo()
        league.save_elo_ratings(zero_min=True)
        return {'Done': 'successfully save_zero_elo ratings'}

    @app.route('/league/update_elo', methods=['POST'])
    def update_elo():
        ret_info = league.update_elo(request.json)
        if ret_info:
            return build_ret(0, )
        else:
            return build_ret(1)

    @app.route('/league/show_trueskill', methods=['GET'])
    def show_trueskill():
        league.show_trueskill()
        return {'Done': 'successfully show trueskill ratings'}

    @app.route('/league/save_trueskill', methods=['GET'])
    def save_trueskill():
        league.show_trueskill()
        league.save_trueskill_ratings()
        return {'Done': 'successfully save_trueskill ratings'}

    @app.route('/league/update_trueskill', methods=['POST'])
    def update_trueskill():
        ret_info = league.update_trueskill(request.json)
        if ret_info:
            return build_ret(0, )
        else:
            return build_ret(1)

    @app.route('/league/reset_player_stat', methods=['POST'])
    def reset_player_stat():
        ret_info = league.reset_player_stat(request.json)
        if ret_info:
            return build_ret(0, )
        else:
            return build_ret(1)

    @app.route('/league/display_player', methods=['POST'])
    def display_player():
        ret_info = league.display_player(request.json)
        if ret_info:
            return build_ret(0, )
        else:
            return build_ret(1)

    @app.route('/league/update_player', methods=['POST'])
    def update_player():
        ret_info = league.update_player(request.json)
        if ret_info:
            return build_ret(0, )
        else:
            return build_ret(1)

    @app.route('/league/refresh_active_player', methods=['GET'])
    def refresh_active_player():
        league.refresh_active_player()
        return {'Done': 'successfully refresh_active_player'}

    @app.route('/league/refresh_hist_player', methods=['GET'])
    def refresh_hist_player():
        league.refresh_hist_player()
        return {'Done': 'successfully refresh_hist_player'}

    @app.route('/league/refresh_all_player', methods=['GET'])
    def refresh_all_player():
        league.refresh_active_player()
        league.refresh_hist_player()
        return {'Done': 'successfully refresh_all_player'}

    @app.route('/league/load_resume', methods=['POST'])
    def update_load_resume():
        ret_info = league.load_resume(request.json)
        if ret_info:
            return build_ret(0, )
        else:
            return build_ret(1)

    @app.route('/league/backup_models', methods=['POST'])
    def backup_models():
        ret_info = league.backup_models(request.json)
        if ret_info:
            return build_ret(0, )
        else:
            return build_ret(1)

    @app.route('/league/remove_hist_player', methods=['POST'])
    def remove_hist_player():
        ret_info = league.remove_hist_player(request.json)
        if ret_info:
            return build_ret(0, )
        else:
            return build_ret(1)


    return app
