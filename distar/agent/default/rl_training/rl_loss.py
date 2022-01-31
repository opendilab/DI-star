import os.path as osp

import torch
from distar.ctools.utils import deep_merge_dicts, read_config
from .as_rl_utils import td_lambda_loss,entropy_loss,policy_gradient_loss,upgo_loss,kl_loss,dapo_loss

default_config = read_config(osp.join(osp.dirname(__file__), "default_reinforcement_loss.yaml"))

class ReinforcementLoss:
    def __init__(self, learner_cfg: dict, player_id) -> None:
        # Here learner_cfg is self._whole_cfg.learner
        self.cfg = deep_merge_dicts(default_config.learner,learner_cfg)

        # policy parameters
        self.gammas = self.cfg.gammas

        # loss weight
        self.loss_weights = self.cfg.loss_weights
        self.action_type_kl_steps = self.cfg.kl.action_type_kl_steps
        self.dapo_steps = self.cfg.dapo.dapo_steps
        self.use_dapo = self.cfg.use_dapo
        if 'MP' not in player_id:
            self.use_dapo = False
            self.loss_weights.dapo = 0.0
        self.dapo_head_weights = self.cfg.dapo_head_weights
        self.pg_head_weights = self.cfg.pg_head_weights
        self.upgo_head_weights = self.cfg.upgo_head_weights
        self.entropy_head_weights = self.cfg.entropy_head_weights
        self.kl_head_weights = self.cfg.kl_head_weights
        self.only_update_value = False
        self.use_total_rhos = self.cfg.get('use_total_rhos',False)

    def compute_loss(self, inputs):
        # learner compute action_value result
        target_policy_logits_dict = inputs['target_logit']  # shape (T,B)
        baseline_values_dict = inputs['value']  # shape (T+1,B)

        behaviour_action_log_probs_dict = inputs['action_log_prob']  # shape (T,B)
        teacher_policy_logits_dict = inputs['teacher_logit']  # shape (T,B)
        if self.use_dapo:
            successive_policy_logits_dict = inputs['successive_logit']
        masks_dict = inputs['mask']  # shape (T,B)
        actions_dict = inputs['action']  # shape (T,B)
        rewards_dict = inputs['reward']  # shape (T,B)
        # dones = inputs['done']  # shape (T,B)
        game_steps = inputs['step']  # shape (T,B) target_action_log_prob
        flag = rewards_dict['winloss'][-1] == 0
        for filed in baseline_values_dict.keys():
            baseline_values_dict[filed][-1] *= flag
        # ===========
        # preparation
        # ===========
        # create loss show dict
        loss_info_dict = {}

        # create preparation info dict
        target_policy_probs_dict = {}
        target_policy_log_probs_dict = {}
        target_action_log_probs_dict = {}
        # log_rhos_dict = {}
        clipped_rhos_dict = {}
        # get distribution info for behaviour policy and target policy
        for head_type in ['action_type', 'delay','queued', 'target_unit', 'selected_units','target_location']:
            # take info from correspondent input dict
            target_policy_logits = target_policy_logits_dict[head_type]

            actions = actions_dict[head_type]
            # compute target log_probs, probs(for entropy,kl), target_action_log_probs, log_rhos(for pg_loss,upgo_loss)
            pi_target = torch.distributions.Categorical(logits=target_policy_logits)
            target_policy_probs = pi_target.probs
            target_policy_log_probs = pi_target.logits
            target_action_log_probs = pi_target.log_prob(actions)

            behaviour_action_log_probs = behaviour_action_log_probs_dict[head_type]
            with torch.no_grad():
                log_rhos = target_action_log_probs - behaviour_action_log_probs
                if head_type == 'selected_units':
                    log_rhos *= masks_dict['selected_units_mask']
                    log_rhos = log_rhos.sum(dim=-1)
                rhos = torch.exp(log_rhos)
                clipped_rhos = rhos.clamp_(max=1)
            # save preparation results to correspondent dict
            target_policy_probs_dict[head_type] = target_policy_probs
            target_policy_log_probs_dict[head_type] = target_policy_log_probs
            if head_type == 'selected_units':
                target_action_log_probs.masked_fill_(~masks_dict['selected_units_mask'], 0)
                target_action_log_probs = target_action_log_probs.sum(-1)
            target_action_log_probs_dict[head_type] = target_action_log_probs
            # log_rhos_dict[head_type] = log_rhos
            clipped_rhos_dict[head_type] = clipped_rhos

        # ====================
        # policy gradient loss
        # ====================
        total_policy_gradient_loss = 0
        policy_gradient_loss_dict = {}

        for field, baseline in baseline_values_dict.items():
            baseline_value = baseline_values_dict[field]
            reward = rewards_dict[field]
            field_policy_gradient_loss, field_policy_gradient_loss_dict = \
                policy_gradient_loss(baseline_value, reward, target_action_log_probs_dict, clipped_rhos_dict,
                                     masks_dict, head_weights_dict=self.pg_head_weights, gamma=1.0, field=field)

            total_policy_gradient_loss += self.loss_weights.pg[field] * field_policy_gradient_loss
            for k, val in field_policy_gradient_loss_dict.items():
                policy_gradient_loss_dict[field + '/' + k] = val

        loss_info_dict.update(policy_gradient_loss_dict)
        # ===========
        # upgo loss
        # ===========
        total_upgo_loss, upgo_loss_dict = upgo_loss(
            baseline_values_dict['winloss'], rewards_dict['winloss'], target_action_log_probs_dict, clipped_rhos_dict,
            masks_dict['actions_mask'], self.upgo_head_weights)

        total_upgo_loss *= self.loss_weights.upgo.winloss
        loss_info_dict.update(upgo_loss_dict)

        # ===========
        # critic loss
        # ===========
        total_critic_loss = 0

        # field is from ['winloss', 'build_order','built_unit','effect','upgrade','battle']
        for field, baseline in baseline_values_dict.items():
            reward = rewards_dict[field]
            # td_lambda_loss = self._td_lambda_loss(baseline, reward) * self.loss_weights.baseline[field]

            # Notice: in general, we need to include done when we consider discount factor, but in our implementation
            # of alphastar, traj_data(with size equal to unroll-len) sent from actor comes from the same episode.
            # If the game is draw, we don't consider it is actually done
            critic_loss = td_lambda_loss(baseline, reward,masks_dict, gamma=self.gammas.baseline[field], field=field)

            total_critic_loss += self.loss_weights.baseline[field] * critic_loss
            loss_info_dict[field + '/td'] = critic_loss.item()
            loss_info_dict[field + '/reward'] = reward.float().mean().item()
            loss_info_dict[field + '/value'] = baseline.mean().item()
        loss_info_dict['battle' + '/reward'] = rewards_dict['battle'].float().mean().item()
        # ============
        # entropy loss
        # ============
        total_entropy_loss, entropy_dict = \
            entropy_loss(target_policy_probs_dict, target_policy_log_probs_dict, masks_dict,
                         head_weights_dict=self.entropy_head_weights)

        total_entropy_loss *= self.loss_weights.entropy
        loss_info_dict.update(entropy_dict)

        # =======
        # kl loss
        # =======

        total_kl_loss, action_type_kl_loss, kl_loss_dict = \
            kl_loss(target_policy_log_probs_dict, teacher_policy_logits_dict, masks_dict, game_steps,
                    action_type_kl_steps=self.action_type_kl_steps, head_weights_dict=self.kl_head_weights)

        total_kl_loss *= self.loss_weights.kl
        action_type_kl_loss *= self.loss_weights.action_type_kl
        loss_info_dict.update(kl_loss_dict)

        # =========
        # DAPO loss
        # =========
        if self.use_dapo:
            total_dapo_loss, dapo_loss_dict = \
                dapo_loss(target_policy_log_probs_dict, successive_policy_logits_dict, masks_dict, game_steps,
                        dapo_steps=self.dapo_steps, head_weights_dict=self.dapo_head_weights) 
            total_dapo_loss *= self.loss_weights.dapo
            loss_info_dict.update(dapo_loss_dict)
        else:
            total_dapo_loss = 0.0

        if self.only_update_value:
            total_loss = total_critic_loss
        else:
            total_loss = total_policy_gradient_loss + \
                         total_upgo_loss + \
                         total_critic_loss + \
                         total_entropy_loss + \
                         total_kl_loss + \
                         action_type_kl_loss + \
                         total_dapo_loss
        loss_info_dict['total_loss'] = total_loss
        return loss_info_dict

    def reset(self, learner_cfg):
        self.cfg = deep_merge_dicts(self.cfg, learner_cfg)
        # policy parameters
        self.gammas = self.cfg.gammas

        # loss weight
        self.loss_weights = self.cfg.loss_weights
        self.action_type_kl_steps = self.cfg.kl.action_type_kl_steps
        self.pg_head_weights = self.cfg.pg_head_weights
        self.upgo_head_weights = self.cfg.upgo_head_weights
        self.entropy_head_weights = self.cfg.entropy_head_weights
        self.kl_head_weights = self.cfg.kl_head_weights
        self.only_update_value = False
