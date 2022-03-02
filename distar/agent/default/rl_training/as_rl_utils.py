def policy_gradient_loss(baseline_value, reward, target_action_log_probs_dict, clipped_rhos_dict, mask,
                         head_weights_dict, gamma=1.0,field=None ):
    """
        separate vtrace loss
    """
    total_policy_gradient_loss = 0.
    policy_gradient_loss_dict = {}
    for head_type in ['action_type', 'delay', 'queued', 'selected_units',  'target_unit',
                      'target_location']:
        clipped_rhos = clipped_rhos_dict[head_type]
        target_action_log_probs = target_action_log_probs_dict[head_type]

        with torch.no_grad():
            # rho = target_prob / behaviour_prob
            advantages = vtrace_advantages(clipped_rhos, clipped_rhos, reward, baseline_value, gammas=gamma, lambda_=1.0)
        if head_type in ['action_type','delay']:
            policy_gradient_loss = - advantages * target_action_log_probs
        else:
            policy_gradient_loss = - advantages * target_action_log_probs * mask['actions_mask'][head_type]
        if field in ['build_order','built_unit','effect']:
            policy_gradient_loss *= mask[field+'_mask']
        policy_gradient_loss = policy_gradient_loss.mean()
        total_policy_gradient_loss += (policy_gradient_loss * head_weights_dict[head_type])
        policy_gradient_loss_dict[head_type] = policy_gradient_loss.item()

    policy_gradient_loss_dict['total'] = total_policy_gradient_loss.item()

    return total_policy_gradient_loss, policy_gradient_loss_dict


def upgo_loss(baseline_value, reward, target_action_log_probs_dict, clipped_rhos_dict, mask, head_weights_dict):
    total_upgo_loss = 0.
    upgo_loss_dict = {}
    for head_type in ['action_type', 'delay', 'queued', 'selected_units',  'target_unit',
                      'target_location']:
        clipped_rhos = clipped_rhos_dict[head_type]
        target_action_log_probs = target_action_log_probs_dict[head_type]

        with torch.no_grad():
            advantages = clipped_rhos * (upgo_returns(reward, baseline_value) - baseline_value[:-1])
        if head_type in ['action_type','delay']:
            upgo_loss = - advantages * target_action_log_probs
        else:
            upgo_loss = - advantages * target_action_log_probs * mask[head_type]
        upgo_loss = upgo_loss.mean()
        total_upgo_loss += (upgo_loss * head_weights_dict[head_type])
        upgo_loss_dict['upgo/' + head_type] = upgo_loss.item()
    upgo_loss_dict['upgo/total'] = total_upgo_loss.item()
    return total_upgo_loss, upgo_loss_dict


def entropy_loss(target_policy_probs_dict, target_policy_log_probs_dict, mask, head_weights_dict):
    total_entropy_loss = 0
    entropy_dict = {}
    for head_type in ['action_type', 'delay', 'queued', 'selected_units',  'target_unit',
                      'target_location']:
        ent = - target_policy_probs_dict[head_type] * target_policy_log_probs_dict[head_type]
        if head_type == 'selected_units':
            ent = ent.sum(dim=-1) / (1e-9 + torch.log(mask['selected_units_logits_mask'].float().sum(dim=-1)+1).unsqueeze(-1))  # normalize
            ent = (ent * mask['selected_units_mask']).sum(-1)
            ent = ent.div(mask['selected_units_mask'].sum(-1) + 1e-9)
        elif head_type == 'target_unit':
            ent = ent.sum(dim=-1) / (1e-9 + torch.log(mask['target_units_logits_mask'].float().sum(dim=-1)+1))  # normalize
        else:
            ent = ent.sum(dim=-1) / torch.log(torch.FloatTensor([ent.shape[-1]]).to(ent.device))
        if head_type not in ['action_type','delay']:
            ent = ent * mask['actions_mask'][head_type]
        entropy = ent.mean()
        entropy_dict['entropy/' + head_type] = entropy.item()
        total_entropy_loss += (-entropy * head_weights_dict[head_type])
    entropy_dict['entropy/total'] = total_entropy_loss.item()
    return total_entropy_loss, entropy_dict


def kl_loss(target_policy_log_probs_dict, teacher_policy_logits_dict, mask, game_steps,
            action_type_kl_steps, head_weights_dict):
    total_kl_loss = 0
    kl_loss_dict = {}

    for head_type in ['action_type', 'queued', 'delay', 'selected_units',  'target_unit',
                      'target_location']:
        target_policy_log_probs = target_policy_log_probs_dict[head_type]
        teacher_policy_logits = teacher_policy_logits_dict[head_type]

        teacher_policy_log_probs = F.log_softmax(teacher_policy_logits, dim=-1)
        teacher_policy_probs = torch.exp(teacher_policy_log_probs)
        kl = teacher_policy_probs * (teacher_policy_log_probs - target_policy_log_probs)

        kl = kl.sum(dim=-1)
        if head_type == 'selected_units':
            kl = (kl * mask['selected_units_mask']).sum(-1)
        if head_type not in ['action_type','delay']:
            kl = kl * mask['actions_mask'][head_type]
        if head_type == 'action_type':
            flag = game_steps < action_type_kl_steps
            action_type_kl = kl * flag * mask['cum_action_mask']
            action_type_kl_loss = action_type_kl.mean()
            kl_loss_dict['kl/extra_at'] = action_type_kl_loss.item()
        kl_loss = kl.mean()
        total_kl_loss += (kl_loss * head_weights_dict[head_type])
        kl_loss_dict['kl/' + head_type] = kl_loss.item()
    kl_loss_dict['kl/total'] = total_kl_loss.item()
    return total_kl_loss, action_type_kl_loss, kl_loss_dict

def dapo_loss(target_policy_log_probs_dict, successive_policy_logits_dict, mask, game_steps,
            dapo_steps, head_weights_dict):
    total_kl_loss = 0
    kl_loss_dict = {}
    flag = game_steps < dapo_steps
    for head_type in ['action_type', 'delay','queued', 'selected_units', 'target_unit',
                      'target_location']:
        target_policy_log_probs = target_policy_log_probs_dict[head_type]
        successive_policy_logits = successive_policy_logits_dict[head_type]
        successive_policy_log_probs = F.log_softmax(successive_policy_logits, dim=-1)
        successive_policy_probs = torch.exp(successive_policy_log_probs)
        kl = successive_policy_probs * (successive_policy_log_probs - target_policy_log_probs)
        kl = kl.sum(dim=-1)
        if head_type == 'selected_units':
            kl = (kl * mask['selected_units_mask']).sum(-1)
        if head_type not in ['action_type','delay']:
            kl = kl * mask['actions_mask'][head_type]
        kl = kl * flag
        kl_loss = kl.mean()
        total_kl_loss += (kl_loss * head_weights_dict[head_type])
        kl_loss_dict['battle/' + head_type] = kl_loss.item()
    kl_loss_dict['battle/total'] = total_kl_loss.item()
    return total_kl_loss, kl_loss_dict


"""Library for RL returns and losses evaluation"""

import torch
import torch.nn.functional as F


def fn(x):
    return x.unsqueeze(0).unsqueeze(0)


def tb_cross_entropy(logit, label):
    assert (len(label.shape) >= 2)
    T, B = label.shape[:2]
    # special 2D case
    if label.shape[2] == 2 and label.shape[2] != logit.shape[2]:
        assert (len(label.shape) == 3)
        n_output_shape = logit.shape[2:]
        label = label[..., 0] * n_output_shape[1] + label[..., 1]
        logit = logit.reshape(T, B, -1)

    label = label.reshape(-1)
    logit = logit.reshape(-1, logit.shape[-1])
    ce = F.cross_entropy(logit, label, reduction='none')
    ce = ce.reshape(T, B, -1)
    return ce.mean(dim=2)


def multistep_forward_view(rewards, gammas, bootstrap_values, lambda_):
    r"""
    Overview:
        Same as trfl.sequence_ops.multistep_forward_view
        Implementing (12.18) in Sutton & Barto
        ```
        result[T-1] = rewards[T-1] + gammas[T-1] * bootstrap_values[T]
        for t in 0...T-2 :
        result[t] = rewards[t] + gammas[t]*(lambdas[t]*result[t+1] + (1-lambdas[t])*bootstrap_values[t+1])
        ```
        Assuming the first dim of input tensors correspond to the index in batch
        There is no special handling for terminal state value,
        if some state has reached the terminal, just fill in zeros for values and rewards beyond terminal
        (including the terminal state, which is, bootstrap_values[terminal] should also be 0)
    Arguments:
        - rewards (:obj:`torch.Tensor`): the returns from 0 to T-1, of size [T_traj, batchsize]
        - gammas (:obj:`torch.Tensor`): discount factor for each step (from 0 to T-1), of size [T_traj, batchsize]
        - bootstrap_values (:obj:`torch.Tensor`): estimation of the value at *step 1 to T*, of size [T_traj, batchsize]
        - lambda_ (:obj:`torch.Tensor`): determining the mix of bootstrapping
        vs further accumulation of multistep returns at each timestep of size [T_traj, batchsize],
        the element for T-1 is ignored and effectively set to 0,
        as there is no information about future rewards.
    Returns:
        - ret (:obj:`torch.Tensor`): Computed lambda return value
         for each state from 0 to T-1, of size [T_traj, batchsize]
    """
    result = torch.empty_like(rewards)
    # Forced cutoff at the last one
    result[-1, :] = rewards[-1, :] + gammas[-1, :] * bootstrap_values[-1, :]
    discounts = gammas * lambda_
    for t in reversed(range(rewards.size()[0] - 1)):
        result[t, :] = rewards[t, :] \
                       + discounts[t, :] * result[t + 1, :] \
                       + (gammas[t, :] - discounts[t, :]) * bootstrap_values[t, :]

    return result


def generalized_lambda_returns(rewards, gammas, bootstrap_values, lambda_):
    r"""
    Overview:
        Functional equivalent to trfl.value_ops.generalized_lambda_returns
        https://github.com/deepmind/trfl/blob/2c07ac22512a16715cc759f0072be43a5d12ae45/trfl/value_ops.py#L74
        Passing in a number instead of tensor to make the value constant for all samples in batch
    Arguments:
        - rewards (:obj:`torch.Tensor`): the returns from 0 to T-1, of size [T_traj, batchsize]
        - gammas (:obj:`torch.Tensor` or :obj:`float`):
          discount factor for each step (from 0 to T-1), of size [T_traj, batchsize]
        - bootstrap_values (:obj:`torch.Tensor` or :obj:`float`):
          estimation of the value at step 0 to *T*, of size [T_traj+1, batchsize]
        - lambda_ (:obj:`torch.Tensor` or :obj:`float`): determining the mix of bootstrapping
          vs further accumulation of multistep returns at each timestep, of size [T_traj, batchsize]
    Returns:
        - ret (:obj:`torch.Tensor`): Computed lambda return value
          for each state from 0 to T-1, of size [T_traj, batchsize]
    """
    if not isinstance(gammas, torch.Tensor):
        gammas = gammas * torch.ones_like(rewards)
    if not isinstance(lambda_, torch.Tensor):
        lambda_ = lambda_ * torch.ones_like(rewards)
    bootstrap_values_tp1 = bootstrap_values[1:, :]
    return multistep_forward_view(rewards, gammas, bootstrap_values_tp1, lambda_)


def td_lambda_loss(values, rewards, mask=None, gamma=1.0, lambda_=0.8,field = None):
    r"""
    Overview:
        Computing TD($\lambda$) loss given constant gamma and lambda.
        There is no special handling for terminal state value,
        if some state has reached the terminal, just fill in zeros for values and rewards beyond terminal
        (*including the terminal state*, values[terminal] should also be 0)
    Arguments:
        - values (:obj:`torch.Tensor`): estimation of the state value at step 0 to T, of size [T_traj+1, batchsize]
        - rewards (:obj:`torch.Tensor`): the returns from time step 0 to T-1, of size [T_traj, batchsize]
        - gamma (:obj:`float`): constant gamma
        - lambda_ (:obj:`float`): constant lambda (between 0 to 1)
    Returns:
        - loss (:obj:`torch.Tensor`): Computed MSE loss, averaged over the batch, of size []
    """
    with torch.no_grad():
        returns = generalized_lambda_returns(rewards, gamma, values, lambda_)
    # discard the value at T as it should be considered in the next slice
    loss = 0.5 * torch.pow(returns - values[:-1], 2)
    if field in ['build_order','built_unit','effect']:
        loss *= mask[field+'_mask']
    loss = loss.mean()
    return loss


def compute_neg_log_prob(logits, actions, mask=None):
    if len(actions.shape) == 1:
        neg_log_prob = F.cross_entropy(logits, actions, reduction='none')
    elif len(actions.shape) == 2:
        # In selected units head, compute probability for each selection, multiple them together as final probability
        batch_size, selections, n = logits.shape
        actions_flat = actions.view(-1)
        logits_flat = logits.view(-1, n)
        neg_log_prob = F.cross_entropy(logits_flat, actions_flat, reduction='none')
        neg_log_prob = neg_log_prob.view(batch_size, selections)
        #  mask out invalid selections
        if mask is not None:
            neg_log_prob *= mask
        neg_log_prob = neg_log_prob.sum(dim=1)
    else:
        raise NotImplementedError
    return neg_log_prob


def upgo_returns(rewards, bootstrap_values):
    r"""
    Overview:
        Computing UPGO return targets. Also notice there is no special handling for the terminal state.
    Arguments:
        - rewards (:obj:`torch.Tensor`): the returns from time step 0 to T-1, of size [T_traj, batchsize]
        - bootstrap_values (:obj:`torch.Tensor`):
          estimation of the state value at step 0 to T, of size [T_traj+1, batchsize]
    Returns:
        - ret (:obj:`torch.Tensor`): Computed lambda return value
          for each state from 0 to T-1, of size [T_traj, batchsize]
    """
    # UPGO can be viewed as a lambda return! The trace continues for V_t (i.e. lambda = 1.0) if r_tp1 + V_tp2 > V_tp1.
    # as the lambdas[-1, :] is ignored in generalized_lambda_returns, we don't care about bootstrap_values_tp2[-1]
    lambdas = (rewards + bootstrap_values[1:]) >= bootstrap_values[:-1]
    lambdas = torch.cat([lambdas[1:], torch.ones_like(lambdas[-1:])], dim=0)
    return generalized_lambda_returns(rewards, 1.0, bootstrap_values, lambdas)


def vtrace_advantages(clipped_rhos, clipped_cs, rewards, bootstrap_values, clipped_pg_rhos=None, gammas=1.0,
                      lambda_=0.8):
    r"""
    Overview:
        Computing vtrace advantages.
    Arguments:
        - clipped_rhos (:obj:`torch.Tensor`): clipped importance sampling weights $\rho$, of size [T_traj, batchsize]
        - clipped_cs (:obj:`torch.Tensor`): clipped importance sampling weights c, of size [T_traj, batchsize]
        - rewards (:obj:`torch.Tensor`): the returns from time step 0 to T-1, of size [T_traj, batchsize]
        - bootstrap_values (:obj:`torch.Tensor`): estimation of the state value at step 0 to T,
          of size [T_traj+1, batchsize]
    Returns:
        - result (:obj:`torch.Tensor`): Computed V-trace advantage, of size [T_traj, batchsize]
    """
    if not isinstance(gammas, torch.Tensor):
        gammas = gammas * torch.ones_like(rewards)
    if not isinstance(lambda_, torch.Tensor):
        lambda_ = lambda_ * torch.ones_like(rewards)
    deltas = clipped_rhos * (rewards + gammas * bootstrap_values[1:, :] - bootstrap_values[:-1, :])  # from 0 to T-1
    vtrace_val = torch.empty_like(bootstrap_values)  # from 0 to T
    vtrace_val[-1, :] = bootstrap_values[-1, :]
    for t in reversed(range(rewards.size()[0])):
        vtrace_val[t, :] = bootstrap_values[t, :] + deltas[t, :] \
                           + gammas[t, :] * lambda_[t, :] * clipped_cs[t, :] * \
                           (vtrace_val[t + 1, :] - bootstrap_values[t + 1, :])
    if clipped_pg_rhos is None:
        clipped_pg_rhos = clipped_rhos
    advantages = clipped_pg_rhos * (rewards + gammas * vtrace_val[1:] - bootstrap_values[:-1])
    return advantages
