import os.path as osp
from collections import namedtuple, OrderedDict, defaultdict
from functools import reduce
import torch
import torch.nn as nn
from collections.abc import Sequence, Mapping

from .encoder import Encoder
from .policy import Policy
from .value import ValueBaseline
from ctools.model import ValueActorCriticBase
from ctools.utils import read_config, deep_merge_dicts
from distar.envs import AlphaStarEnv

alphastar_model_default_config = read_config(osp.join(osp.dirname(__file__), "actor_critic_default_config.yaml"))


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


class AlphaStarActorCritic(ValueActorCriticBase):
    EvalInput = namedtuple(
        'EvalInput', ['map_size', 'entity_raw', 'scalar_info', 'spatial_info', 'entity_info', 'prev_state']
    )
    EvalOutput = namedtuple('EvalOutput', ['actions', 'logits'])
    MimicOutput = namedtuple('MimicOutput', ['logits', 'next_state'])
    StepInput = namedtuple('StepInput', ['home', 'away'])
    StepOutput = namedtuple('StepOutput', ['actions', 'logits', 'baselines'])
    CriticInput = namedtuple(
        'CriticInput', [
            'lstm_output_home', 'embeddings_entity_away',
            'embeddings_spatial_away', 'baseline_feature_home', 'baseline_feature_away',
            'score_embedding_home', 'score_embedding_away', 'cum_stat_home', 'immediate_cum_stat_home',
            'immediate_cum_stat_away'
        ]
    )
    CriticOutput = namedtuple('CriticOutput', ['winloss', 'build_order', 'built_unit', 'effect', 'upgrade', 'battle'])

    def __init__(self, model_config=None):
        super(AlphaStarActorCritic, self).__init__()
        cfg = deep_merge_dicts(alphastar_model_default_config["model"], model_config)
        self.cfg = self._merge_input_dim(cfg)
        self.encoder = Encoder(self.cfg.encoder)
        self.policy = Policy(self.cfg.policy)
        self.only_update_baseline = cfg.get('only_update_baseline', False)
        if self.cfg.use_value_network:
            self.value_networks = nn.ModuleDict()
            self.value_cum_stat_keys = OrderedDict()
            for k, v in self.cfg.value.items():
                if k in self.cfg.enable_baselines:
                    # creating a ValueBaseline network for each baseline, to be used in _critic_forward
                    self.value_networks[v.name] = ValueBaseline(v.param)
                    # name of needed cumulative stat items
                    self.value_cum_stat_keys[v.name] = v.cum_stat_keys

        self.freeze_module(self.cfg.freeze_targets)

    def _merge_input_dim(self, cfg):
        env_info = AlphaStarEnv({}).info()
        cfg.encoder.obs_encoder.entity_encoder.input_dim = env_info.obs_space['entity'].shape[-1]
        cfg.encoder.obs_encoder.spatial_encoder.input_dim = env_info.obs_space['spatial'].shape[
            0] + cfg.encoder.scatter.output_dim
        handle = cfg.encoder.obs_encoder.scalar_encoder.module
        for k in handle.keys():
            handle[k].input_dim = env_info.obs_space['scalar'].shape[k]
        cfg.encoder.score_cumulative.input_dim = env_info.obs_space['scalar'].shape['score_cumulative']
        return cfg

    def freeze_module(self, freeze_targets=None):
        """
        Note:
            must be called after the model initialization, before the model forward
        """
        if freeze_targets is None:
            # if freeze_targets is not provided, try to use self.freeze_targets
            if self.freeze_targets is None:
                raise Exception("not provided arguments(freeze_targets)")
            else:
                freeze_targets = self.freeze_targets
        else:
            # if freeze_targets is provided, update self.freeze_targets for next usage
            self.freeze_targets = freeze_targets

        def get_submodule(name):
            part = name.split('.')
            module = self
            for p in part:
                module = getattr(module, p)
            return module

        for name in freeze_targets:
            module = get_submodule(name)
            module.eval()
            for m in module.parameters():
                m.requires_grad_(False)

    # overwrite
    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, 'freeze_targets'):
            self.freeze_module()

    # overwrite
    def mimic_single(self, inputs, **kwargs):
        lstm_output, next_state, entity_embeddings, map_skip, scalar_context, spatial_info, _, _, _, _, _, _= \
            self.encoder(
            inputs
        )
        policy_inputs = self.policy.MimicInput(
            inputs['actions'], inputs['entity_raw'], inputs['scalar_info']['available_actions'], lstm_output,
            entity_embeddings, map_skip, scalar_context, spatial_info, inputs['entity_num'],
            inputs['selected_units_num']
        )
        logits = self.policy(policy_inputs, mode='mimic')
        return {'policy_outputs': logits, 'next_state': next_state}

    # overwrite
    def mimic(self, inputs, **kwargs):
        self.traj = [len(b['spatial_info']) for b in inputs]
        self.batch_size = len(inputs[0]['spatial_info'])
        prev_state = inputs[0].pop('prev_state')
        end_idx = [[i for i in inputs[j]['end_index']] for j in range(len(inputs))]
        inputs = self._merge_traj(inputs)
        # encoder
        embedded_entity, embedded_spatial, embedded_scalar, scalar_context, baseline_feature,\
            cum_stat, entity_embeddings, map_skip = self.encoder.encode_parallel_forward(inputs)
        embedded_entity, embedded_spatial, embedded_scalar = [
            self._split_traj(t) for t in [embedded_entity, embedded_spatial, embedded_scalar]
        ]
        # lstm
        lstm_output = []
        for idx, embedding in enumerate(zip(embedded_entity, embedded_spatial, embedded_scalar)):
            active_state = [i for i in range(self.batch_size) if i not in end_idx[idx]]
            tmp_state = [prev_state[i] for i in active_state]
            tmp_output, tmp_state = self.encoder.core_lstm(embedding[0], embedding[1], embedding[2], tmp_state)
            for _idx, active_idx in enumerate(active_state):
                prev_state[active_idx] = tmp_state[_idx]
            lstm_output.append(tmp_output.squeeze(0))
        next_state = prev_state
        lstm_output = self._merge_traj(lstm_output)
        # head
        policy_inputs = self.policy.MimicInput(
            inputs['actions'], inputs['entity_raw'], inputs['scalar_info']['available_actions'], lstm_output,
            entity_embeddings, map_skip, scalar_context, inputs['spatial_info']
        )
        logits = self.policy(policy_inputs, mode='mimic')
        return self.MimicOutput(logits, next_state)

    def _merge_traj(self, data):

        def merge(t):
            if isinstance(t[0], torch.Tensor):
                # t = torch.stack(t, dim=0)
                # return t.reshape(-1, *t.shape[2:])
                t = torch.cat(t, dim=0)
                return t
            elif isinstance(t[0], list):
                return reduce(lambda x, y: x + y, t)
            elif isinstance(t[0], dict):
                return {k: merge([m[k] for m in t]) for k in t[0].keys()}
            else:
                raise TypeError(type(t[0]))

        if isinstance(data, torch.Tensor):
            return data.reshape(-1, *data.shape[2:])
        else:
            return merge(data)

    def _split_traj(self, data):
        assert isinstance(data, torch.Tensor)
        ret = [d.unsqueeze(0) for d in torch.split(data, self.traj, 0)]
        assert len(ret) == len(self.traj), 'resume data length must equal to original data'
        return ret

    # overwrite
    def compute_action(self, inputs, **kwargs):
        """
            Overview: agent evaluate(only actor)
            Note:
                batch size = 1
            Overview: forward for agent evaluate (only actor is evaluated). batch size must be 1
            Inputs:
                - inputs: EvalInput namedtuple with following fields
                    - map_size
                    - entity_raw
                    - scalar_info
                    - spatial_info
                    - entity_info
                    - prev_state
            Output:
                - EvalOutput named dict
        """
        if self.cfg.use_value_network:
            assert 'away_inputs' in inputs.keys()
            away_inputs = inputs['away_inputs']
            away_inputs['prev_state'] = inputs['away_hidden_state']
            lstm_output_away, next_state_away, entity_embeddings_away, map_skip_away, scalar_context_away, \
            spatial_info_away, baseline_feature_away, cum_stat_away, score_embedding_away, \
            embedded_spatial_away, embedded_entity_away, immediate_cum_stat_away \
                = self.encoder(
                away_inputs
            )
            lstm_output, next_state, entity_embeddings, map_skip, scalar_context, spatial_info, \
            baseline_feature, cum_stat, score_embedding, embedded_spatial, embedded_entity, \
            immediate_cum_stat = self.encoder(
                inputs
            )

            embedded_entity_away = embedded_entity_away.reshape(-1, embedded_entity_away.shape[-1])
            embedded_spatial_away = embedded_spatial_away.reshape(-1, embedded_spatial_away.shape[-1])

            critic_inputs = self.CriticInput(
                lstm_output, embedded_entity_away,
                embedded_spatial_away, baseline_feature, baseline_feature_away, score_embedding,
                score_embedding_away, cum_stat, immediate_cum_stat, immediate_cum_stat_away
            )
            baselines = self._critic_forward(critic_inputs)
        else:
            lstm_output, next_state, entity_embeddings, map_skip, scalar_context, spatial_info, _, _, _, _, _, _ \
                = self.encoder(
                inputs
            )
            baselines = None

        policy_inputs = self.policy.EvaluateInput(
            inputs['entity_raw'],
            inputs['scalar_info']['available_actions'],
            lstm_output,
            entity_embeddings,
            map_skip,
            scalar_context,
            spatial_info,
            inputs['entity_num'],
        )
        actions, logits = self.policy(policy_inputs, mode='evaluate', **kwargs)

        return self.EvalOutput(actions, logits), next_state, baselines

    # overwrite
    def step(self, inputs, **kwargs):
        """
            Overview: forward for training (actor and critic)
            Inputs:
                - inputs: StepInput namedtuple with observations
                    - obs_home: observation from my self as EvalInput
                    - obs_away: observation from the rival as EvalInput
            Outputs:
                - ret: StepOutput namedtuple containing
                    - actions: output from the model
                    - baselines: critic values
                    - next_state_home
                    - next_state_away
        """
        # encoder(home and away)
        prev_state = inputs['prev_state']
        inputs['obs_home']['prev_state'] = [p['home'] for p in prev_state]
        inputs['obs_away']['prev_state'] = [p['away'] for p in prev_state]

        lstm_output_home, \
        next_state_home, \
        entity_embeddings, \
        map_skip, \
        scalar_context, \
        spatial_info, \
        baseline_feature_home, \
        cum_stat_home, \
        score_embedding_home = self.encoder(
            inputs['obs_home']
        )
        lstm_output_away, next_state_away, _, _, _, _, baseline_feature_away, cum_stat_away, \
        score_embedding_away = self.encoder(inputs['obs_away'])

        # value
        critic_inputs = self.CriticInput(
            lstm_output_home, lstm_output_away, baseline_feature_home, baseline_feature_away, score_embedding_home,
            score_embedding_away, cum_stat_home, cum_stat_away
        )
        baselines = self._critic_forward(critic_inputs)

        # policy
        policy_inputs = self.policy.EvaluateInput(
            inputs['obs_home']['entity_raw'], inputs['obs_home']['scalar_info']['available_actions'], lstm_output_home,
            entity_embeddings, map_skip, scalar_context, spatial_info
        )
        actions, logits = self.policy(policy_inputs, mode='evaluate', **kwargs)
        next_state = [{'home': h, 'away': a} for h, a in zip(next_state_home, next_state_away)]
        return self.StepOutput(actions, logits, baselines), next_state

    def compute_action_value(self, inputs, **kwargs):
        batch_size = inputs['batch_size']
        traj_len = inputs['traj_len']
        prev_state = inputs['prev_state']
        prev_state = [p['home'] for p in prev_state]
        # merge obs_home and obs_away together, add last obs, so trajectory length added one more
        embedded_entity, embedded_spatial, embedded_scalar, scalar_context, baseline_feature, \
        cum_stat, entity_embeddings, map_skip, score_embedding, immediate_cum_stat = self.encoder.encode_parallel_forward(inputs)
        embeddings = [embedded_entity, embedded_spatial, embedded_scalar]
        embeddings_entity_away = embeddings[0][(traj_len + 1) * batch_size:]
        embeddings_spatial_away = embeddings[1][(traj_len + 1) * batch_size:]
        embeddings_home = [e[:(traj_len + 1) * batch_size].view(-1, batch_size, e.shape[-1]) for e in embeddings]
        # go through core lstm
        lstm_output_home, next_state = self.encoder.core_lstm(*embeddings_home, prev_state)  # traj + 1, 2*b, -1
        # split embeddings to home and away
        lstm_output_home = lstm_output_home.reshape(-1, lstm_output_home.shape[-1])  # (traj + 1) * b, -1
        baseline_feature_home, baseline_feature_away = torch.chunk(baseline_feature, 2, dim=0)
        score_embedding_home, score_embedding_away = torch.chunk(score_embedding, 2, dim=0)
        cum_stat_home, cum_stat_away = dict(), dict()
        for k, v in cum_stat.items():
            cum_stat_home[k], cum_stat_away[k] = torch.chunk(v, 2, dim=0)
        immediate_cum_stat_home, immediate_cum_stat_away = dict(), dict()
        for k, v in immediate_cum_stat.items():
            immediate_cum_stat_home[k], immediate_cum_stat_away[k] = torch.chunk(v, 2, dim=0)
        # value
        critic_input = [lstm_output_home, embeddings_entity_away, embeddings_spatial_away, baseline_feature_home,
                        baseline_feature_away, score_embedding_home, score_embedding_away, cum_stat_home,
                        immediate_cum_stat_home, immediate_cum_stat_away]
        if self.only_update_baseline:
            critic_input = detach_grad(critic_input)
        critic_inputs = self.CriticInput(*critic_input)
        baselines = self._critic_forward(critic_inputs, parallel=True, traj_len=traj_len + 1, batch_size=batch_size)
        # get home embedding for policy
        home_size = traj_len * batch_size
        map_skip = [i[:home_size] for i in map_skip]
        actions = {k: v[:home_size] for k, v in inputs['actions'].items()}
        entity_raw = {k: v[:home_size] for k, v in inputs['entity_raw'].items()}
        entity_num = inputs['entity_num'][:home_size]
        max_entity_num = entity_num.max()
        selected_units_num = inputs['selected_units_num'][:home_size]
        entity_embeddings_home = entity_embeddings[:home_size, :max_entity_num]
        policy_inputs = self.policy.MimicInput(
            actions, entity_raw, inputs['scalar_info']['available_actions'][:home_size], lstm_output_home[:home_size],
            entity_embeddings_home, map_skip, scalar_context[:home_size], inputs['spatial_info'][:home_size],
            entity_num, selected_units_num
        )
        logits = self.policy(policy_inputs, mode='mimic')
        mid = len(next_state) // 2
        next_state = list(zip(*[next_state[:mid], next_state[mid:]]))
        next_state = [{'home': n[0], 'away': n[1]} for n in next_state]
        return {'policy_outputs': logits, 'baselines': baselines, 'next_state': next_state}

    # overwrite
    def _critic_forward(self, inputs, parallel=False, traj_len=0, batch_size=0):
        """
        Overview: Evaluate value network on each baseline
        """

        def select_item(data, key):
            # Input: data:dict key:list Returns: ret:list
            # filter data and return a list of values with keys in key
            ret = []
            for k, v in data.items():
                if k in key:
                    ret.append(v)
            return ret

        cum_stat_home, immediate_cum_stat_home = inputs.cum_stat_home, inputs.immediate_cum_stat_home
        immediate_cum_stat_away = inputs.immediate_cum_stat_away
        # 'lstm_output_home', 'lstm_output_away', 'baseline_feature_home', 'baseline_feature_away'
        # are torch.Tensors and are shared across all baselines
        same_part = torch.cat(inputs[:7], dim=1)
        ret = {k: None for k in self.CriticOutput._fields}
        for (name_n, m), (name_c, key) in zip(self.value_networks.items(), self.value_cum_stat_keys.items()):
            assert name_n == name_c
            cum_stat_home_subset = select_item(cum_stat_home, key)
            immediate_cum_stat_home_subset = select_item(immediate_cum_stat_home, key)
            immediate_cum_stat_away_subset = select_item(immediate_cum_stat_away, key)
            inputs = torch.cat(
                [same_part] + cum_stat_home_subset + immediate_cum_stat_home_subset + immediate_cum_stat_away_subset,
                dim=1)
            # apply the value network to inputs
            ret[name_n] = m(inputs)
            if parallel:
                ret[name_n] = ret[name_n].view(traj_len, batch_size)
        return self.CriticOutput(**ret)
