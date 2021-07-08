import collections

import torch
import torch.nn as nn

from ctools.torch_utils import fc_block, build_activation
from .core import CoreLstm
from .obs_encoder import ScalarEncoder, SpatialEncoder, EntityEncoder


def build_obs_encoder(name):
    obs_encoder_dict = {
        'scalar_encoder': ScalarEncoder,
        'spatial_encoder': SpatialEncoder,
        'entity_encoder': EntityEncoder,
    }
    return obs_encoder_dict[name]


class Encoder(nn.Module):

    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.cfg = cfg
        self.encoder = nn.ModuleDict()
        for item in cfg.obs_encoder.encoder_names:
            self.encoder[item] = build_obs_encoder(item)(cfg.obs_encoder[item])
        self.core_lstm = CoreLstm(cfg.core_lstm)

        self.scatter_project = fc_block(cfg.scatter.input_dim, cfg.scatter.output_dim)
        self.scatter_dim = cfg.scatter.output_dim
        self.use_score_cumulative = cfg.obs_encoder.use_score_cumulative
        if self.use_score_cumulative:
            self.score_cumulative_encoder = fc_block(
                self.cfg.score_cumulative.input_dim,
                self.cfg.score_cumulative.output_dim,
                activation=build_activation(self.cfg.score_cumulative.activation)
            )
        self.scatter_type = cfg.scatter.get('scatter_type', 'cover')

    def _scatter_connection(self, spatial_info, entity_embeddings, entity_raw, entity_mask):
        project_embeddings = self.scatter_project(entity_embeddings)  # b, n, scatter_dim
        B, _, H, W = spatial_info.shape
        device = spatial_info.device

        entity_num = entity_raw['location'].shape[1]
        index = entity_raw['location'].view(-1, 2)
        bias = torch.arange(B).unsqueeze(1).repeat(1, entity_num).view(-1).to(device)
        bias *= H * W
        index[:, 0].clamp_(0, H - 1)
        index[:, 1].clamp_(0, W - 1)
        index = index[:, 0] * W + index[:, 1]
        index += bias
        index = index.repeat(self.scatter_dim, 1)
        # flat scatter map and project embeddings
        scatter_map = torch.zeros(self.scatter_dim, B * H * W, device=device)
        project_embeddings *= entity_mask.unsqueeze(dim=2)
        project_embeddings = project_embeddings.view(-1, self.scatter_dim).permute(1, 0)
        if self.scatter_type == 'cover':
            scatter_map.scatter_(dim=1, index=index, src=project_embeddings)
        elif self.scatter_type == 'add':
            scatter_map.scatter_add_(dim=1, index=index, src=project_embeddings)
        else:
            raise NotImplementedError
        scatter_map = scatter_map.reshape(self.scatter_dim, B, H, W)
        scatter_map = scatter_map.permute(1, 0, 2, 3)
        return torch.cat([spatial_info, scatter_map], dim=1)

    def forward(self, inputs):
        '''
        Arguments:
            - inputs:
            dict with field:
                - scalar_info
                - spatial_info
                - entity_raw
                - entity_info
                - map_size
                - prev_state
                - score_cumulative
        Outputs:
            - lstm_output: The LSTM state for the next step. Tensor of size [seq_len, batch_size, hidden_size]
            - next_state: The LSTM state for the next step.
              As list [H,C], H and C are of size [num_layers, batch_size, hidden_size]
            - entity_embeddings: The embedding of each entity. Tensor of size [batch_size, entity_num, output_dim]
            - map_skip
            - scalar_context
            - baseline_feature
            - cum_stat: OrderedDict of various cumulative_statistics
            - socre_embedding: score cumulative embedding for baseline
        '''
        embedded_scalar, scalar_context, baseline_feature, cum_stat, immediate_cum_stat= self.encoder['scalar_encoder'](
            inputs['scalar_info']
        )
        entity_embeddings, embedded_entity, entity_mask = self.encoder['entity_encoder'](
            inputs['entity_info'], inputs['entity_num']
        )
        spatial_input = self._scatter_connection(
            inputs['spatial_info'], entity_embeddings, inputs['entity_raw'], entity_mask
        )
        embedded_spatial, map_skip = self.encoder['spatial_encoder'](spatial_input, inputs['map_size'])

        embedded_entity, embedded_spatial, embedded_scalar = (
            embedded_entity.unsqueeze(0), embedded_spatial.unsqueeze(0), embedded_scalar.unsqueeze(0)
        )
        lstm_output, next_state = self.core_lstm(
            embedded_entity, embedded_spatial, embedded_scalar, inputs['prev_state']
        )
        lstm_output = lstm_output.squeeze(0)
        if self.use_score_cumulative:
            score_embedding = self.score_cumulative_encoder(inputs['scalar_info']['score_cumulative'])
        else:
            score_embedding = None  # placeholder
        return lstm_output, next_state, entity_embeddings, map_skip, scalar_context, inputs[
            'spatial_info'], baseline_feature, cum_stat, score_embedding, embedded_spatial, embedded_entity, immediate_cum_stat

    def encode_parallel_forward(self, inputs):
        embedded_scalar, scalar_context, baseline_feature, cum_stat, immediate_cum_stat= self.encoder['scalar_encoder'](
            inputs['scalar_info']
        )
        entity_embeddings, embedded_entity, entity_mask = self.encoder['entity_encoder'](
            inputs['entity_info'], inputs['entity_num']
        )
        spatial_input = self._scatter_connection(
            inputs['spatial_info'], entity_embeddings, inputs['entity_raw'], entity_mask
        )
        embedded_spatial, map_skip = self.encoder['spatial_encoder'](spatial_input, inputs['map_size'])
        if self.use_score_cumulative:
            score_embedding = self.score_cumulative_encoder(inputs['scalar_info']['score_cumulative'])
        else:
            score_embedding = None  # placeholder
        return [
            embedded_entity, embedded_spatial, embedded_scalar, scalar_context, baseline_feature, cum_stat,
            entity_embeddings, map_skip, score_embedding, immediate_cum_stat
        ]
