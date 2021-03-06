import collections

import torch
import torch.nn as nn
from typing import Dict
from torch import Tensor

from distar.ctools.torch_utils import fc_block

from .obs_encoder import ScalarEncoder, SpatialEncoder, EntityEncoder
from .lstm import script_lnlstm
from .module_utils import scatter_connection


class Encoder(nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.whole_cfg = cfg
        self.cfg = cfg.model.encoder
        self.encoder = nn.ModuleDict()
        self.scalar_encoder = ScalarEncoder(self.whole_cfg)
        self.spatial_encoder = SpatialEncoder(self.whole_cfg)
        self.entity_encoder = EntityEncoder(self.whole_cfg)
        self.scatter_project = fc_block(self.cfg.scatter.input_dim, self.cfg.scatter.output_dim, activation='relu')
        self.scatter_dim = self.cfg.scatter.output_dim
        self.scatter_type = self.cfg.scatter.get('scatter_type', 'cover')

    def forward(self, spatial_info:Tensor, entity_info: Dict[str, Tensor], scalar_info: Dict[str, Tensor], entity_num: Tensor):
        embedded_scalar, scalar_context, baseline_feature = self.scalar_encoder(
            scalar_info
        )
        entity_embeddings, embedded_entity, entity_mask = self.entity_encoder(
            entity_info, entity_num
        )
        entity_location = torch.cat([entity_info['x'].unsqueeze(dim=-1), entity_info['y'].unsqueeze(dim=-1)], dim=-1)
        shape = spatial_info['height_map'].shape
        project_embeddings = self.scatter_project(entity_embeddings)
        project_embeddings = project_embeddings * entity_mask.unsqueeze(dim=2)

        scatter_map = scatter_connection(
            shape, project_embeddings, entity_location, self.scatter_dim, self.scatter_type
        )
        embedded_spatial, map_skip = self.spatial_encoder(spatial_info, scatter_map)
        lstm_input = torch.cat([embedded_scalar, embedded_entity, embedded_spatial], dim=-1)
        return lstm_input, scalar_context, baseline_feature, entity_embeddings, map_skip
