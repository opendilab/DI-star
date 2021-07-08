from collections.abc import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from ctools.torch_utils import fc_block, build_activation
from ctools.torch_utils.network.rnn import sequence_mask
from distar.model.alphastar.module_utils import Transformer


class EntityEncoder(nn.Module):
    r'''
    B=batch size EN=any number of entities ID=input_dim OS=output_size=256
     (B*EN*ID)  (EN'*OS)          (EN'*OS)          (EN'*OS)           (B*EN*OS)
    x -> combine -> Transformer ->  act ->  entity_fc  -> split ->   entity_embeddings
          batch                         |      (B*EN*OS)   (B*OS)        (B*OS)
                                        \->  split ->  mean -> embed_fc -> embedded_entity
    '''

    def __init__(self, cfg):
        super(EntityEncoder, self).__init__()
        self.act = build_activation(cfg.activation)
        self.transformer = Transformer(
            input_dim=cfg.input_dim,
            head_dim=cfg.head_dim,
            hidden_dim=cfg.hidden_dim,
            output_dim=cfg.output_dim,
            head_num=cfg.head_num,
            mlp_num=cfg.mlp_num,
            layer_num=cfg.layer_num,
            dropout_ratio=cfg.dropout_ratio,
            activation=self.act,
            ln_type=cfg.ln_type
        )
        self.entity_fc = fc_block(cfg.output_dim, cfg.output_dim, activation=self.act)
        self.embed_fc = fc_block(cfg.output_dim, cfg.output_dim, activation=self.act)

    def forward(self, x, entity_num):
        '''
        Input:
            x: list(tuple) of batch_size * Tensor of size [entity_num, input_dim]
               entity_num may differ for each tensor in the list
               See detailed-architecture.txt line 19-64 for more detail
               about the fields in the dim 2
            entity:num: valid entity num
        Output:
            entity_embeddings: tuple(len=batch_size)->element: torch.Tensor, shape(entity_num_b, output_dim)
            embedded_entity: Tensor of size [batch_size, output_dim]
        '''
        mask = sequence_mask(entity_num)
        x = self.transformer(x, mask=mask)
        x = self.act(x)
        entity_embeddings = self.entity_fc(x)
        x_mask = x * mask.unsqueeze(dim=2)
        # embedded_entity = x_mask.sum(dim=1) / 512
        embedded_entity = x_mask.sum(dim=1) / entity_num
        embedded_entity = self.embed_fc(embedded_entity)
        return entity_embeddings, embedded_entity, mask

