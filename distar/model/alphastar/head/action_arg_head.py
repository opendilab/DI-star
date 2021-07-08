'''
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. Implementation for action_type_head, including basic processes.
    2. Implementation for delay_head, including basic processes.
    3. Implementation for queue_type_head, including basic processes.
    4. Implementation for selected_units_type_head, including basic processes.
    5. Implementation for target_unit_head, including basic processes.
    6. Implementation for location_head, including basic processes.
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ctools.torch_utils import fc_block, conv2d_block, deconv2d_block, build_activation, one_hot, get_lstm, \
    ResBlock, NearestUpsample, BilinearUpsample, binary_encode, SoftArgmax
from ctools.torch_utils import CategoricalPdPytorch
from ctools.torch_utils.network.rnn import sequence_mask
from distar.model.alphastar.module_utils import GatedResBlock, FiLMedResBlock


class DelayHead(nn.Module):
    '''
        Overview: The delay head uses autoregressive_embedding to get delay_logits and delay.
        Interface: __init__, forward
    '''

    def __init__(self, cfg):
        '''
            Overview: initialize architect.
            Arguments:
                - cfg (:obj:`dict`): head architecture definition
        '''
        super(DelayHead, self).__init__()
        self.act = build_activation(cfg.activation)
        self.fc1 = fc_block(cfg.input_dim, cfg.decode_dim, activation=self.act, norm_type=None)
        self.fc2 = fc_block(cfg.decode_dim, cfg.decode_dim, activation=self.act, norm_type=None)
        self.fc3 = fc_block(cfg.decode_dim, cfg.delay_dim, activation=None, norm_type=None)  # regression
        self.embed_fc1 = fc_block(cfg.delay_dim, cfg.delay_map_dim, activation=self.act, norm_type=None)
        self.embed_fc2 = fc_block(cfg.delay_map_dim, cfg.input_dim, activation=None, norm_type=None)
        self.pd = CategoricalPdPytorch
        self.delay_dim = cfg.delay_dim

    def forward(self, embedding, delay=None):
        '''
            Overview: This head uses autoregressive_embedding to get delay_logits. Autoregressive_embedding
                      is decoded using a 2-layer (each with size 256) linear network with ReLUs, before being
                      embedded into delay_logits that has size 128 (one for each possible requested delay in
                      game steps). Then delay is sampled from delay_logits using a multinomial, though unlike
                      all other arguments, no temperature is applied to delay_logits before sampling.
                      Delay is projected to a 1D tensor of size 1024 through a 2-layer (each with size 256)
                      linear network with ReLUs, and added to autoregressive_embedding.
            Arguments:
                - embedding (:obj:`tensor`): autoregressive_embedding
                - delay (:obj:`tensor or None`): when SL training, the caller indicates delay value to calculate
                    embedding
            Returns:
                - (:obj`tensor`): delay for calculation loss, shape(B, delay_dim), dtype(torch.float)
                - (:obj`tensor`): delay action, shape(B, ), dtype(torch.long)
                - (:obj`tensor`): autoregressive_embedding that combines information from lstm_output
                                  and all previous sampled arguments, shape(B, input_dim)
        '''
        x = self.fc1(embedding)
        x = self.fc2(x)
        x = self.fc3(x)
        if delay is None:
            p = F.softmax(x, dim=1)
            handle = self.pd(p)
            delay = handle.sample()

        delay_encode = one_hot(delay, self.delay_dim)
        embedding_delay = self.embed_fc1(delay_encode)
        embedding_delay = self.embed_fc2(embedding_delay)  # get autoregressive_embedding

        return x, delay, embedding + embedding_delay


class QueuedHead(nn.Module):
    '''
        Overview: The queue head uses autoregressive_embedding, action_type and entity_embeddings to get
                  queued_logits and sampled queued.
        Interface: __init__, forward
    '''

    def __init__(self, cfg):
        '''
            Overview: initialize architect.
            Arguments:
                - cfg (:obj:`dict`): head architecture definition
        '''
        super(QueuedHead, self).__init__()
        self.act = build_activation(cfg.activation)
        # to get queued logits
        self.fc1 = fc_block(cfg.input_dim, cfg.decode_dim, activation=self.act, norm_type=None)
        self.fc2 = fc_block(cfg.decode_dim, cfg.decode_dim, activation=self.act, norm_type=None)
        self.fc3 = fc_block(cfg.decode_dim, cfg.queued_dim, activation=None, norm_type=None)

        # to get autoregressive_embedding
        self.embed_fc1 = fc_block(cfg.queued_dim, cfg.queued_map_dim, activation=self.act, norm_type=None)
        self.embed_fc2 = fc_block(cfg.queued_map_dim, cfg.input_dim, activation=None, norm_type=None)
        self.pd = CategoricalPdPytorch

        self.queued_dim = cfg.queued_dim

    def forward(self, embedding, temperature=1.0, queued=None):
        '''
            Overview: This head uses autoregressive_embedding to get queued_logits. Queued Head is similar to
                      the delay head except a temperature of 0.8 is applied to the logits before sampling, the
                      size of queued_logits is 2 (for queueing and not queueing), and the projected queued is
                      not added to autoregressive_embedding if queuing is not possible for the chosen action_type.
            Arguments:
                - embedding (:obj:`tensor`): autoregressive_embedding
                - temperature (:obj:`float`): temperature
                - queued (obj:`tensor or None`): when SL training, the caller indicates queued to calculate embedding
            Returns:
                - (:obj`tensor`): queued_logits corresponding to the probabilities of queueing and not queueing
                - (:obj`tensor`): queued that whether or no to queue this action
                - (:obj`tensor`): autoregressive_embedding that combines information from lstm_output
                                  and all previous sampled arguments.
        '''
        x = self.fc1(embedding)
        x = self.fc2(x)
        x = self.fc3(x)
        x = x.div(0.8)
        if queued is None:
            p = F.softmax(x, dim=1)
            handle = self.pd(p)
            queued = handle.sample()

        queued_one_hot = one_hot(queued, self.queued_dim)
        embedding_queued = self.embed_fc1(queued_one_hot)
        embedding_queued = self.embed_fc2(embedding_queued)  # get autoregressive_embedding

        return x, queued, embedding + embedding_queued


class SelectedUnitsHead(nn.Module):
    '''
        Overview: The selected units head uses autoregressive_embedding, action_type and entity_embeddings to get
                  units_logits and sampled units.
        Interface: __init__, forward
    '''

    def __init__(self, cfg):
        '''
            Overview: initialize architect.
            Arguments:
                - cfg (:obj:`dict`): head architecture definition
        '''
        super(SelectedUnitsHead, self).__init__()
        self.act = build_activation(cfg.activation)
        self.key_fc = fc_block(cfg.entity_embedding_dim, cfg.key_dim, activation=None, norm_type=None)
        # determines which entity types can accept action_type
        self.func_fc = fc_block(cfg.unit_type_dim, cfg.func_dim, activation=self.act, norm_type=None)
        self.fc1 = fc_block(cfg.input_dim, cfg.func_dim, activation=None, norm_type=None)
        self.fc2 = fc_block(cfg.func_dim, cfg.key_dim, activation=None, norm_type=None)
        self.embed_fc = fc_block(cfg.key_dim, cfg.input_dim, activation=None, norm_type=None)
        self.lstm = get_lstm(cfg.lstm_type, cfg.key_dim, cfg.hidden_dim, cfg.num_layers, norm_type=cfg.lstm_norm_type)

        self.max_entity_num = cfg.max_entity_num
        self.key_dim = cfg.key_dim
        self.use_mask = cfg.use_mask
        self.pd = CategoricalPdPytorch

        self.end_embedding = torch.nn.Parameter(torch.FloatTensor(1, self.key_dim))
        stdv = 1. / math.sqrt(self.end_embedding.size(1))
        self.end_embedding.data.uniform_(-stdv, stdv)
        self.units_reorder = cfg.get('units_reorder', False)

    def _get_key_mask(self, entity_embedding, entity_num):
        '''
            Overview: computes a key corresponding to each entity by feeding entity_embeddings through
                      a 1D convolution with 32 channels and kernel size 1.
                      pad with the maximum entity number in a batch and pack into tensor
            Arguments:
                - entity_embedding (:obj:`tensor`): entity embeddings
                - entity_num (:obj:'tensor'): entity numbers
            Returns:
                - key (:obj`tensor`): entity embeddings, which are the keys in the next match
                - mask (:obj:`tensor`): entity mask
                - key_embeddings (:obj`tensor`): embedded keys should be added to autoregressive embedding later
        '''

        bs = entity_embedding.shape[0]
        padding_end = torch.zeros(1, self.end_embedding.shape[1]).repeat(bs, 1,
                                                                         1).to(entity_embedding.device)  # b, 1, c
        key = self.key_fc(entity_embedding)  # b, n, c
        key = torch.cat([key, padding_end], dim=1)
        end_embeddings = torch.ones(key.shape, dtype=key.dtype, device=key.device) * self.end_embedding.squeeze(dim=0)
        flag = torch.ones(key.shape[:2], dtype=torch.bool, device=key.device).unsqueeze(dim=2)
        flag[torch.arange(bs), entity_num.squeeze(dim=1)] = 0
        end_embeddings = end_embeddings * ~flag
        key = key * flag
        key = key + end_embeddings
        # key_reduce = torch.div(key, 64)
        key_reduce = torch.div(key, entity_num.unsqueeze(1))
        key_embeddings = self.embed_fc(key_reduce)
        new_entity_num = entity_num + 1  # add end entity
        mask = sequence_mask(new_entity_num)
        return key, mask, key_embeddings

    def _get_query(self, embedding, func_embed):
        '''
            Overview: passes autoregressive_embedding through a linear of size 256, adds func_embed, and
                      passes the combination through a ReLU and a linear of size 32.
            Arguments:
                - embedding (:obj:`tensor`): autoregressive_embedding
                - func_embed (:obj:`tensor`): embedding derived from available_unit_type_mask
            Returns:
                - (:obj`tensor`): result
        '''
        x = self.fc1(embedding)
        x = self.fc2(F.relu(x + func_embed))
        return x

    def _get_pred_with_logit(self, logit, temperature):
        p = F.softmax(logit, dim=-1)
        handle = self.pd(p)
        units = handle.sample()
        return units

    def _query(
        self, key, entity_num, autoregressive_embedding, func_embed, logits_mask, temperature, selected_units,
        key_embeddings, selected_units_num
    ):
        ae = autoregressive_embedding
        logits, results = None, None  # placeholder
        if selected_units is not None:  # train
            bs = selected_units.shape[0]
            seq_len = selected_units_num.max()
            state = None
            queries = []
            entity_num = entity_num.squeeze(dim=1)  # n, 1 -> n
            selected_mask = sequence_mask(selected_units_num)  # b, s
            logits_mask = logits_mask.repeat(seq_len, 1, 1)  # b, n -> s, b, n
            logits_mask[0, torch.arange(bs), entity_num] = 0  # end flag is not available at first selection
            for i in range(seq_len):
                if i > 0:
                    ae = ae + key_embeddings[torch.arange(bs),
                                             selected_units[:, i - 1]] * selected_mask[:, i].unsqueeze(1)
                    logits_mask[i] = logits_mask[i - 1]
                    if i == 1:  # enable end flag
                        logits_mask[i, torch.arange(bs), entity_num] = 1
                    logits_mask[i][torch.arange(bs), selected_units[:, i - 1]] = 0  # mask selected units
                lstm_input = self._get_query(ae, func_embed).unsqueeze(0)
                lstm_output, state = self.lstm(lstm_input, state)
                queries.append(lstm_output)
            queries = torch.cat(queries, dim=0).unsqueeze(dim=2)  # s, b, 1, -1
            key = key.unsqueeze(dim=0)  # 1, b, n, -1
            #Q = queries / (1e-6 + torch.norm(queries, 2, -1, keepdim=True))
            #K = key / (1e-6 + torch.norm(key, 2, -1, keepdim=True))
            #query_result = Q * K * 64
            query_result = queries * key
            logits = query_result.sum(dim=3)  # s, b, n
            logits = logits.masked_fill(~logits_mask, -1e9)
            logits = logits.permute(1, 0, 2).contiguous()
        else:  # eval
            bs = ae.shape[0]
            end_flag = torch.zeros(bs).to(ae.device).bool()
            results, logits = [], []
            state = None
            entity_num = entity_num.squeeze(dim=1)
            logits_mask[torch.arange(bs), entity_num] = 0
            selected_units_num = torch.ones(bs, dtype=torch.long, device=ae.device) * 64
            result = None
            for i in range(64):
                if i > 0:
                    ae = ae + key_embeddings[torch.arange(bs), result] * ~end_flag.unsqueeze(dim=1)
                    if i == 1:  # end flag can be selected at second selection
                        logits_mask[torch.arange(bs), entity_num] = 1
                    logits_mask[torch.arange(bs), result] = 0  # mask selected units
                lstm_input = self._get_query(ae, func_embed).unsqueeze(0)
                lstm_output, state = self.lstm(lstm_input, state)
                queries = lstm_output.permute(1, 0, 2)  # b, 1, c
                query_result = queries * key
                #Q = queries / (1e-6 + torch.norm(queries, 2, -1, keepdim=True))
                #K = key / (1e-6 + torch.norm(key, 2, -1, keepdim=True))
                #query_result = Q * K * 64

                step_logits = query_result.sum(dim=2)  # b, n
                step_logits = step_logits.masked_fill(~logits_mask, -1e9)
                step_logits = step_logits.div(0.8)
                result = self._get_pred_with_logit(step_logits, temperature)
                selected_units_num[result == entity_num] = i + 1
                end_flag[result == entity_num] = 1
                results.append(result)
                logits.append(step_logits)
                if end_flag.all():
                    break
            results = torch.stack(results, dim=0)
            results = results.transpose(1, 0).contiguous()
            logits = torch.stack(logits, dim=0)
            logits = logits.transpose(1, 0).contiguous()
        return logits, results, ae, selected_units_num

    def forward(
        self,
        embedding,
        available_unit_type_mask,
        entity_embedding,
        temperature=1.0,
        selected_units=None,
        entity_num=None,
        selected_units_num=None
    ):
        '''
        Input:
            embedding: [batch_size, input_dim(1024)]
            available_unit_type_mask: A mask of which entity types can accept action_type, and this is a
                                      one-hot of this entity type with maximum equal to the number of unit
                                      types. [batch_size, num_unit_type]
            entity_embedding: [batch_size, num_units, entity_embedding_dim(256)]
            selected_units: when SL training, the caller indicates selected_units to calculate embedding
            entity_num: entity embedding was padding to same size, entity_num gives true entity numbers
            selected_units_num: selected units was also padding to same size
        Note:
            num_units can be different among the samples in a batch, if so and batch_size > 1, unis_mask and
            entity_embedding are both list(len=batch_size) and each element is shape [1, ...]
        Output:
            logits: List(batch_size) - List(num_selected_units) - num_units
            units: [batch_size, num_units] 0-1 vector
            new_embedding: [batch_size, input_dim(1024)]
        '''

        key, mask, key_embeddings = self._get_key_mask(entity_embedding, entity_num)
        func_embed = self.func_fc(available_unit_type_mask)
        logits, units, embedding, selected_units_num = self._query(
            key, entity_num, embedding, func_embed, mask, temperature, selected_units, key_embeddings,
            selected_units_num
        )
        return logits, units, embedding, selected_units_num


class TargetUnitHead(nn.Module):
    '''
        Overview: The target unit head uses autoregressive_embedding to get target_unit_logits and target_unit.
        Interface: __init__, forward
    '''

    def __init__(self, cfg):
        '''
            Overview: initialize architect.
            Arguments:
                - cfg (:obj:`dict`): head architecture definition
        '''
        super(TargetUnitHead, self).__init__()
        self.act = build_activation(cfg.activation)
        self.key_fc = fc_block(cfg.entity_embedding_dim, cfg.key_dim, activation=None, norm_type=None)
        self.func_fc = fc_block(cfg.unit_type_dim, cfg.func_dim, activation=self.act, norm_type=None)
        self.fc1 = fc_block(cfg.input_dim, cfg.func_dim, activation=None, norm_type=None)
        self.fc2 = fc_block(cfg.func_dim, cfg.key_dim, activation=None, norm_type=None)
        self.use_mask = cfg.use_mask

        self.pd = CategoricalPdPytorch
        self.key_dim = cfg.key_dim

    def _get_query(self, embedding, available_unit_type_mask):
        func_embed = self.func_fc(available_unit_type_mask)
        x = self.fc1(embedding)
        x = self.fc2(F.relu(x + func_embed))
        return x

    def forward(
        self,
        embedding,
        available_unit_type_mask,
        entity_embedding,
        temperature=1.0,
        target_unit=None,
        entity_num=None
    ):
        '''
            Overview: First func_embed is computed the same as in the Selected Units head, and used in the
                      same way for the query (added to the output of the autoregressive_embedding passed
                      through a linear of size 256). The query is then passed through a ReLU and a linear
                      of size 32, and the query is applied to the keys which are created the same way as
                      in the Selected Units head to get target_unit_logits. target_unit is sampled from
                      target_unit_logits using a multinomial with temperature 0.8. Note that since this is
                      one of the two terminal arguments (along with Location Head, since no action has
                      both a target unit and a target location), it does not return autoregressive_embedding.
            Arguments:
                - embedding (:obj`tensor`): autoregressive_embedding, [batch_size, input_dim(1024)]
                - available_unit_type_mask (:obj`tensor`): [batch_size, num_unit_type]
                - entity_embedding (:obj`tensor`): [batch_size, num_units, entity_embedding_dim(256)]
                - temperature (:obj:`float`): logits sample temperature
                - target_unit (:obj:`Tensor` or None): when SL training, the caller indicates target_unit
                - entity_num (:obj:'tensor): entity embedding was padding to same size, this is true entity numbers
            Returns:
                - (:obj`tensor`): logits, List(batch_size) - List(num_selected_units) - num_units
                - (:obj`tensor`): target_unit, [batch_size] target_unit index
        '''
        key = self.key_fc(entity_embedding)
        mask = sequence_mask(entity_num)
        query = self._get_query(embedding, available_unit_type_mask)  # b, -1
        Q = query.unsqueeze(1) / (1e-6 + torch.norm(query.unsqueeze(1), 2, -1, keepdim=True))
        K = key / (1e-6 + torch.norm(key, 2, -1, keepdim=True))
        logits = Q * K * 64
        # logits = query.unsqueeze(1) * key
        logits = logits.sum(dim=2)  # b, n, -1
        logits.masked_fill_(~mask, value=-1e9)

        logits = logits.div(0.8)
        # add ZH
        if target_unit is None:
            p = F.softmax(logits, dim=1)
            handle = self.pd(p)
            target_unit = handle.sample()
        return logits, target_unit


class LocationHead(nn.Module):
    '''
        Overview: The location head uses autoregressive_embedding and map_skip to get target_location_logits
                  and target_location.
        Interface: __init__, forward
    '''

    def __init__(self, cfg):
        '''
            Overview: initialize architect.
            Arguments:
                - cfg (:obj:`dict`): head architecture definition
        '''
        super(LocationHead, self).__init__()
        self.act = build_activation(cfg.activation)
        self.reshape_size = cfg.reshape_size
        self.reshape_channel = cfg.reshape_channel

        self.conv1 = conv2d_block(
            cfg.map_skip_dim * 4 + cfg.reshape_channel, cfg.res_dim, 1, 1, 0, activation=self.act, norm_type=None
        )
        self.res = nn.ModuleList()
        self.res_act = nn.ModuleList()
        self.res_dim = cfg.res_dim
        self.loc_type = cfg.get('loc_type', 'alphastar')
        self.proj_dim = cfg.get('proj_dim', 1600)
        if cfg.input_dim != self.proj_dim:
            self.project_embed = fc_block(self.proj_dim, cfg.input_dim)
        else:
            self.project_embed = None
        if self.loc_type == 'alphastar':
            self.FiLM_gamma = nn.ModuleList()
            self.FilM_beta = nn.ModuleList()
            self.FiLM = nn.ModuleList()
            self.res = nn.ModuleList()
            self.res_act = nn.ModuleList()
            for i in range(cfg.res_num):
                self.FiLM_gamma.append(nn.Linear(cfg.input_dim, self.res_dim, bias=False))
                self.FilM_beta.append(nn.Linear(cfg.input_dim, self.res_dim, bias=False))
                self.res.append(GatedResBlock(self.res_dim, self.res_dim, 3, 1, 1, activation=self.act, norm_type=None))
                self.FiLM.append(FiLMedResBlock(self.res_dim))
            for i in range(cfg.res_num):
                torch.nn.init.xavier_uniform_(self.FiLM_gamma[i].weight)
                torch.nn.init.xavier_uniform_(self.FilM_beta[i].weight)
        else:
            for i in range(cfg.res_num):
                self.res_act.append(
                    build_activation('glu')(self.res_dim, self.res_dim, cfg.map_skip_dim + cfg.reshape_channel,
                                            'conv2d')
                )
                self.res.append(ResBlock(self.res_dim, self.res_dim, 3, 1, 1, activation=self.act, norm_type=None))

        self.upsample = nn.ModuleList()  # upsample list
        dims = [self.res_dim] + cfg.upsample_dims
        assert (cfg.upsample_type in ['deconv', 'nearest', 'bilinear'])
        for i in range(len(cfg.upsample_dims)):
            if cfg.upsample_type == 'deconv':
                self.upsample.append(deconv2d_block(dims[i], dims[i + 1], 4, 2, 1, activation=self.act, norm_type=None))
            elif cfg.upsample_type == 'nearest':
                self.upsample.append(
                    nn.Sequential(
                        NearestUpsample(2),
                        conv2d_block(dims[i], dims[i + 1], 3, 1, 1, activation=self.act, norm_type=None)
                    )
                )
            elif cfg.upsample_type == 'bilinear':
                self.upsample.append(
                    nn.Sequential(
                        BilinearUpsample(2),
                        conv2d_block(dims[i], dims[i + 1], 3, 1, 1, activation=self.act, norm_type=None)
                    )
                )
        # self.ensemble_conv = conv2d_block(dims[-1], 1, 1, 1, 0, activation=None, norm_type=None)

        self.ratio = cfg.location_expand_ratio
        self.use_mask = cfg.use_mask
        self.output_type = cfg.output_type
        assert (self.output_type in ['cls', 'soft_argmax'])
        if self.output_type == 'cls':
            self.pd = CategoricalPdPytorch
        else:
            self.soft_argmax = SoftArgmax()


    def forward(self, embedding, map_skip, temperature=1.0, location=None):
        '''
            Overview: First autoregressive_embedding is reshaped to have the same height/width as the final skip
                      in map_skip (which was just before map information was reshaped to a 1D embedding) with 4
                      channels, and the two are concatenated together along the channel dimension, passed through
                      a ReLU, passed through a 2D convolution with 128 channels and kernel size 1, then passed
                      through another ReLU. The 3D tensor (height, width, and channels) is then passed through a
                      series of Gated ResBlocks with 128 channels, kernel size 3, and FiLM, gated on
                      autoregressive_embedding and using the elements of map_skip in order of last ResBlock skip
                      to first. Afterwards, it is upsampled 2x by each of a series of transposed 2D convolutions
                      with kernel size 4 and channel sizes 128, 64, 16, and 1 respectively (upsampled beyond the
                      128x128 input to 256x256 target location selection). Those final logits are flattened and
                      sampled (masking out invalid locations using `action_type`, such as those outside the camera
                      for build actions) with temperature 0.8 to get the actual target position.
            Arguments:
                - embedding (:obj`tensor`): autoregressive_embedding, [batch_size, input_dim(1024)]
                - map_skip (:obj`tensor`): tensors of the outputs of intermediate computations, len=res_num, each
                    element is a torch FloatTensor with shape[batch_size, res_dim, map_y // 8, map_x // 8]
                - available_location_mask (:obj`tensor`): [batch_size, 1, map_y, map_x]
                - temperature (:obj:`float`): temperature
                - location (:obj:`Tensor`):  when SL training, the caller indicates location
            Returns:
                - (:obj`tensor`): outputs, shape[batch_size, map_y, map_x](cls), shape[batch_size, 2](soft_argmax)
                - (:obj`tensor`): location, shape[batch_size, 2]
        '''
        if self.project_embed is not None:
            embedding = self.project_embed(embedding)
        reshape_embedding = embedding.reshape(-1, self.reshape_channel, *self.reshape_size)
        reshape_embedding = F.interpolate(reshape_embedding, size=map_skip[0].shape[2:], mode='bilinear', align_corners=False)
        # cat_feature = [torch.cat([reshape_embedding, map_skip[i]], dim=1) for i in range(len(map_skip))]
        cat_feature = torch.cat([reshape_embedding, torch.cat(map_skip, dim=1)], dim=1)

        x1 = self.act(cat_feature)
        x = self.conv1(x1)
        # reverse cat_feature instead of reversing resblock
        if self.loc_type == 'alphastar':
            # x = self.act(x)
            for index, layer in enumerate(self.res):
                x = layer(x, map_skip[len(map_skip) - index - 1])
                gamma = self.FiLM_gamma[index](embedding)
                beta = self.FilM_beta[index](embedding)
                x = self.FiLM[index](x, gammas=gamma, betas=beta)
                # x = x * gamma.view(x.shape[0], x.shape[1], 1, 1) + beta.view(x.shape[0], x.shape[1], 1, 1)
        else:
            for layer, act, skip in zip(self.res, self.res_act, reversed(cat_feature)):
                x = layer(x)
                x = act(x, skip)
        for layer in self.upsample:
            x = layer(x)
        # x = self.ensemble_conv(x)
        # x = F.interpolate(x, size=available_location_mask.shape[2:], mode='bilinear')
        if self.output_type == 'cls':
            W = x.shape[3]
            logits_flatten = x.view(x.shape[0], -1)
            logits_flatten = logits_flatten.div(0.8)
            if location is None:
                p = F.softmax(logits_flatten, dim=1)
                handle = self.pd(p)
                location = handle.sample()
                location = torch.stack([location // W, location % W], dim=1).float()
            return logits_flatten, location
        elif self.output_type == 'soft_argmax':
            x = self._map2origin_size(x)
            x = self.soft_argmax(x)
            if location is None:
                location = x.detach().long()
            return x, location
