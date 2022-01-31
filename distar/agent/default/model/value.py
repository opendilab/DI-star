import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from distar.ctools.torch_utils import fc_block, fc_block2, build_activation
from distar.ctools.torch_utils.network.res_block import ResFCBlock2


class ValueBaseline(nn.Module):
    '''
    Network to be applied on each baseline input, parameters for the current baseline are defined in the cfg

    input_dim            res_dim            res_dim                    1
    x -> fc (norm & act) -> ResFCBlock*res_num -> fc (no norm no act) -> atan act -> value
    '''

    def __init__(self, cfg, use_value_feature):
        super(ValueBaseline, self).__init__()
        self.act = build_activation(cfg.activation)
        if use_value_feature:
            input_dim = cfg.input_dim + 1056
        else:
            input_dim = cfg.input_dim
        self.project = fc_block(input_dim, cfg.res_dim, activation=self.act, norm_type=None)
        blocks = [ResFCBlock2(cfg.res_dim, self.act, cfg.norm_type) for _ in range(cfg.res_num)]
        self.res = nn.Sequential(*blocks)
        self.value_fc = fc_block2(cfg.res_dim, 1, activation=None, norm_type=None, gain=0.1)
        self.atan = cfg.atan
        self.PI = np.pi

    def forward(self, x):
        x = self.project(x)
        x = self.res(x)
        x = self.value_fc(x)

        x = x.squeeze(1)
        if self.atan:
            x = (2.0 / self.PI) * torch.atan((self.PI / 2.0) * x)
        return x


def test_value_baseline():

    class CFG:

        def __init__(self):
            self.activation = 'relu'
            self.norm_type = 'LN'
            self.input_dim = 1024
            self.res_dim = 256
            self.res_num = 16

    model = ValueBaseline(CFG())
    inputs = torch.randn(4, 1024)
    output = model(inputs)
    assert (output.shape == (4, ))


if __name__ == "__main__":
    test_value_baseline()
