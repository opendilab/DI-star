import torch
from collections import defaultdict
import math
from torch.nn.utils import clip_grad_norm_
from torch._six import inf


def build_grad_clip(cfg):
    clip_type = cfg.get('type', 'none')
    clip_threshold = cfg.get('threshold', 1.4)
    clip_norm_type = cfg.get('norm_type', 2)
    clip_begin_step = cfg.get('begin_step', 100)
    clip_ignore_threshold = cfg.get('ignore_threshold', 3)
    if clip_norm_type == 'inf':
        clip_norm_type = inf
    return GradClip(clip_type, clip_threshold, clip_norm_type, clip_begin_step, clip_ignore_threshold)


class GradClip(object):
    def __init__(self, clip_type, threshold, norm_type, begin_step, ignore_threshold):
        assert (clip_type in ['max_norm', 'clip_value', 'none', 'clip_const', 'pytorch_norm', 'momentum_norm'])
        self.clip_type = clip_type
        self.threshold = threshold
        self.norm_type = norm_type
        self.clip_value = 0
        self.step = 0
        self.beta1 = 0.95
        self.beta2 = 0.999
        self.begin_step = begin_step
        self.ignore_threshold = ignore_threshold
        self.state = defaultdict(dict)
        if self.clip_type == 'momentum_norm':
            self.norm_mom = None
            self.flag = 0
    def apply(self, parameters):
        self.step += 1
        with torch.no_grad():
            if self.clip_type == 'none':
                if isinstance(parameters, torch.Tensor):
                    parameters = [parameters]
                parameters = list(filter(lambda p: p.grad is not None, parameters))
                norm_type = float(self.norm_type)
                total_norm = 0
                for p in parameters:
                    param_norm = p.grad.data.norm(norm_type)
                    total_norm += param_norm.item() ** norm_type
                total_norm = total_norm ** (1. / norm_type)

            elif self.clip_type == 'max_norm':
                bias_correction1 = 1 - self.beta1 ** self.step
                if isinstance(parameters, torch.Tensor):
                    parameters = [parameters]
                parameters = list(filter(lambda p: p.grad is not None, parameters))
                norm_type = float(self.norm_type)
                total_norm = 0
                for p in parameters:
                    param_norm = p.grad.data.norm(norm_type)
                    total_norm += param_norm.item() ** norm_type
                total_norm = total_norm ** (1. / norm_type)

                if self.step <= self.begin_step:
                    clip_coef = 1
                    self.clip_value = self.beta1 * self.clip_value + (1 - self.beta1) * total_norm
                else:
                    clip_coef = (self.clip_value / bias_correction1) * self.threshold / (total_norm + 1e-6)
                    #  ignore anomaly grad
                    # if total_norm > (self.clip_value / bias_correction1) * self.ignore_threshold:
                    #     clip_coef = 0
                    # else:
                    self.clip_value = self.beta1 * self.clip_value + (1 - self.beta1) * total_norm
                if clip_coef < 1:
                    for p in parameters:
                        p.grad.data.mul_(clip_coef)
            elif self.clip_type == 'momentum_norm':
                bias_correction1 = 1 - self.beta1 ** self.step
                if isinstance(parameters, torch.Tensor):
                    parameters = [parameters]
                parameters = list(parameters)
                norm_type = float(self.norm_type)
                if self.flag == 0:
                    self.norm_mom = [None] * len(parameters)
                total_norm = 0
                norm_scale = self.threshold
                for idx, p in enumerate(parameters):
                    if p.grad is not None:
                        g_norm = p.grad.data.norm(norm_type)
                        if self.norm_mom[idx] is not None:
                            m_norm = self.norm_mom[idx]
                            if g_norm < norm_scale * m_norm:
                                s_temp = 1.0
                            else:
                                s_temp = norm_scale * m_norm / (g_norm+1e-6)
                        else:
                            s_temp = 1.0

                        p.grad.data.mul_(s_temp)

                for idx, p in enumerate(parameters):
                    if p.grad is not None:                        
                        g_norm = p.grad.data.norm(norm_type)
                        if self.norm_mom[idx] is None:
                            self.norm_mom.append(float(g_norm))
                        else:
                            self.norm_mom[idx] = self.norm_mom[idx] * 0.99 + float(g_norm) * 0.01
                        total_norm += g_norm.item() ** norm_type
                total_norm = total_norm ** (1. / norm_type)
                self.flag = 1


            elif self.clip_type == 'clip_value':
                norm_type = float(self.norm_type)
                total_norm = 0
                bias_correction2 = 1 - self.beta2 ** self.step
                for p in parameters:
                    if p.grad is None:
                        continue
                    param_norm = p.grad.data.norm(norm_type)
                    total_norm += param_norm.item() ** norm_type
                    state = self.state[p]
                    grad = p.grad.data
                    if len(state) == 0:
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p.data, device=p.data.device)
                    state['exp_avg_sq'].mul_(self.beta2).addcmul_(1 - self.beta2, grad, grad)
                    if self.step >= 100:    # initial value is inaccurate
                        flag = grad.abs() > (state['exp_avg_sq'].sqrt() / math.sqrt(bias_correction2)) * 5
                        grad.mul_(~flag).add_(((state['exp_avg_sq'].sqrt() / math.sqrt(bias_correction2)) * 5).mul_(flag))
                total_norm = total_norm ** (1. / norm_type)

            elif self.clip_type == 'clip_const':
                if isinstance(parameters, torch.Tensor):
                    parameters = [parameters]
                norm_type = float(self.norm_type)
                total_norm = 0
                for p in filter(lambda p: p.grad is not None, parameters):
                    param_norm = p.grad.data.norm(norm_type)
                    total_norm += param_norm.item() ** norm_type
                    p.grad.data.clamp_(min=-self.threshold, max= self.threshold)
                total_norm = total_norm ** (1. / norm_type)

            elif self.clip_type == 'pytorch_norm':
                total_norm = clip_grad_norm_(parameters, self.threshold, self.norm_type)
                if isinstance(total_norm, torch.Tensor):
                    total_norm = total_norm.item()

            elif self.clip_type == 'clip_norm':
                total_norm = clip_grad_norm_(parameters, self.threshold, self.norm_type)
                if isinstance(total_norm, torch.Tensor):
                    total_norm = total_norm.item()

        return total_norm
