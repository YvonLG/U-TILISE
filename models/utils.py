
from typing import Callable
from functools import partial

from torch import nn
from torch.nn import init

def get_activ(activ: str|Callable):
    if callable(activ):
        return activ
    
    if activ == 'relu':
        return partial(nn.ReLU, inplace=True)
    elif activ == 'none':
        return nn.Identity
    else:
        raise NotImplementedError

def get_norm(norm: str|Callable, n_groups: int|None=None):
    if callable(norm):
        return norm
    
    if norm == 'batch':
        return nn.BatchNorm2d
    elif norm == 'instance':
        return nn.InstanceNorm2d
    elif norm == 'groups':
        return lambda num_channels: nn.GroupNorm(n_groups, num_channels)
    elif norm == 'none':
        return nn.Identity
    else:
        raise NotImplementedError

# https://github.com/junyanz/BicycleGAN/blob/master/models/networks.py
def initialize_weights(net: nn.Module, initialization: str='normal'):
    def init_func(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)) and hasattr(m, 'weight'):
            if initialization == 'normal':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif initialization == 'xavier':
                init.xavier_normal_(m.weight.data, gain=0.02)
            elif initialization == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif initialization == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=0.02)
            else:
                raise NotImplementedError
            
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
    net.apply(init_func)