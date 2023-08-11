
from functools import partial
from typing import Callable

from torch import nn

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
    elif norm == 'none':
        return nn.Identity
    else:
        raise NotImplementedError