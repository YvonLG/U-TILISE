
from typing import Callable
from functools import partial

import torch
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

class ConvBlock(nn.Module):
    """3x3 Conv2D followed by 3x3 res. Conv2D."""
    def __init__(self, input_nc: int, output_nc: int, activ: str|Callable='relu', norm: str|Callable='batch', padding: str='zeros', n_groups: int|None=None, last_activ=True):
        super(ConvBlock, self).__init__()
        
        my_activ = get_activ(activ)
        my_norm = get_norm(norm, n_groups)

        self.conv1 = nn.Conv2d(input_nc, output_nc, 3, 1, 1, padding_mode=padding)
        self.norm1 = my_norm(output_nc)
        self.activ1 = my_activ()
        
        self.conv2 = nn.Conv2d(output_nc, output_nc, 3, 1, 1, padding_mode=padding)
        self.norm2 = my_norm(output_nc)
        self.activ2 = my_activ() if last_activ else nn.Identity()

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activ1(x)

        y = self.conv2(x)
        y = self.norm2(y)
        y = self.activ2(x + y)
        return y

class DownBlock(nn.Module):
    """Strided Conv2D followed by ConvBlock. Halves the spatial dim."""
    def __init__(self, input_nc: int, output_nc: int, activ: str|Callable='relu', norm: str|Callable='batch', padding: str='zeros', n_groups: int|None=None):
        super(DownBlock, self).__init__()

        my_activ = get_activ(activ)
        my_norm = get_norm(norm, n_groups)

        self.down = nn.Conv2d(input_nc, input_nc, 3, 2, 1, padding_mode=padding)
        self.norm = my_norm(input_nc)
        self.activ = my_activ()

        # last_activ is by default True in DownBlock
        self.conv_block = ConvBlock(input_nc, output_nc, activ, norm, padding, n_groups, last_activ=True)

    def forward(self, x: torch.Tensor):
        x = self.down(x)
        x = self.norm(x)
        x = self.activ(x)
        x = self.conv_block(x)
        return x
    
class UpBlock(nn.Module):
    """Either ConvTranspose2D or Upsample followed by Conv2D. Can be preceded by a ConvBlock as well."""
    def __init__(self, input_nc: int, output_nc: int, activ: str|Callable='relu', norm: str|Callable='batch', padding: str='zeros', n_groups: int|None=None,
                 upsample: str='bilinear', has_conv_block: bool=True):
        super(UpBlock, self).__init__()

        self.conv_block = None
        if has_conv_block:
            self.conv_block = ConvBlock(input_nc, input_nc, activ, norm, padding, n_groups)

        my_activ = get_activ(activ)
        my_norm = get_norm(norm, n_groups)

        if upsample == 'basic':
            self.up = nn.ConvTranspose2d(input_nc, output_nc, 3, 2, 1, padding_mode=padding, output_padding=1)
        elif upsample == 'bilinear':
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode=upsample),
                nn.Conv2d(input_nc, output_nc, 3, 1, 1, padding_mode=padding)
            )
        else:
            raise NotImplementedError
        
        self.norm = my_norm(output_nc)
        self.activ = my_activ()

    def forward(self, x: torch.Tensor):
        if self.conv_block is not None:
            x = self.conv_block(x)
        x = self.up(x)
        x = self.norm(x)
        x = self.activ(x)
        return x

