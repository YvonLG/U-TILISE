
from typing import Callable, List

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat

from conv_blocks import ConvBlock, DownBlock, UpBlock
from temporal_encoder import TemporalAttentionEncoder2D

class UTILISE(nn.Module):
    def __init__(self,
                 input_nc,
                 output_nc,
                 nch=64,
                 n_depth: int=3,
                 activ: str|Callable='relu',
                 norm: str|Callable='none',
                 padding: str='zeros',
                 n_groups: int|None=None,
                 upsample: str='bilinear',
                 n_heads: int=4,
                 d_key: int=4,
                 ffn_nc: List[int]=[256],
                 attn_dropout: float=0.1,
                 dropout: float=0.1,
                 pos_encoding: bool=True,
                 learn_encoding: bool=False
                 ):
        super().__init__()

        self.n_depth = n_depth

        encoder = [
            ConvBlock(input_nc, nch, activ, 'none', padding, n_groups)
        ]
        for i in range(n_depth):
            encoder += [
                DownBlock(nch, nch if i != n_depth-1 else 2 * nch, activ, norm, padding, n_groups)
            ]
        self.encoder = nn.ModuleList(encoder)
        
        self.tae = TemporalAttentionEncoder2D(2 * nch, n_heads, d_key, ffn_nc, attn_dropout, dropout, pos_encoding, learn_encoding)
        self.oneone_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(nch * n_heads, nch, 1, 1),
                nn.ReLU(inplace=True)
            ) for _ in range(n_depth)
        ])

        decoder = []
        for i in range(n_depth):
            is_first = i == 0
            decoder += [
                UpBlock(2 * nch, nch, activ, norm, padding, n_groups, upsample, has_conv_block=is_first)
            ]
        decoder += [
            ConvBlock(2 * nch, nch, activ, norm, padding, n_groups),
            ConvBlock(nch, output_nc, activ, 'none', padding, n_groups, last_activ=False)
        ]
        self.decoder = nn.ModuleList(decoder)
    
    def forward(self, x: torch.Tensor, doy: torch.Tensor|None, pad_mask: torch.Tensor|None, return_attn: bool=False):
        b, t, _, _, _ = x.shape
        x = rearrange(x, 'b t c h w -> (b t) c h w')

        skips = []
        for block in self.encoder:
            x = block(x)
            skips.append(x)
        skips = skips[::-1]

        x_tae = rearrange(x, '(b t) c h w -> b t c h w', b=b)
        out, attn = self.tae(x_tae, doy, pad_mask)
        
        x = rearrange(out, 'b t c h w -> (b t) c h w')
        x = self.decoder[0](x)
        
        n_heads = attn.size(1)
        attn_scalable = rearrange(attn, 'b head t1 t2 h w -> (b head t1) t2 h w')
        for i in range(1, len(skips)):

            attn_skip = F.interpolate(attn_scalable, scale_factor=2 ** i, mode='bilinear')
            attn_skip = rearrange(attn_skip, '(b head t1) t2 hs ws -> head (hs ws b) t1 t2', head=n_heads, b=b, t1=t, t2=t)

            skip = skips[i]
            _, _, hs, ws = skip.shape
            skip = rearrange(skip, '(b t) c hs ws -> 1 (hs ws b) t c', b=b)
            skip = torch.matmul(attn_skip, skip)

            skip = rearrange(skip, 'head (hs ws b) t c -> (b t) (head c) hs ws', hs=hs, ws=ws)
            skip = self.oneone_convs[i-1](skip)

            x = torch.cat([x, skip], dim=1)
            x = self.decoder[i](x)
        
        x = self.decoder[-1](x)
        x = rearrange(x, '(b t) c h w -> b t c h w', b=b)
        x = F.sigmoid(x)
        if return_attn:
            return x, attn
        return x

