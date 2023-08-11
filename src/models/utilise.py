
from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F

from .convblocks import ConvBlock, DownBlock, UpBlock
from .ltae import LTAE

class UTILISE(nn.Module):
    def __init__(self, input_nc, output_nc, nch=64, n_depth: int=3, activ: str|Callable='relu', norm: str|Callable='batch', padding: str='zeros', n_groups: int|None=None,
                 upsample: str='bilinear', n_heads: int=4, d_key: int=4, attn_dropout: float=0.1, pos_encoding: bool=True):
        super(UTILISE, self).__init__()

        self.n_depth = n_depth

        encoder = [
            ConvBlock(input_nc, nch, activ, norm, padding, n_groups)
        ]

        for i in range(n_depth):
            encoder.append(
                DownBlock(nch, nch if i != n_depth-1 else 2 * nch, activ, norm, padding, n_groups)
            )

        skip_convs = [nn.Conv2d(ch, ch, 1, 1) for ch in [nch] * (n_depth) + [2 * nch]]
        self.skip_convs = nn.ModuleList(reversed(skip_convs))

        self.encoder = nn.ModuleList(encoder)

        self.ltae = LTAE(2 * nch, n_heads, d_key, attn_dropout, pos_encoding)

        decoder = []
        for i in range(n_depth):
            is_first = i == 0
            decoder.append(
                UpBlock(2 * nch, nch, activ, norm, padding, n_groups, upsample, has_conv_block=is_first)
            )
        
        decoder += [
            ConvBlock(2 * nch, nch, activ, norm, padding, n_groups),
            ConvBlock(nch, output_nc, activ, norm, padding, n_groups, last_activ=False)
        ]

        self.decoder = nn.ModuleList(decoder)

    def forward(self, x: torch.Tensor, doy: torch.Tensor|None, return_attn: bool=False):
        # bs x d_seq x nc x h x w
        bs, d_seq, nc, h, w = x.shape
        x = torch.flatten(x, 0, 1)

        embs = []
        for block in self.encoder:
            x = block(x)
            embs.append(x)

        v = x.view(bs, d_seq, *x.shape[1:])

        attn = self.ltae(v, doy=doy)
        
        for i, emb in enumerate(reversed(embs)):
            emb = emb.view(bs, d_seq, *emb.shape[1:])
            weight = F.interpolate(attn, scale_factor=2 ** i, mode='bilinear')
            weight = torch.tile(weight.unsqueeze(2), (1, 1, emb.size(2), 1, 1))

            emb = emb * weight
            emb = torch.flatten(emb, 0, 1)

            emb = self.skip_convs[i](emb)
            
            if i == 0:
                y = emb
            else:
                y = torch.cat([y, emb], dim=1)
            y = self.decoder[i](y)
        
        y = self.decoder[-1](y)
        y = y.view(bs, d_seq, *y.shape[1:])
        y = F.sigmoid(y)

        return y, attn if return_attn else y



if __name__ == '__main__':
    import torch

    utilise = UTILISE(4, 4, pos_encoding=True)

    x = torch.zeros((1, 10, 4, 128, 128))
    doy = torch.rand((1, 10))

    print(utilise(x, doy, return_attn=True))


    


