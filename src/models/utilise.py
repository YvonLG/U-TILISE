
from typing import Callable, List

import torch
from torch import nn
import torch.nn.functional as F

from .convblocks import ConvBlock, DownBlock, UpBlock
from .ltae import LTAE

class UTILISE(nn.Module):
    def __init__(self, input_nc, output_nc, nch=64, n_depth: int=3, activ: str|Callable='relu', norm: str|Callable='batch', padding: str='zeros', n_groups: int|None=None,
                 upsample: str='bilinear', n_heads: int=4, d_key: int=4, mlp_nc: List[int]=[256], attn_dropout: float=0.1, pos_encoding: bool=True):
        super(UTILISE, self).__init__()

        self.n_depth = n_depth

        encoder = [
            ConvBlock(input_nc, nch, activ, norm, padding, n_groups)
        ]

        for i in range(n_depth):
            encoder += [
                DownBlock(nch, nch if i != n_depth-1 else 2 * nch, activ, norm, padding, n_groups)
            ]

        self.encoder = nn.ModuleList(encoder)

        self.ltae = LTAE(2 * nch, n_heads, d_key, mlp_nc, attn_dropout, pos_encoding)

        skip_fc = [
            nn.Sequential(
                nn.Linear(nch * n_heads, nch),
                nn.ReLU()
            ) for _ in range(n_depth)
        ]
        self.skip_fc = nn.ModuleList(skip_fc)

        decoder = []
        for i in range(n_depth):
            is_first = i == 0
            decoder += [
                UpBlock(2 * nch, nch, activ, norm, padding, n_groups, upsample, has_conv_block=is_first)
            ]
        
        decoder += [
            ConvBlock(2 * nch, nch, activ, norm, padding, n_groups),
            ConvBlock(nch, output_nc, activ, norm, padding, n_groups, last_activ=False)
        ]

        self.decoder = nn.ModuleList(decoder)

    def forward(self, x: torch.Tensor, doy: torch.Tensor|None, pad_mask: torch.Tensor|None=None, return_attn: bool=False):
        bs, d_seq, nc, h, w = x.shape
        x = x.view(-1, nc, h, w)

        skips = []
        for block in self.encoder:
            x = block(x)
            skips.append(x.view(bs, d_seq, *x.shape[1:]))
        skips = skips[::-1]

        v = x.view(bs, d_seq, *x.shape[1:])
        out, attn = self.ltae(v, doy, pad_mask)
        n_heads = attn.size(1)

        attn_scalable = attn.view(-1, *attn.shape[-3:])
        for i, skip in enumerate(skips):

            if i == 0:
                out = out.view(-1, *out.shape[2:])
                out = self.decoder[i](out)
                continue

            _, _, snc, sh, sw = skip.shape

            attn_scale = F.interpolate(attn_scalable, scale_factor=2 ** i, mode='bilinear')
            attn_scale = attn_scale.view(bs, n_heads, d_seq, d_seq, sh, sw)


            attn_scale = attn_scale.permute(0, 4, 5, 1, 2, 3).reshape(-1, n_heads, d_seq, d_seq)
            skip = skip.permute(0, 3, 4, 1, 2).reshape(-1, d_seq, snc)

            skip2 = torch.matmul(attn_scale, skip.unsqueeze(1))
            skip2 = skip2.permute(0, 2, 1, 3).reshape(skip2.size(0), d_seq, -1)
            skip2 = self.skip_fc[i-1](skip2)

            # skip2 = skip2 + skip ?

            skip2 = skip2.view(bs, sh, sw, d_seq, snc).permute(0, 3, 4, 1, 2)
            skip2 = skip2.reshape(-1, snc, sh, sw)

            out = torch.cat([out, skip2], dim=1)
            out = self.decoder[i](out)

        out = self.decoder[-1](out)
        out = F.sigmoid(out).view(bs, d_seq, *out.shape[-3:])

        if return_attn:
            return out, attn
        return out
            




if __name__ == '__main__':
    import torch

    utilise = UTILISE(4, 4, pos_encoding=True, n_heads=5)

    x = torch.zeros((1, 10, 4, 128, 128))
    doy = torch.rand((1, 10))

    print(utilise(x, doy, return_attn=True))


    


