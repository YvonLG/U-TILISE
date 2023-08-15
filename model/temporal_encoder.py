
from typing import List

import torch
from torch import nn
from einops import rearrange, repeat

class TemporalAttentionEncoder2D(nn.Module):
    """Input and output are spatial tensors but all the logic happens along the time dimension."""
    def __init__(self, input_nc: int, n_heads: int, d_key: int, ffn_nc: List[int]=[256], attn_dropout: float=0.1,
                 dropout: float=0.1, pos_encoding: bool=True, learn_encoding: bool=False):
        super().__init__()
        
        self.n_heads = n_heads
        self.pos_encoding = pos_encoding

        self.attn_heads = MultiHeadAttention(n_heads, input_nc, d_key, attn_dropout, dropout)

        ffn_nc = [input_nc] + ffn_nc + [input_nc]
        ffn = []
        for i in range(1, len(ffn_nc)):
            ffn += [
                nn.Linear(ffn_nc[i-1], ffn_nc[i]),
                nn.GELU()
            ]
        self.ffn = nn.Sequential(*ffn)

        self.groupnorm1 = nn.GroupNorm(n_heads, input_nc)
        self.groupnorm2 = nn.GroupNorm(n_heads, input_nc)
        self.groupnorm3 = nn.GroupNorm(n_heads, input_nc)

        T = 1000
        self.d_hid = input_nc // n_heads
        self.denom = torch.pow(T, 2 * torch.arange(1, self.d_hid+1) / self.d_hid)

        self.w_pos = None
        if learn_encoding:
            self.w_pos = nn.Linear(2, self.d_hid)

    def get_encoding(self, doy: torch.Tensor):

        if self.w_pos is not None:
            angle = 2 * torch.pi * (doy-1) / 365
            embed = torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)
            pos = self.w_pos(embed)

        else:
            b, t = doy.shape
            denom = repeat(self.denom, 'd_hid -> b t d_hid', b=b, t=t)
            doy = repeat(doy, 'b t -> b t d_hid', d_hid=self.d_hid)

            pos = doy / denom
            pos[:,:,0::2] = torch.sin(pos[:,:,0::2])
            pos[:,:,1::2] = torch.cos(pos[:,:,1::2])
        
        return pos

    def forward(self, x: torch.Tensor, doy: torch.Tensor|None, mask: torch.Tensor|None):
        _, _, _, h, w = x.shape

        v = rearrange(x, 'b t c h w -> (h w b) c t')
        v = self.groupnorm1(v).transpose(1, 2)

        if self.pos_encoding:
            pos = self.get_encoding(doy)
            pos = repeat(pos, 'b t d_hid -> (h w b) t (repeat d_hid)', h=h, w=w, repeat=self.n_heads)
            v = v + pos

        if mask is not None:
            mask = repeat(mask, 'b t -> (b repeat) t', repeat=h * w)
            mask = rearrange(mask, 'b t -> 1 b 1 t')
        
        out, attn = self.attn_heads(v, mask)
        out = out + v
        out = self.groupnorm2(out.transpose(1, 2)).transpose(1, 2)
        out = self.ffn(out) + out
        out = self.groupnorm3(out.transpose(1, 2)).transpose(1, 2)

        out = rearrange(out, '(h w b) t c -> b t c h w', h=h, w=w)
        attn = rearrange(attn, 'head (h w b) t1 t2 -> head b t1 t2 h w', h=h, w=w)

        return out, attn

class MultiHeadAttention(nn.Module):
    """Modified mutli-head attention. The embedding v is defined as the module input without the linear projection
    of the original implementation."""
    def __init__(self, n_heads: int, d_model: int, d_key: int, attn_dropout: float=0.1, dropout: float=0.1):
        super().__init__()

        self.n_heads = n_heads
        self.d_key = d_key
        self.scale = d_key ** (-0.5)
        
        self.w_kqs = nn.Linear(d_model, 2 * n_heads * d_key)
        self.fc = nn.Linear(n_heads * d_model, d_model)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, v: torch.Tensor, mask: torch.Tensor):

        kq = self.w_kqs(v)
        kq = rearrange(kq, 'b t (head k) -> head b t k', head=self.n_heads)
        k, q = kq.chunk(2, dim=-1)

        attn: torch.Tensor = torch.matmul(q, k.transpose(2, 3)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask, -1e5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v.unsqueeze(0))

        out = rearrange(out, 'head b t c -> b t (head c)', head=self.n_heads)

        out = self.fc(out)
        out = self.dropout(out)
        return out, attn
