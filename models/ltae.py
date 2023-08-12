
from typing import List

import torch
from torch import nn

# modified from https://github.com/VSainteuf/utae-paps/blob/main/src/backbones/ltae.py
class LTAE(nn.Module):
    """2D L-TAE without temporal aggregation."""
    def __init__(self, input_nc: int, n_heads: int, d_key: int, mlp_nc: List[int], attn_dropout: float=0.1, pos_encoding: bool=True):
        super(LTAE, self).__init__()

        self.n_heads = n_heads
        self.pos_encoding = pos_encoding

        self.group_norm1 = nn.GroupNorm(n_heads, input_nc)
        self.attn_heads = MultiHeadAttention(input_nc, n_heads, d_key)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.group_norm2 = nn.GroupNorm(n_heads, input_nc)
        
        mlp_nc = [input_nc] + mlp_nc + [input_nc]
        mlp = []
        for i in range(1, len(mlp_nc)):
            mlp += [
                nn.Linear(mlp_nc[i-1], mlp_nc[i]),
                nn.GELU()
            ]
        self.mlp = nn.Sequential(*mlp)

        self.group_norm3 = nn.GroupNorm(n_heads, input_nc)

        T = 1000
        self.d_hid = input_nc // n_heads
        self.denom = torch.pow(T, 2 * torch.arange(1, self.d_hid+1) / self.d_hid) # d_hid

    def forward(self, x: torch.Tensor, doy: torch.Tensor|None, pad_mask: torch.Tensor|None):
        bs, d_seq, nc, h, w = x.shape

        v = x.permute(0, 3, 4, 2, 1).reshape(-1, nc, d_seq)
        v = self.group_norm1(v).transpose(1, 2) # bs * h * w x d_seq x nc

        if self.pos_encoding and doy is not None:
            doy = doy.repeat_interleave(h*w, dim=0)
            pos = self.get_pos_encoding(doy)
            pos = pos.tile(1, 1, self.n_heads)
            v += pos

        if pad_mask is not None:
            pad_mask = pad_mask.view(bs, 1, 1, d_seq).repeat_interleave(h*w, dim=0)
        out, attn = self.attn_heads(v, pad_mask)
        attn = self.attn_dropout(attn)

        out = out + v
        out = self.group_norm2(out.transpose(1, 2)).transpose(1, 2)

        out = out + self.mlp(out)
        out = self.group_norm3(out.transpose(1, 2)).transpose(1, 2)

        out = out.view(bs, h, w, d_seq, nc).permute(0, 3, 4, 1, 2).contiguous()
        attn = attn.view(bs, h, w, self.n_heads, d_seq, d_seq).permute(0, 3, 4, 5, 1, 2).contiguous()
        return out, attn
    
    def get_pos_encoding(self, doy: torch.Tensor) -> torch.Tensor:
        # bs x d_seq
        bs, d_seq = doy.shape
        doy = doy.view(bs, d_seq, 1).expand(-1, -1, self.d_hid) # bs x d_seq x d_model
        denom = self.denom.view(1, 1, self.d_hid).expand(bs, d_seq, -1)
        denom = denom.to(doy.device)

        pos = doy / denom
        pos[:,:,0::2] = torch.sin(pos[:,:,0::2])
        pos[:,:,1::2] = torch.cos(pos[:,:,1::2])
        return pos

class MultiHeadAttention(nn.Module):
    def __init__(self, input_nc: int, n_heads: int, d_key: int):
        super(MultiHeadAttention, self).__init__()

        self.n_heads = n_heads
        self.d_key = d_key
        self.scale = d_key ** 0.5
        self.d_embed = input_nc // n_heads

        self.fc1 = nn.Linear(input_nc, 2 * n_heads * d_key)
        self.fc2 = nn.Linear(input_nc * n_heads, input_nc)
    
    def forward(self, v: torch.Tensor, pad_mask: torch.Tensor|None):
        bs, d_seq, nc = v.shape

        kq = self.fc1(v)

        kq = kq.view(bs, d_seq, self.n_heads, 2 * self.d_key)
        kq = kq.permute(0, 2, 1, 3) # bs x n_heads x d_seq x 2 * d_key

        k, q = kq.chunk(2, dim=3) # bs x n_heads x d_seq x d_key
        
        attn = torch.matmul(q, k.transpose(2, 3)) / self.scale # bs x n_heads x d_seq x d_seq
        if pad_mask is not None:
            attn = torch.masked_fill(attn, pad_mask, -1e3)
        attn = torch.softmax(attn, dim=3)

        out = torch.matmul(attn, v.unsqueeze(1))

        out = out.permute(0, 2, 1, 3).reshape(bs, d_seq, -1)
        out = self.fc2(out)

        return out, attn
    
if __name__ == '__main__':
    
    ltae = LTAE(128, 4, 5, [256])

    x = torch.rand(3, 6, 128, 32, 32)
    pad_mask = torch.zeros((3, 6), dtype=torch.bool)
    pad_mask[0,1] = 1
    doy = torch.rand(3, 6)

    out, attn  = ltae(x, doy, pad_mask)

    print(attn[0,0,3,1])