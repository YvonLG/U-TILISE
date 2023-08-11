
import torch
from torch import nn

# modified from https://github.com/VSainteuf/utae-paps/blob/main/src/backbones/ltae.py
class LTAE(nn.Module):
    """2D L-TAE without temporal aggregation."""
    def __init__(self, input_nc: int, n_heads: int, d_key: int, attn_dropout: float=0.1, pos_encoding: bool=True):
        super(LTAE, self).__init__()

        self.n_heads = n_heads
        self.pos_encoding = pos_encoding

        self.norm = nn.GroupNorm(n_heads, input_nc)
        self.attn_heads = MultiHeadAttention(input_nc, n_heads, d_key)
        self.attn_dropout = nn.Dropout(attn_dropout)

        T = 1000
        self.d_model = input_nc // n_heads
        self.denom = T ** (2 * torch.arange(1, self.d_model+1) / self.d_model) # d_model


    def forward(self, x: torch.Tensor, doy: torch.Tensor|None):
        bs, d_seq, nc, h, w = x.shape

        v = x.permute(0, 3, 4, 2, 1).reshape(-1, nc, d_seq)
        v = self.norm(v).transpose(1, 2) # bs * h * w x d_seq x nc

        if self.pos_encoding and doy is not None:
            pos = self.get_pos_encoding(doy)
            pos = pos.tile((h*w, 1, self.n_heads))

            v += pos

        attn = self.attn_heads(v)
        attn = attn.view(bs, h, w, d_seq).movedim(3, 1)

        attn = self.attn_dropout(attn)

        return attn
    
    def get_pos_encoding(self, doy: torch.Tensor) -> torch.Tensor:
        # bs x d_seq
        bs, d_seq = doy.shape
        doy = doy.unsqueeze(-1).tile((1, 1, self.d_model)) # bs x d_seq x d_model
        denom = self.denom.view(1, 1, self.d_model).tile((bs, d_seq, 1))

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

        self.fc1 = nn.Conv1d(input_nc, 2 * n_heads * d_key, kernel_size=1, groups=n_heads)

        self.fc2 = nn.Sequential(
            nn.GroupNorm(n_heads, n_heads * d_key),
            nn.Conv1d(n_heads * d_key, n_heads * d_key, kernel_size=1, groups=n_heads)
        )
    
    def forward(self, v: torch.Tensor):
        # bs x d_seq x nc

        kq = self.fc1(v.transpose(1, 2)).transpose(1, 2) # bs x d_seq x 2 * n_heads * d_key

        k, q = torch.split(kq, self.n_heads * self.d_key, dim=2) # bs x d_seq x n_heads * d_key
        q = self.fc2(q.mean(dim=1).unsqueeze(-1)) # bs x n_heads * d_days x 1
        
        attn = torch.bmm(k, q).squeeze() / self.scale # bs x 1 x d_seq
        attn = torch.softmax(attn, dim=1)

        return attn
    
if __name__ == '__main__':
    
    ltae = LTAE(64, 4, 5)

    x = torch.rand(3, 6, 64, 32, 32)
    doy = torch.rand(3, 6)

    print(ltae(x, doy).shape)