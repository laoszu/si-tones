import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RelativeMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.pos_proj = nn.Linear(d_model, d_model, bias=False)

        self.u_bias = nn.Parameter(torch.zeros(num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.zeros(num_heads, self.d_head))

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_head)

    def _rel_shift(self, x):
        B, H, T, T2 = x.shape
        zero_pad = torch.zeros(B, H, T, 1, device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)
        x_padded = x_padded.view(B, H, T2 + 1, T)
        x = x_padded[:, :, 1:, :]
        return x[:, :, :T, :]  # (B, H, T, T)

    def forward(self, x, pos_enc, key_padding_mask=None):
        B, T, _ = x.shape
        H, D = self.num_heads, self.d_head

        Q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        K = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        V = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        R = self.pos_proj(pos_enc)
        R = R.view(1, 2 * T - 1, H, D).transpose(1, 2)

        # content based attention
        Q_u = Q + self.u_bias.unsqueeze(0).unsqueeze(2)
        attn_content = torch.matmul(Q_u, K.transpose(-2, -1))

        # position based attention
        Q_v = Q + self.v_bias.unsqueeze(0).unsqueeze(2)
        attn_pos = torch.matmul(Q_v, R.transpose(-2, -1))
        attn_pos = self._rel_shift(attn_pos)

        attn = (attn_content + attn_pos) / self.scale

        if key_padding_mask is not None:
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn = self.dropout(F.softmax(attn, dim=-1))

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)  # (B, T, d_model)
        return self.out_proj(out)


class SelfAttentionModule(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()

        self.norm = nn.LayerNorm(d_model)
        self.attn = RelativeMultiHeadAttention(d_model, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos_enc, key_padding_mask=None):
        residual = x
        x = self.norm(x)
        x = self.attn(x, pos_enc, key_padding_mask)
        return residual + self.dropout(x)