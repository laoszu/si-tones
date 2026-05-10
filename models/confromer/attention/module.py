import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.confromer.attention.rmha import RelativeMultiHeadAttention

class SelfAttentionModule(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        self.norm = nn.LayerNorm(d_model)
        self.attn = RelativeMultiHeadAttention(d_model, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, pos_enc: Tensor, key_padding_mask: Tensor | None = None, ) -> Tensor:
        residual = x
        x = self.norm(x)
        x = self.attn(x, pos_enc, key_padding_mask)
        
        return residual + self.dropout(x)