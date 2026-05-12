import torch
import torch.nn as nn

from models.confromer.convolution.module import ConvolutionModule
from models.confromer.feed_forward.module import FeedForwardModule
from models.confromer.attention.module import SelfAttentionModule
from models.confromer.attention.embedding import RelPosEncoding

class ConformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ffn_dim, conv_kernel, dropout):
        super().__init__()

        # feed forward module (1.)
        self.ffn1 = FeedForwardModule(d_model, ffn_dim=ffn_dim, dropout=dropout)

        # self-attention module
        self.self_attn = SelfAttentionModule(d_model, num_heads, dropout)

        # convolution module
        self.conv = ConvolutionModule(d_model, conv_kernel, dropout=dropout)

        # feed forward module (2.)
        self.ffn2 = FeedForwardModule(d_model, ffn_dim=ffn_dim,dropout=dropout)

        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x, pos_enc, key_padding_mask=None):
        x = x + 0.5 * self.ffn1(x)
        x = self.self_attn(x, pos_enc, key_padding_mask)
        x = self.conv(x)

        x = x + 0.5 * self.ffn2(x)
        x = self.norm_out(x)
        return x



class Encoder(nn.Module):
    def __init__(self, input_dim, num_heads, ffn_dim, num_layers, conv_kernel, dropout, d_model=None):
        super().__init__()

        self.d_model = d_model if d_model is not None else input_dim
        self.input_proj = nn.Linear(input_dim, self.d_model) if input_dim != self.d_model else nn.Identity()
        self.pos_encoding = RelPosEncoding(self.d_model)

        self.layers = nn.ModuleList([
            ConformerBlock(self.d_model, num_heads, ffn_dim, conv_kernel, dropout)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths=None):
        x = self.dropout(self.input_proj(x))

        key_padding_mask = None
        if lengths is not None:
            B, T, _ = x.shape
            key_padding_mask = torch.arange(T, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)

        pos_enc = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x, pos_enc, key_padding_mask)

        return x, lengths