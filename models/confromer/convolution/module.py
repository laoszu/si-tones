import torch
import torch.nn as nn

from models.confromer.convolution.depthwise import DepthwiseConv
from models.confromer.convolution.pointwise import PointwiseConv

class CausalLayerNorm(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x):
        # x: (B, C, T) -> transpose > (B, T, C) > LayerNorm > transpose
        return self.norm(x.transpose(1, 2)).transpose(1, 2)

class ConvolutionModule(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, exp_factor: int = 2, dropout: float = 0.1):
        super(ConvolutionModule, self).__init__()

        self.seq = nn.Sequential(
            nn.LayerNorm(in_channels),
            PointwiseConv(in_channels, in_channels * exp_factor, stride=1, padding=0, bias=True),
            nn.GLU(dim=1), 
            DepthwiseConv(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=False),
            CausalLayerNorm(in_channels), # prev: BatchNomrm1D
            nn.SiLU(), # also known as swish function
            PointwiseConv(in_channels, in_channels, stride=1, padding=0, bias=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.seq[0](x)
        out = out.transpose(1, 2)
        
        # temp. (B, C, T)
        for layer in self.seq[1:]:
            out = layer(out)

        out = out.transpose(1, 2)

        # (B,T, C)
        # batch, time, channels format
        return residual + out