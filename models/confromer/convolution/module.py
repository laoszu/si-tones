import torch
import torchaudio
import torch.nn as nn

from .depthwise import DepthwiseConv
from .pointwise import PointwiseConv


class ConvolutionModule(nn.Module):
    def __init__(self, in_channels, kernel_size, exp_factor, dropout):
        super(ConvolutionModule, self).__init__()

        self.seq = nn.Sequential(
            nn.LayerNorm(),
            PointwiseConv(in_channels, in_channels*exp_factor, stride=1, padding=0, bias=True),
            nn.GLU(),
            DepthwiseConv(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.SiLU(),  # also known as swish function
            PointwiseConv(in_channels, in_channels, stride=1, padding=0, bias=True),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.seq(x)