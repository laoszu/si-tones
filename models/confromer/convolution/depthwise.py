import torch
import torchaudio
import torch.nn as nn

class DepthwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(DepthwiseConv, self).__init__()

        # spatial filtering
        self.conv = nn.Conv1d (
            in_channels=in_channels,
            out_channels=out_channels,

            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
    
    def forward(self, x):
        return self.conv(x)