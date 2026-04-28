import torch
import torchaudio
import torch.nn as nn

class PointwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride, padding, bias=True):
        super(PointwiseConv, self).__init__()

        # kernel size: 1x1
        self.conv = nn.Conv1d (
            in_channels=in_channels,
            out_channels=out_channels,

            kernel_size=1,
            stride=stride,
            padding=padding,
            bias=bias
        )
    
    def forward(self, x):
        return self.conv(x)