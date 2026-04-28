import torch
import torchaudio
import torch.nn as nn

from convolution.module import ConvolutionModule
from feed_forward.module import FeedForwardModule

class Encoder(nn.Module):

    def __init__(self, input_dim, num_heads, ffn_dim, num_layers, conv_kernel, dropout):
        super().__init__()

        # feed forward module (1.)

        # self-attention module

        # convolution module

        # feed forward module (2.)