import torch
import torch.nn as nn
from torch import Tensor
import math

class RelPosEncoding(nn.Module):
    def __init__(self, dim, max_len):
        super(RelPosEncoding, self).__init__()

        self.dim = dim  # embedding dimension
        self.pos_enc= None

        self.extend(torch.tensor(0,0).expand(1, max_len))

    def extend(self, x : Tensor):
        if self.pos_enc is not None:
            if self.pos_enc.size(1)>= x.size(1)*2 - 1:
                self.pos_enc = self.pos_enc.to(device=x.device, dtype=x.dtype)
            return
        
        pos_enc_pos = torch.zeros(x.size(1), self.dim)
        pos_enc_neg = torch.zeros(x.size(1), self.dim)

        

    def forward(self, x):
        pass