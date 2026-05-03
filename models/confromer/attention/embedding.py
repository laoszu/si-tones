import torch
import torch.nn as nn
from torch import Tensor
import math

class RelPosEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 5000):
        super(RelPosEncoding, self).__init__()

        self.dim = dim  # embedding dimension
        self.pos_enc= None

        self._build(max_len)

    def _build(self, length: int):
        pos = torch.arange(length - 1, -length, -1, dtype=torch.float32)

        dim_idx = torch.arange(0, self.dim, 2, dtype=torch.float32)

        # PE(pos, 2i) = sin(pos / 10000^(2i/dim))
        inv_freq = 1.0 / (10000 ** (dim_idx / self.dim))

        # outer product: (2L-1, dim/2)
        sinusoid = torch.outer(pos, inv_freq)

        enc = torch.zeros(pos.size(0), self.dim)
        enc[:, 0::2] = torch.sin(sinusoid)
        enc[:, 1::2] = torch.cos(sinusoid)

        self.register_buffer("pos_enc", enc.unsqueeze(0))


    def extend(self, x : Tensor):
        T = x.size(1)
        needed = 2 * T - 1
        if self.pos_enc is None or self.pos_enc.size(1) < needed:
            self._build(T)
        self.pos_enc = self.pos_enc.to(device=x.device, dtype=x.dtype)

    def forward(self, x: Tensor) -> Tensor:
        self.extend(x)
        T = x.size(1)
        
        total = self.pos_enc.size(1)
        center = total // 2
        start = center - (T - 1)
        end = center + T

        return self.pos_enc[:, start:end, :]