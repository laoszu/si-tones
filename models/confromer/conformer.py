
'''
Encoder + Linear Layer

'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import Encoder

class Conformer(nn.Module):
    def __init__(self, input_dim, num_heads, ffn_dim, num_layers, conv_kernel, dropout, vocab_size, d_model=None):
        super().__init__()

        self.encoder = Encoder(
            input_dim=input_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            conv_kernel=conv_kernel,
            dropout=dropout,
            d_model=d_model,
        )

        # linear head -> log_softmax (CTC)
        self.classifier = nn.Linear(self.encoder.d_model, vocab_size)

    def forward(self, x, lengths=None):
        out, lengths = self.encoder(x, lengths) # (B, T, d_model)
        logits = self.classifier(out)
        log_probs = F.log_softmax(logits, dim=-1)

        return log_probs.transpose(0, 1), lengths