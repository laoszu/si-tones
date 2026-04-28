import torch
import torchaudio
import torch.nn as nn

class FeedForwardModule(nn.Module):
    def __init__(self, in_features, kernel_size, exp_factor=4, dropout=0.01):
        super(FeedForwardModule, self).__init__()

        # as defined in the article, expansion factpr will be 4
        self.seq = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, in_features*exp_factor, bias=True),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features*exp_factor, in_features, bias=True),
            nn.Dropout(dropout)
        )
        
    
    def forward(self, x):
        return self.seq(x)