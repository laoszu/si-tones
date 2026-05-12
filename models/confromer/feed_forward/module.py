import torch
import torch.nn as nn

class FeedForwardModule(nn.Module):
    def __init__(self, in_features, ffn_dim=None, exp_factor=4, dropout=0.1):
        super(FeedForwardModule, self).__init__()

        self.hidden = ffn_dim if ffn_dim is not None else in_features*exp_factor

        # as defined in the article, expansion factpr will be 4
        self.seq = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, self.hidden, bias=True),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden, in_features, bias=True),
            nn.Dropout(dropout)
        )
        
    
    def forward(self, x):
        return self.seq(x)