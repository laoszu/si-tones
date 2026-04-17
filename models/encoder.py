import torch
import torch.nn as nn

class ConvSubsampler(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # 2x time shorter sequence
        # save energy, save patience
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)

        return x.transpose(1, 2)


class Encoder(nn.Module):
    def __init__(self, input_dims=80, hidden_dims=256, num_layers=4, dropout=0.3):
        super().__init__()
        self.subsampler = ConvSubsampler(input_dims, hidden_dims)

        self.layers = nn.ModuleList([
            nn.LSTM(
                input_size=hidden_dims * 2 if i > 0 else hidden_dims,
                hidden_size=hidden_dims,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            for i in range(num_layers)
        ])
        
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dims * 2) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.subsampler(x)  # (B, T//2, hidden_dims)

        for lstm, norm in zip(self.layers, self.norms):
            residual = x

            out, _ = lstm(x)  # (B, T//2, hidden_dims*2)
            out = self.dropout(out)
            out = norm(out)

            # residual only when shapes match (from layer 1 onward)
            if out.shape == residual.shape:
                out = out + residual
            x = out

        return x