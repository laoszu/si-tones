import torch
import torch.nn as nn
import torchaudio

class CTC(torch.nn.Module):
    '''
        Lightweight solution for audio's speech classification.
        CTC stands for Connectionst Temporal Classification

        Consists of:
            - CNN (feature extraction)
            - Bidirectional LSTM (capturing sequences in the speech)
            - Linear (final classification)
        
        For now, only pinyin; Mapping into the hanzi will be implemented later.
    '''
    def __init__(self, input_dims=80, hidden_dims=256, num_layers=4, num_classes=None, dropout=0.3):
        super(CTC, self).__init__()

        # cnn as the feature extractor
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dims, hidden_dims, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dims),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Conv1d(hidden_dims, hidden_dims, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dims),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # bi-lstm as the sequence capturer
        self.lstm = nn.LSTM(
            input_size=hidden_dims,
            hidden_size=hidden_dims,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.layer_norm = nn.LayerNorm(hidden_dims * 2)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dims * 2, num_classes)

    def forward(self, x):
        # x is (batch, time, features)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)

        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.linear(x)

        return torch.log_softmax(x, dim=-1)