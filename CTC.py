import torch

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
    def __init__(self, input_dims=80, hidden_dims=256, num_layers=30):
        super(CTC, self).__init__()

        pass

