import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    mels, transcripts = zip( *batch)
    
    mel_lengths = torch.tensor([m.shape[0] for m in mels])
    mels_padded = pad_sequence(mels, batch_first=True, padding_value=0.0)  # (B, T_max, n_mels)
    
    chars = sorted(set( ch for t in transcripts for ch in t ))
    chartoidx = { ch: i + 1 for i, ch in enumerate(chars) }  # 0 reserved for padding
    
    encoded = [ torch.tensor([chartoidx[ch] for ch in t], dtype=torch.long) for t in transcripts ]
    transcript_lengths = torch.tensor([len(e) for e in encoded])
    transcripts_padded = pad_sequence(encoded, batch_first=True, padding_value=0)  # (B, L_max)
    
    return mels_padded, mel_lengths, transcripts_padded, transcript_lengths