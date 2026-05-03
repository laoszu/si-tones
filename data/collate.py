import torch
from torch.nn.utils.rnn import pad_sequence

from utils.tokenizer import Tokenizer

tokenizer = Tokenizer()
tokenizer.load_vocab("vocab.json")

def collate_fn(batch):
    mels, transcripts = zip(*batch)

    # (B, T_max, n_mels)
    mel_lengths = torch.tensor([m.shape[0] for m in mels])
    mels_padded = pad_sequence(mels, batch_first=True, padding_value=0.0)

    # chars -> pintin -> indexes
    encoded = [
        torch.tensor(tokenizer.encode(t), dtype=torch.long)
        for t in transcripts
    ]

    transcript_lengths = torch.tensor([len(e) for e in encoded])
    transcripts_padded = pad_sequence(
        encoded, batch_first=True, padding_value=tokenizer.char_to_id["<blank>"]
    )

    return mels_padded, mel_lengths, transcripts_padded, transcript_lengths