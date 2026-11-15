import torch
import torchaudio

from utils.config import cfg

class AudioProcessor:

    # for a single audio

    def __init__(self, sample_rate: int = cfg["sample_rate"], n_mels: int = cfg["n_mels"], n_fft: int = cfg["n_fft"]):
        self.sample_rate = sample_rate

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=int(sample_rate * cfg["hop_ms"] / 1000),
        )

    def load(self, path: str) -> tuple[torch.Tensor, torch.Tensor]:
        waveform, sr = torchaudio.load(path)

        # stereo -> mono
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)

        mel = self.mel_transform(waveform)
        mel = torch.log(mel + cfg["log_eps"])
        mel = mel.squeeze(0).transpose(0, 1) # (T, n_mels)

        return mel, waveform

    def to_batch(self, path: str, device: str = 'cpu') -> tuple[torch.Tensor, torch.Tensor]:
        mel, waveform = self.load(path)
        mels = mel.unsqueeze(0).to(device)          # (1, T, n_mels)
        mel_lengths = torch.tensor([mel.size(0)]).to(device)  # (1,)
        return mels, mel_lengths