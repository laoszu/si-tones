import torch
import torchaudio

class AudioProcessor:

    # for a single audio

    def __init__(self, sample_rate: int = 16000, n_mels: int = 80, n_fft: int = 512):
        self.sample_rate = sample_rate

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=int(sample_rate*0.01), # 10ms hop
        )

    def load(self, path: str) -> tuple[torch.Tensor, torch.Tensor]:
        waveform, sr = torchaudio.load(path)

        # stereo -> mono
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)

        mel = self.mel_transform(waveform)
        mel = torch.log(mel + 1e-9)
        mel = mel.squeeze(0).transpose(0, 1) # (T, n_mels)

        return mel, waveform

    def to_batch(self, path: str, device: str = 'cpu') -> tuple[torch.Tensor, torch.Tensor]:
        mel, waveform = self.load(path)
        mels = mel.unsqueeze(0).to(device)          # (1, T, n_mels)
        mel_lengths = torch.tensor([mel.size(0)]).to(device)  # (1,)
        return mels, mel_lengths