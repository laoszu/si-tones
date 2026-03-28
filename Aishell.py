import torch
import torchaudio
from torch.utils.data import Dataset
import os
import glob

class Aishell(Dataset):
    def __init__(self, path, sample_rate=44100, n_mels=80):
        self.path = path

        # sr should be either 16kHz or 44.1kHz
        # may depend on the subset choice
        self.sample_rate = sample_rate
        self.n_mels = n_mels

        transcript_path = os.path.join(path, "transcript/aishell_transcript_v0.8.txt")
        self.transcripts = {}
        with open(transcript_path, "r", encoding="utf-8") as f:
            while line := f.readline():
                line = line.strip()
                if line:
                    parts = line.split()
                    self.transcripts[parts[0]] = "".join(parts[1:])

        wav_dir = os.path.join(path, "wav")
        all_wavs = glob.glob(os.path.join(wav_dir, "**", "*.wav"), recursive=True)
        self.wav_files = [
            p for p in sorted(all_wavs)
            if os.path.splitext(os.path.basename(p))[0] in self.transcripts
        ]

        hop_length = int(self.sample_rate * 0.01) # 10ms hop, no matter the sample rate
        self.mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=512,
            hop_length=hop_length,
        )

    def _load_audio(self, wav_file):
        waveform, sr = torchaudio.load(wav_file)

        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)

        mel = self.mel_spectrogram_transform(waveform)
        mel = torch.log(mel + 1e-9)  # log-compression

        return mel.squeeze(0).transpose(0, 1)  # (time, ficzers)

    def _load_transcript(self, wav_file):
        utt_id = os.path.splitext(os.path.basename(wav_file))[0]
        return self.transcripts.get(utt_id, "")

    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, index):
        wav_file = self.wav_files[index]
        return self._load_audio(wav_file), self._load_transcript(wav_file)