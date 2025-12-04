import numpy as np
import pandas as pd
import librosa as lr
import librosa.display as lrd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import os
import glob

class THCHS30Dataset(Dataset):
    '''
        Processing thchs-30 dataset.
        sample_rate by default should be 16000 (16kHz, standard)
    '''
    def __init__(self, path, sample_rate=16000, n_mels=80):
        self.path = path
        self.sample_rate = sample_rate
        self.n_mels = n_mels

        self.wav_files = []
        for wav_file in os.listdir(self.path):
            if wav_file.endswith('.wav'):
                self.wav_files.append(os.path.join(self.path, wav_file))

        self.mel_spectogram_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=512,
            hop_length=160 # 10ms hop for 16khz
        )
    
    def _load_audio(self, wav_file):
        '''
            Loads the waveform and returns the mel-spectogram.
        '''
        y, sr = lr.load(wav_file, sr=self.sample_rate)

        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        if y.shape[0] > 1:
            y = torch.mean(y, dim=0, keepdim=True)
        
        mel_spectogram = self.mel_spectogram_transform(y)
        mel_spectogram = torch.log(mel_spectogram + 1e-9)
        return mel_spectogram.squeeze(0).transpose(0, 1) #  domain of (time, n_mels)

    def _load_trn(self, wav_file):
        '''
            Loads the pinyin trascription.
            Temporarily omits the characters.
        '''
        trn_file = wav_file + ".trn"

        with open(trn_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            text = lines[1].strip() if len(lines) > 1 else ""
            
        return text
    
    
    def __len__(self):
        return len(self.wav_files)
    
    def __getitem__(self, index):
        wav_file = self.wav_files[index]
        mel_spct = self._load_audio(wav_file)
        trn = self._load_trn(wav_file)

        return mel_spct, trn

