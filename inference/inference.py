import os
import json
import sys
import tempfile

import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torchaudio
import soundfile as sf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from models.confromer.conformer import Conformer
from utils.tokenizer import Tokenizer

import librosa
import numpy as np

def preprocess(audio_path, sample_rate, n_mels):
    waveform, sr = sf.read(audio_path, always_2d=True)
    waveform = waveform.mean(axis=1)

    if sr != sample_rate:
        import librosa
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=sample_rate)
        sr = sample_rate

    waveform_t = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=512,
        hop_length=160,
        n_mels=n_mels,
    )(waveform_t).squeeze(0).T

    mel = (mel - mel.mean()) / (mel.std() + 1e-9)
    return mel, waveform_t, sr

def greedy_decode(lp, length, blank_id):
    raw_ids = lp.argmax(dim=-1)
    collapsed, prev = [], None
    for idx in raw_ids[:length].tolist():
        if idx != prev and idx != blank_id:
            collapsed.append(idx)
        prev = idx
    return collapsed, raw_ids

def to_b64(fig):
    buff = io.BytesIO()
    fig.savefig(buff, format='png', dpi=100, bbox_inches='tight')
    buff.seek(0)
    plt.close(fig)
    return base64.b64encode(buff.read()).decode('utf-8')


def extract_pitch(waveform_np, sr, hop_length=160):
    f0 = librosa.yin(
        waveform_np,
        fmin=librosa.note_to_hz('C2'),# ~65 Hz
        fmax=librosa.note_to_hz('C7'), # ~2093 Hz
        sr=sr,
        hop_length=hop_length,
    )

    f0[f0 < 60] = np.nan
    return f0

_model = None
_tokenizer = None
_cfg = None
_dev = None

def load_model():
    global _model, _tokenizer, _cfg, _dev

    _dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open(os.path.join(ROOT, 'config.json')) as f:
        _cfg = json.load(f)

    _tokenizer = Tokenizer()
    _tokenizer.load_vocab(_cfg['vocab_path'])

    ckpt_path = os.path.join(ROOT, 'checkpoints', 'conformer_best.pt')
    ckpt = torch.load(ckpt_path, map_location=_dev, weights_only=False)

    _model = Conformer(
        input_dim=_cfg['n_mels'],
        d_model=_cfg['d_model'],
        num_heads=_cfg['num_heads'],
        ffn_dim=_cfg['ffn_dim'],
        num_layers=_cfg['num_layers'],
        conv_kernel=_cfg['conv_kernel'],
        dropout=_cfg['dropout'],
        vocab_size=len(_tokenizer),
    ).to(_dev)
    _model.load_state_dict(ckpt['model'])
    _model.eval()

def run_inference(audio_bytes: bytes) -> str:
    if _model is None:
        load_model()

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        mel, waveform, sr = preprocess(tmp_path, _cfg['sample_rate'], _cfg['n_mels'])
    finally:
        os.unlink(tmp_path)

    mels = mel.unsqueeze(0).to(_dev)
    mel_lengths = torch.tensor([mel.size(0)]).to(_dev)

    with torch.no_grad():
        log_probs = _model(mels, mel_lengths)

    blank_id = _tokenizer.char_to_id['<blank>']
    collapsed, _ = greedy_decode(log_probs[0], mel_lengths[0].item(), blank_id)
    predicted = _tokenizer.decode(collapsed)

    wave_np = waveform.squeeze().numpy()
    f0 = extract_pitch(wave_np, sr, hop_length=160)

    t_wave = np.arange(len(wave_np)) / sr
    t_f0 = np.arange(len(f0)) * 160 / sr

    fig, ax1 = plt.subplots(figsize=(10, 3))

    ax1.plot(t_wave, wave_np, linewidth=0.4, color='steelblue', alpha=0.6, label='waveform')
    ax1.set_ylabel('Amplitude', color='steelblue')
    ax1.set_xlabel('Time (s)')

    ax2 = ax1.twinx()
    ax2.plot(t_f0, f0, linewidth=1.5, color='orangered', label='F0 (pitch)')
    ax2.set_ylabel('Frequency (Hz)', color='orangered')
    ax2.set_ylim(50, 400)

    ax1.set_title('Waveform + Pitch (F0)')
    fig.legend(loc='upper right')
    wav_img = to_b64(fig)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.imshow(mel.T.numpy(), aspect='auto', origin='lower', cmap='viridis')
    ax.set_title('Mel spectrogram')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Mel bin')
    mel_img = to_b64(fig)


    return {
        "trans": predicted if predicted else '(blank)',
        "wav": wav_img,
        "mel": mel_img
    }