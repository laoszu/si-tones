import os
import json
import glob
import sys
import argparse
from datetime import datetime

import torch
import torchaudio
import matplotlib.pyplot as plt
import soundfile as sf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from models.confromer.conformer import Conformer
from utils.tokenizer import Tokenizer

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio', default='examples/chinese_granny.mp3')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--config', default='checkpoints/config.json')
    parser.add_argument('--out', default='results')
    args = parser.parse_args()

    os.chdir(ROOT)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open(args.config) as f:
        cfg = json.load(f)

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        files = sorted(glob.glob('checkpoints/conformer_epoch_*.pt'))
        if not files:
            print('No checkpoint found')
            sys.exit(1)
        ckpt_path = files[-1]

    tokenizer = Tokenizer()
    tokenizer.load_vocab(cfg['vocab_path'])

    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    model = Conformer(
        input_dim = cfg['n_mels'],
        d_model = cfg['d_model'],
        num_heads = cfg['num_heads'],
        ffn_dim = cfg['ffn_dim'],
        num_layers = cfg['num_layers'],
        conv_kernel = cfg['conv_kernel'],
        dropout = cfg["dropout"],
        vocab_size = len(tokenizer),
    ).to(dev)
    model.load_state_dict(ckpt['model'])
    model.eval()

    epoch = ckpt.get('epoch', '?')
    mel, waveform, sr = preprocess(args.audio, cfg['sample_rate'], cfg['n_mels'])

    mels = mel.unsqueeze(0).to(dev)
    mel_lengths = torch.tensor([mel.size(0)]).to(dev)

    with torch.no_grad():
        log_probs = model(mels, mel_lengths)

    blank_id = tokenizer.char_to_id['<blank>']
    collapsed, raw_ids = greedy_decode(log_probs[0], mel_lengths[0].item(), blank_id)
    predicted = tokenizer.decode(collapsed)
    blank_frac = (raw_ids == blank_id).float().mean().item()

    audio_name = os.path.splitext(os.path.basename(args.audio))[0]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(args.out, f'{audio_name}_epoch{epoch}_{timestamp}')
    os.makedirs(out_dir, exist_ok=True)

    pred_path = os.path.join(out_dir, 'prediction.txt')
    with open(pred_path, 'w', encoding='utf-8') as f:
        f.write(f'file: {args.audio}\n')
        f.write(f'checkpoint: {ckpt_path}\n')
        f.write(f'epoch: {epoch}\n')
        f.write(f'blank_frac: {blank_frac:.2%}\n')
        f.write(f'n_tokens: {len(collapsed)}\n')
        f.write(f'\nprediction:\n{predicted if predicted else "(blank)"}\n')
    print(f'Prediction: {predicted if predicted else "(blank)"}')
    print(f'Saved: {pred_path}')

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(mel.T.numpy(), aspect='auto', origin='lower', cmap='viridis')
    fig.colorbar(ax.images[0], ax=ax, label='Amplitude')
    ax.set_title(f'Mel Spectrogram of {audio_name}')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Mel bin')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'mel.png'), dpi=150)
    plt.close()

    probs = log_probs[0].exp().cpu()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(probs[:, blank_id].numpy(), label='blank', alpha=0.7)
    ax.plot(probs.max(dim=-1).values.numpy(), label='max token', alpha=0.7)
    ax.set_title('Blank vs. max token probability')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Probability')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'probs.png'), dpi=150)
    plt.close()

    print(f'Results saved to: {out_dir}')

if __name__ == '__main__':
    main()