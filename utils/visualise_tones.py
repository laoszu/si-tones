import warnings

import librosa
import numpy as np
from matplotlib import font_manager as fm
from scipy.signal import butter, sosfiltfilt

from utils.config import cfg
from utils.f0 import clean, f0_range_auto, f0_track, longest_run
from utils.syllables import segment
from utils.tone_plots import plot_spectrogram, plot_waveform
from utils.tone_text import find_cjk_font, parse

warnings.filterwarnings("ignore", category=RuntimeWarning)


def visualize_tones(wav_path, transcript, start_sec=0., end_sec=None, sr=None, syllable_bounds=None,
                    f0_range=None, gate=None, edge_trim=None, font_path=None, save_prefix=None):
    # returns waveform and spectrogram figures
    tones_cfg = cfg["tones"]
    plot = tones_cfg["plot"]
    sr = sr or cfg["sample_rate"]
    gate = tones_cfg["gate"] if gate is None else gate
    edge_trim = tones_cfg["edge_trim"] if edge_trim is None else edge_trim

    y_full = librosa.load(wav_path, sr=sr, mono=True)[0]
    end = min(end_sec or len(y_full) / sr, len(y_full) / sr)
    y_raw = y_full[int(start_sec * sr):int(end * sr)]

    # highpass removes hum
    y = sosfiltfilt(butter(tones_cfg["highpass_order"], tones_cfg["highpass_hz"], "highpass", fs=sr, output="sos"), y_raw)

    chars, pinyin, tones = parse(transcript)
    font = fm.FontProperties(fname=font_path) if font_path else find_cjk_font()

    fmin, fmax = f0_range or f0_range_auto(y, sr, gate)
    t, f0 = f0_track(y, sr, fmin, fmax, gate)
    f0 = clean(f0, fmin, fmax)
    t = t + start_sec
    bounds = syllable_bounds or segment(y, sr, len(chars), ~np.isnan(f0), t - start_sec, start_sec)

    # cut edges where two voiced syllables touch
    f = f0.copy()
    w = tones_cfg["edge_window"]
    for _, b in bounds[:-1]:
        before = ~np.isnan(f0[(t >= b - w) & (t < b)])
        after = ~np.isnan(f0[(t >= b) & (t <= b + w)])
        if before.any() and after.any():
            f[abs(t - b) < edge_trim] = np.nan
    segs = [longest_run((t >= a) & (t <= b) & ~np.isnan(f)) for a, b in bounds]

    # figure size, pps = points per second
    duration = end - start_sec
    width = float(np.clip(plot["width_per_second"] * duration + plot["width_base"], *plot["width_range"]))
    pps = width * plot["usable_width"] * 72 / duration

    figs = (
        plot_waveform(y_raw, sr, start_sec, end, bounds, chars, pinyin, tones, font, width, pps),
        plot_spectrogram(y, sr, start_sec, end, bounds, chars, tones, t, f0, f, segs,
                         (fmin, fmax), font, width, pps),
    )

    if save_prefix:
        for name, fig in zip(("waveform", "spectrogram"), figs):
            fig.savefig(f"{save_prefix}_{name}.png", dpi=plot["dpi"], bbox_inches="tight")

    return figs
