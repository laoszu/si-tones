import librosa
import numpy as np
from scipy.ndimage import binary_closing, generic_filter, uniform_filter1d
from scipy.signal import medfilt

from utils.config import cfg


def f0_track(y, sr, fmin, fmax, gate):
    c = cfg["tones"]["f0"]
    hop = int(sr * c["hop_ms"] / 1000)
    f, _, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr,
                           frame_length=int(sr * c["pyin_window_ms"] / 1000), hop_length=hop)
    t = librosa.times_like(f, sr=sr, hop_length=hop)

    # no f0 where it is quiet
    rms_hop = int(sr * c["rms_hop_ms"] / 1000)
    rms = librosa.feature.rms(y=y, frame_length=int(sr * c["rms_window_ms"] / 1000), hop_length=rms_hop)[0]
    rms_t = librosa.times_like(rms, sr=sr, hop_length=rms_hop)
    f[np.interp(t, rms_t, rms) < gate * rms.max()] = np.nan

    return t, f

def f0_range_auto(y, sr, gate):
    t = cfg["tones"]
    c = t["f0"]
    f = f0_track(y, sr, *t["f0_search"], gate)[1]
    f = f[~np.isnan(f)]

    if len(f) < c["min_frames"]:
        return tuple(map(float, t["f0_default"]))

    # most common pitch (in semitones) -> range around it
    ref, step, pad = c["ref_hz"], c["hist_bin_st"], c["hist_pad_st"]
    st = 12 * np.log2(f / ref)
    edges = np.arange(st.min() - pad, st.max() + pad + step, step)
    hist = uniform_filter1d(np.histogram(st, edges)[0].astype(float), c["hist_smooth"])
    centers = (edges[:-1] + edges[1:]) / 2
    mode = ref * 2 ** (centers[hist.argmax()] / 12)

    return max(t["f0_limits"][0], mode / c["range_down"]), min(t["f0_limits"][1], mode * c["range_up"])

def runs(mask):
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))

def clean(f, fmin, fmax):
    c = cfg["tones"]["f0"]
    f = f.copy()

    # edges of the range are usually errors
    f[(f < fmin * c["edge_low"]) | (f > fmax * c["edge_high"])] = np.nan

    # octave jumps
    st = 12 * np.log2(f / c["ref_hz"])
    local_median = generic_filter(st, np.nanmedian, size=c["median_size"], mode="nearest")
    f[np.abs(st - local_median) > c["octave_jump_st"]] = np.nan

    out = np.full_like(f, np.nan)
    for a, b in runs(~np.isnan(f)):
        if b - a >= c["min_run"]:
            k = c["medfilt_kernel"]
            out[a:b] = medfilt(f[a:b], k if b - a >= k else c["medfilt_short"])

    return out

def longest_run(mask):
    n = cfg["tones"]["f0"]["closing_size"]
    out = np.zeros_like(mask)
    found = runs(binary_closing(np.pad(mask, n), np.ones(n))[n:-n])

    if found:
        a, b = max(found, key=lambda r: r[1] - r[0])
        out[a:b] = mask[a:b]

    return out