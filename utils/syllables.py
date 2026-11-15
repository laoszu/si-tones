import librosa
import numpy as np

from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, find_peaks, sosfiltfilt

from utils.config import cfg

def segment(y, sr, n, voiced, tv, offset):
    c = cfg["tones"]["syllables"]
    neg_inf = c["neg_inf"]
    hop = c["hop_ms"] / 1000
    margin = c["grid_margin"]
    band = sosfiltfilt(butter(c["band_order"], c["band_hz"], "bandpass", fs=sr, output="sos"), y)
    rms = librosa.feature.rms(y=band, frame_length=int(sr * c["rms_window_ms"] / 1000), hop_length=int(sr * hop))[0]
    db = uniform_filter1d(20 * np.log10(rms + c["db_eps"]), c["smooth"])
    t = np.arange(len(db)) * hop

    active = db > np.percentile(db, c["peak_percentile"]) - c["active_db_below_peak"]
    first, last = np.where(active)[0][[0, -1]]

    if n == 1:
        return [(offset + t[first], offset + t[last])]

    # candidates for boundaries: energy minima + grid
    minima, props = find_peaks(-db[first:last + 1], prominence=c["prominence"])
    strength = np.zeros(len(t))
    strength[first + minima] = np.minimum(props["prominences"], c["prominence_cap"])
    grid = set(range(first + margin, last - margin + 1, c["grid_step"])) | set(map(int, first + minima))
    cand = np.array([first] + sorted(g for g in grid if first + margin <= g <= last - margin) + [last])
    K = len(cand)

    # score of a syllable from candidate i to k
    active_time = np.cumsum(active) * hop
    voiced_frames = np.cumsum(np.interp(t, tv, voiced.astype(float)) > c["voiced_threshold"])
    dur = active_time[cand][None] - active_time[cand][:, None]
    mean_dur = max((active_time[last] - active_time[first]) / n, c["min_dur"])

    score = -c["dur_weight"] * np.log(np.maximum(dur, c["floor_dur"]) / mean_dur) ** 2 / (2 * c["dur_sigma"] ** 2)
    score[dur < c["min_dur"]] = neg_inf
    score[(voiced_frames[cand][None] - voiced_frames[cand][:, None]) < c["min_voiced_frames"]] -= c["unvoiced_penalty"]
    score[np.tri(K, dtype=bool)] = neg_inf
    boundary = np.where(np.arange(K) < K - 1, strength[cand], 0)

    # viterbi
    dp = np.full((n + 1, K), c["dp_init"])
    back = np.zeros((n + 1, K), int)
    dp[1] = score[0] + boundary

    for j in range(2, n + 1):
        total = dp[j - 1][:, None] + score
        back[j] = total.argmax(0)
        dp[j] = total[back[j], np.arange(K)] + boundary

    path = [K - 1]
    for j in range(n, 1, -1):
        path.append(back[j][path[-1]])
    idx = [int(cand[k]) for k in (path + [0])[::-1]]

    # boundary in a pause goes to the middle of it
    def boundary_time(i):
        if active[i]:
            return t[i]

        left = right = i
        while left > 0 and not active[left - 1]:
            left -= 1
        while right < len(t) - 1 and not active[right + 1]:
            right += 1

        return t[(left + right) // 2]

    edges = [t[idx[0]]] + [boundary_time(i) for i in idx[1:-1]] + [t[idx[-1]]]

    return [(offset + edges[i], offset + edges[i + 1]) for i in range(n)]
