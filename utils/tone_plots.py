import librosa
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

from utils.config import cfg
from utils.tone_text import bold_font

def tone_color(tone):
    # same colors as in pleco
    return cfg["tones"]["colors"][str(tone)]

def make_axes(width, height, ylabel, start, end, bounds, tones, title_lines=0, tint=True):
    plot = cfg["tones"]["plot"]
    m = plot["margins"]
    fig, ax = plt.subplots(figsize=(width, height))
    fig.subplots_adjust(left=m["left"], right=m["right"], bottom=m["bottom"],
                        top=m["top"] - plot["title_line_height"] * title_lines)

    ax.set_xlim(start, end)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)

    for (a, b), tone in zip(bounds, tones):
        ax.axvline(a, color=plot["guide_color"], lw=plot["guide_lw"], ls="--")
        if tint:
            ax.axvspan(a, b, color=tone_color(tone), alpha=plot["tint_alpha"], lw=0)
    ax.axvline(bounds[-1][1], color=plot["guide_color"], lw=plot["guide_lw"], ls="--")

    return fig, ax

def plot_waveform(y_raw, sr, start, end, bounds, chars, pinyin, tones, font, width, pps):
    plot = cfg["tones"]["plot"]
    per_line = cfg["tones"]["chars_per_line"]
    height = plot["waveform_height"]
    lines = ["".join(chars)[k:k + per_line] for k in range(0, len(chars), per_line)]

    fig, ax = make_axes(width, height, "Amplitude", start, end, bounds, tones, len(lines))
    ax.set_title("\n".join(lines), loc="left", fontproperties=bold_font(font, plot["title_size"]))

    amp = np.abs(y_raw).max()
    ax.plot(np.arange(len(y_raw)) / sr + start, y_raw, color=plot["wave_color"], lw=plot["wave_lw"])
    ax.set_ylim(plot["wave_ylim"][0] * amp, plot["wave_ylim"][1] * amp)
    ax.set_yticks([-amp, 0, amp])
    ax.set_yticklabels(["-1", "0", "1"])

    # hanzi and pinyin above the waveform
    transform = ax.get_xaxis_transform()
    axes_height = height * (plot["waveform_axes_frac"] - plot["title_line_height"] * len(lines))

    for (a, b), char, syllable, tone in zip(bounds, chars, pinyin, tones):
        width_pt = (b - a) * pps
        x = (a + b) / 2
        y_pos = plot["label_y"]
        color = tone_color(tone)

        size = float(np.clip(width_pt * plot["hanzi_width_scale"], *plot["hanzi_size_waveform"]))
        ax.text(x, y_pos, char, fontproperties=bold_font(font, size), transform=transform,
                ha="center", va="top", color=color)
        y_pos -= size * plot["hanzi_line_spacing"] / 72 / axes_height

        # rotate pinyin if it doesn't fit
        label = f"{syllable}{tone if tone != 5 else ''}"
        fit = width_pt / (plot["pinyin_char_width"] * len(label))
        if fit >= plot["pinyin_min_fit"]:
            rotation, size = 0, min(plot["pinyin_size"], fit)
        else:
            rotation = 90
            size = float(np.clip(width_pt * plot["hanzi_width_scale"], *plot["pinyin_size_rotated"]))
        ax.text(x, y_pos, label, transform=transform, ha="center", va="top", color=color,
                fontweight="bold", rotation=rotation, fontsize=size)

    return fig

def plot_spectrogram(y, sr, start, end, bounds, chars, tones, t, f0, f, segs, f0_limits, font,
                     width, pps):
    plot = cfg["tones"]["plot"]
    f0_color = cfg["tones"]["f0_color"]
    halo = plot["halo_color"]
    fig, ax = make_axes(width, plot["spectrogram_height"], "Frequency [Hz]", start, end, bounds, tones, tint=False)

    # pre-emphasis
    emphasized = np.append(y[0], y[1:] - plot["preemphasis"] * y[:-1])
    S = np.abs(librosa.stft(emphasized, n_fft=cfg["n_fft"], win_length=int(plot["spec_window_ms"] / 1000 * sr),
                            hop_length=int(plot["spec_hop_ms"] / 1000 * sr))) ** 2
    db = 10 * np.log10(S + plot["spec_log_eps"])
    max_hz = plot["spec_max_hz"]
    db = np.clip(db, db.max() - plot["spec_range_db"], None)[librosa.fft_frequencies(sr=sr, n_fft=cfg["n_fft"]) <= max_hz]
    ax.imshow(db, origin="lower", aspect="auto", cmap="Greys",
              extent=[start, start + len(y) / sr, 0, max_hz])

    # space on top for hanzi
    ax_f0 = ax.twinx()
    ax_f0.set_ylim(f0_limits[0] * plot["f0_ylim_scale"][0], f0_limits[1] * plot["f0_ylim_scale"][1])
    ax_f0.set_ylabel("F0 [Hz]", color=f0_color)
    ax_f0.tick_params(axis="y", colors=f0_color)
    ax_f0.plot(t, f0, color=f0_color, lw=plot["f0_lw"], alpha=plot["f0_alpha"],
               path_effects=[pe.Stroke(linewidth=plot["f0_halo_lw"], foreground=halo), pe.Normal()])

    for (a, b), char, tone, mask in zip(bounds, chars, tones, segs):
        if mask.sum() <= 1:
            continue

        color = tone_color(tone)
        ax_f0.plot(t[mask], f[mask], color=color, lw=plot["seg_lw"],
                   path_effects=[pe.Stroke(linewidth=plot["seg_halo_lw"], foreground=halo), pe.Normal()])

        size = float(np.clip((b - a) * pps * plot["hanzi_width_scale"], *plot["hanzi_size_spectrogram"]))
        ax_f0.annotate(char, (t[mask].mean(), np.nanmax(f[mask])), xytext=(0, plot["annot_offset"]),
                       textcoords="offset points", fontproperties=bold_font(font, size),
                       ha="center", va="bottom", color=color,
                       path_effects=[pe.Stroke(linewidth=plot["annot_halo_lw"], foreground=halo), pe.Normal()])

    return fig