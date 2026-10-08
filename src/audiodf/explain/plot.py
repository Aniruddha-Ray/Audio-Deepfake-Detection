"""Demo picture of an explanation: spectrogram with the SHAP contribution of every time x frequency region, and the window scores."""

from __future__ import annotations

import numpy as np


def draw(x: np.ndarray, sr: int, masker, values: np.ndarray, windows: list, facts: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    from audiodf.explain.attribution import BANDS

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), height_ratios=[3, 1], sharex=True)
    ax1.specgram(x + 1e-9, NFFT=512, Fs=sr, noverlap=384, cmap="gray_r")
    lim = max(float(np.abs(values).max()), 1e-6)
    for r, v in zip(masker.regions, values):
        lo, hi, _ = BANDS[r.index % len(BANDS)]
        color = plt.cm.RdBu_r(0.5 + 0.5 * v / lim)
        ax1.add_patch(Rectangle((r.t0, lo), r.t1 - r.t0, hi - lo, facecolor=color, alpha=0.45, edgecolor="k", linewidth=0.3))
        ax1.text((r.t0 + r.t1) / 2, (lo + hi) / 2, f"{v:+.2f}", ha="center", va="center", fontsize=7)
    ax1.set_ylabel("Hz")
    v = facts["verdict"]
    ax1.set_title(f"Action: {v['action']}  |  fake score {v['fake_probability']:.2f}  |  red = pushed towards synthetic")
    mids = [(w["start_s"] + w["end_s"]) / 2 for w in windows]
    ax2.bar(mids, [w["p_fake"] for w in windows], width=0.8, color="tab:red")
    ax2.axhline(v["verify_threshold"], color="tab:orange", linestyle="--", label="verify")
    ax2.axhline(v["escalate_threshold"], color="tab:red", linestyle=":", label="escalate")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("2 s window\nP(fake)")
    ax2.set_xlabel("seconds of speech")
    ax2.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
