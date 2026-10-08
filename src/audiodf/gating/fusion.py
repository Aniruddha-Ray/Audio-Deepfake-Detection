"""The gate: the weight of the Whisper branch from the estimated channel quality, and the choice of its two parameters."""

from __future__ import annotations

import numpy as np

from audiodf.evaluation.metrics import compute_metrics

W_MIN_GRID = (0.0, 0.1, 0.2)
W_MAX_GRID = (0.5, 0.65, 0.8)  # never above 0.8: neither model is ever switched off


def gate_weight(q: np.ndarray, w_min: float, w_max: float) -> np.ndarray:
    """Whisper's weight: w_min on a clean call (q = 0) up to w_max on a degraded one (q = 1)."""
    return w_min + (w_max - w_min) * np.clip(np.asarray(q, dtype=float), 0.0, 1.0)


def gated_fuse(p_wavlm: np.ndarray, p_whisper: np.ndarray, q: np.ndarray, w_min: float, w_max: float) -> np.ndarray:
    """(1 - w) * P_WavLM + w * P_Whisper, with w from the call's q; p_* are (n,) or (n, T), q is (n,)."""
    w = gate_weight(q, w_min, w_max)
    if np.ndim(p_wavlm) == 2:
        w = w[:, None]
    return (1 - w) * np.asarray(p_wavlm, dtype=float) + w * np.asarray(p_whisper, dtype=float)


def tune_gate(p_wavlm: np.ndarray, p_whisper: np.ndarray, q: np.ndarray, label: np.ndarray, tie: float = 0.05,
              w_min_grid=W_MIN_GRID, w_max_grid=W_MAX_GRID) -> tuple[tuple[float, float], dict]:
    """(w_min, w_max) minimising the mean EER over every time-to-decision (the columns of p_*), as the fusion weights in training;
    settings within `tie` EER points of the best are equal and the one with the smaller Whisper weight wins. Returns the choice and
    {"w_min=.. w_max=..": mean EER} for every setting."""
    table = {}
    for lo in w_min_grid:
        for hi in w_max_grid:
            if hi < lo:
                continue
            fused = gated_fuse(p_wavlm, p_whisper, q, lo, hi)
            table[(lo, hi)] = float(np.mean([compute_metrics(label, fused[:, k])["eer_pct"] for k in range(fused.shape[1])]))
    floor = min(table.values())
    near = [k for k, v in table.items() if v <= floor + tie]
    best = min(near, key=lambda k: (k[0] + k[1], k))
    return best, {f"w_min={lo:g} w_max={hi:g}": round(v, 3) for (lo, hi), v in table.items()}
