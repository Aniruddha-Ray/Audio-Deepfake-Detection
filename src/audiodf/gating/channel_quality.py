"""Channel-quality estimate of a call from its audio: is there room echo, is there background noise?

Hand-computed features of the first 10 s of speech (energy percentiles and spread, spectral shape of the quietest and loudest frames,
how fast the energy falls after speech, the modulation spectrum of the envelope) feed two small gradient-boosted classifiers, "echo" and
"noisy" (SNR below 20 dB). They are fitted on training copies whose echo and noise are known from the impairment plan, a label that has
nothing to do with real / fake (the plan is drawn from the clip ID and seed only). q = max(P(echo), P(noisy)) in [0, 1].

The gate must not be a detector in disguise: `leak_auc` measures how well q alone separates fake from real inside a set, and the
result report of new_plan.md 7.3v lists it for every test set (it should stay near chance).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

SR = 16000
FRAME, HOP, NFFT = 400, 160, 512
SECONDS = 10.0
NOISY_SNR_DB = 20.0
FEATURE_VERSION = 1  # a change in the features below must bump this (stored with the model)
FEATURE_NAMES = (
    [f"e_p{p}" for p in (5, 10, 25, 50, 75, 90, 95)] + ["e_p90_p10", "e_p90_p50", "e_p50_p10", "active_10db", "active_20db", "active_35db"]
    + [f"{grp}_{f}" for grp in ("quiet", "loud") for f in ("flatness", "centroid_khz", "low500", "high3k")]
    + [f"slope_p{p}" for p in (1, 5, 10, 25, 75, 90, 95)] + ["slope_neg_mean", "slope_pos_mean"]
    + ["mod_2_8", "mod_8_16", "mod_0p5_2"] + ["ltas_slope_db_per_khz", "hf_ratio_4k"])


def _frames(x: np.ndarray) -> np.ndarray:
    if len(x) < FRAME + 10 * HOP:  # a very short clip: pad with silence so every statistic exists
        x = np.pad(x, (0, FRAME + 10 * HOP - len(x)))
    return np.lib.stride_tricks.sliding_window_view(x, FRAME)[::HOP]


def _spectral(power: np.ndarray) -> list[float]:
    """Flatness, centroid (kHz), share below 500 Hz, share above 3 kHz of the mean power spectrum of some frames."""
    mean = power.mean(axis=0) + 1e-12
    freqs = np.fft.rfftfreq(NFFT, 1 / SR)
    flat = float(np.exp(np.log(mean).mean()) / mean.mean())
    return [flat, float((freqs * mean).sum() / mean.sum() / 1000), float(mean[freqs < 500].sum() / mean.sum()),
            float(mean[freqs >= 3000].sum() / mean.sum())]


def channel_features(x: np.ndarray, seconds: float = SECONDS) -> np.ndarray:
    """(len(FEATURE_NAMES),) float32 feature vector of audio x (16 kHz, speech from the first sample)."""
    x = np.asarray(x, dtype=np.float32)[: int(seconds * SR)]
    fr = _frames(x)
    e = 10 * np.log10((fr.astype(np.float64) ** 2).mean(axis=1) + 1e-10)
    pc = np.percentile(e, [5, 10, 25, 50, 75, 90, 95])
    top = pc[-1]
    feats = list(pc) + [pc[5] - pc[1], pc[5] - pc[3], pc[3] - pc[1]] + [float((e > top - d).mean()) for d in (10, 20, 35)]
    power = np.abs(np.fft.rfft(fr * np.hanning(FRAME), NFFT, axis=1)) ** 2
    order = np.argsort(e)
    k = max(len(e) // 4, 3)
    feats += _spectral(power[order[:k]]) + _spectral(power[order[-k:]])
    d = np.diff(e)
    feats += list(np.percentile(d, [1, 5, 10, 25, 75, 90, 95])) + [float(d[d < 0].mean()) if (d < 0).any() else 0.0,
                                                                  float(d[d > 0].mean()) if (d > 0).any() else 0.0]
    env = np.sqrt((fr.astype(np.float64) ** 2).mean(axis=1))
    env = env - env.mean()
    ms = np.abs(np.fft.rfft(env)) ** 2
    mf = np.fft.rfftfreq(len(env), HOP / SR)
    total = ms[(mf >= 0.5) & (mf < 32)].sum() + 1e-12
    feats += [float(ms[(mf >= lo) & (mf < hi)].sum() / total) for lo, hi in ((2, 8), (8, 16), (0.5, 2))]
    ltas = 10 * np.log10(power.mean(axis=0) + 1e-12)
    freqs = np.fft.rfftfreq(NFFT, 1 / SR) / 1000
    sel = (freqs >= 0.3) & (freqs <= 7.5)
    feats += [float(np.polyfit(freqs[sel], ltas[sel], 1)[0]), float(power[:, freqs >= 4].sum() / (power.sum() + 1e-12))]
    return np.asarray(feats, dtype=np.float32)


class ChannelQualityModel:
    """Two classifiers on `channel_features`: echo and noisy. predict() -> (P(echo), P(noisy), q)."""

    def __init__(self):
        make = lambda: HistGradientBoostingClassifier(max_depth=4, learning_rate=0.1, max_iter=150, random_state=0)  # noqa: E731
        self.echo, self.noisy = make(), make()
        self.version = FEATURE_VERSION

    def fit(self, x: np.ndarray, echo: np.ndarray, snr_db: np.ndarray) -> "ChannelQualityModel":
        """echo: bool per clip; snr_db: SNR of the added noise (nan = none), so a clip is "noisy" below NOISY_SNR_DB."""
        self.echo.fit(x, np.asarray(echo, dtype=int))
        self.noisy.fit(x, (np.nan_to_num(np.asarray(snr_db, dtype=float), nan=99.0) < NOISY_SNR_DB).astype(int))
        return self

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pe, pn = self.echo.predict_proba(x)[:, 1], self.noisy.predict_proba(x)[:, 1]
        return pe, pn, np.maximum(pe, pn)

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "ChannelQualityModel":
        m = joblib.load(path)
        if m.version != FEATURE_VERSION:
            raise ValueError(f"channel-quality model built with feature version {m.version}, code is {FEATURE_VERSION}")
        return m


def leak_auc(q: np.ndarray, label: np.ndarray) -> float:
    """AUC of q as a score for fake (label 1) against real: 0.5 means q says nothing about real / fake."""
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(label, q)) if len(set(label)) == 2 else float("nan")
