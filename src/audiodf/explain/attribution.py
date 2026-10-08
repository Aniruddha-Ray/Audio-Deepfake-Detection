"""SHAP attributions of the served model's score over time x frequency regions of the call.

The model reads raw 16 kHz audio, so per-sample attributions would mean nothing to a person. The explained audio (the first 10 s of speech, the
decision horizon used everywhere in this project) is cut into regions: 2 s time slices x 4 frequency bands (at most 5 x 4 = 20). A region is
"removed" by attenuating its cells in the short-time spectrum by 30 dB and resynthesising the audio; the model then scores the audio exactly as the
serving path does (2 s windows, 1 s hop, mean of the window probabilities, fused with the bundle's weights). KernelSHAP over the regions gives each
one's contribution to the fake score, relative to the audio with every region removed; the contributions add up to that difference.

`faithfulness` checks the attribution: removing the top regions must lower the score more than removing as many random ones.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import istft, stft

BANDS = ((0, 300, "below 300 Hz"), (300, 1000, "300 Hz-1 kHz"), (1000, 3000, "1-3 kHz"), (3000, 8000, "3-8 kHz"))
SLICE_S = 2.0
FLOOR_DB = -30.0
N_FFT, HOP = 512, 128


@dataclass(frozen=True)
class Region:
    index: int
    t0: float
    t1: float
    band: str

    @property
    def label(self) -> str:
        return f"{self.t0:.0f}-{self.t1:.0f} s, {self.band}"


class RegionMasker:
    """Removes (attenuates) any set of time x frequency regions of one waveform."""

    def __init__(self, wave: np.ndarray, sr: int = 16000, slice_s: float = SLICE_S, bands=BANDS, floor_db: float = FLOOR_DB):
        self.wave, self.sr = np.asarray(wave, dtype=np.float32), sr
        self.freqs, self.times, self.spec = stft(self.wave, sr, nperseg=N_FFT, noverlap=N_FFT - HOP, boundary="even", padded=True)
        n_slices = max(1, int(np.ceil(len(self.wave) / (slice_s * sr))))
        self.regions = [Region(k * len(bands) + b, k * slice_s, min((k + 1) * slice_s, len(self.wave) / sr), bands[b][2])
                        for k in range(n_slices) for b in range(len(bands))]
        # every spectrum cell belongs to exactly one region: frames by time slice, bins by band (the top band keeps the Nyquist bin)
        frame_slice = np.minimum((self.times // slice_s).astype(int), n_slices - 1)
        bin_band = np.minimum(np.searchsorted([b[1] for b in bands], self.freqs, side="right"), len(bands) - 1)
        self._cells = [(bin_band == r.index % len(bands), frame_slice == r.index // len(bands)) for r in self.regions]
        self.gain_off = 10 ** (floor_db / 20)

    @property
    def n(self) -> int:
        return len(self.regions)

    def apply(self, keep: np.ndarray) -> np.ndarray:
        """Audio with the regions where keep is 0 removed (keep: (n,) of 0 / 1)."""
        keep = np.asarray(keep).astype(bool)
        if keep.all():
            return self.wave.copy()
        gain = np.ones(self.spec.shape, dtype=np.float32)
        for k, (fsel, tsel) in enumerate(self._cells):
            if not keep[k]:
                gain[np.ix_(fsel, tsel)] = self.gain_off
        _, x = istft(self.spec * gain, self.sr, nperseg=N_FFT, noverlap=N_FFT - HOP, boundary=True)
        x = x[: len(self.wave)]
        return np.pad(x, (0, len(self.wave) - len(x))).astype(np.float32)


def window_starts(n_samples: int, seg: int, hop: int) -> list[int]:
    return list(range(0, n_samples - seg + 1, hop)) if n_samples >= seg else [0]


def make_scorer(engine, seg: int, hop: int, batch: int = 32):
    """f(list of waveforms) -> (n,) fake score, as the serving path computes it: 2 s windows every `hop`, mean of each branch's window
    probabilities, fused with the engine's weights. Window branches only (the SVM reads the whole buffer and is not explained here)."""
    if getattr(engine, "uses_svm", False):
        raise ValueError("explanations cover the window branches only; this bundle uses the SVM")

    def score(waves) -> np.ndarray:
        segs, owner = [], []
        for k, w in enumerate(waves):
            w = np.asarray(w, dtype=np.float32)
            if len(w) < seg:
                w = np.pad(w, (0, seg - len(w)))
            for s in window_starts(len(w), seg, hop):
                segs.append(w[s:s + seg])
                owner.append(k)
        segs, owner = np.stack(segs), np.asarray(owner)
        probs = {}
        for i in range(0, len(segs), 4096):
            for b, p in engine.window_probabilities(segs[i:i + 4096]).items():
                probs.setdefault(b, []).append(np.asarray(p, dtype=float))
        out = np.zeros(len(waves))
        for b, parts in probs.items():
            p = np.concatenate(parts)
            out += engine.weights.get(b, 0.0) * np.bincount(owner, weights=p, minlength=len(waves)) / np.bincount(owner, minlength=len(waves))
        return out

    return score


def window_scores(score_windows, wave: np.ndarray, seg: int, hop: int, sr: int) -> list[dict]:
    """P(fake) of each 2 s window of the explained audio (the timeline the analyst sees)."""
    starts = window_starts(len(wave), seg, hop)
    vals = [float(score_windows([wave[s:s + seg]])[0]) for s in starts]
    return [{"start_s": round(s / sr, 1), "end_s": round(min(s + seg, max(len(wave), seg)) / sr, 1), "p_fake": round(v, 4)} for s, v in zip(starts, vals)]


def shap_regions(masker: RegionMasker, score, nsamples: int = 160, seed: int = 0) -> dict:
    """KernelSHAP over the masker's regions. Returns base (all removed), full (nothing removed) scores and one value per region."""
    import shap

    def f(z: np.ndarray) -> np.ndarray:
        return score([masker.apply(row) for row in z])

    np.random.seed(seed)
    explainer = shap.KernelExplainer(f, np.zeros((1, masker.n)))
    values = np.asarray(explainer.shap_values(np.ones((1, masker.n)), nsamples=nsamples, silent=True)).reshape(-1)
    full, base = float(f(np.ones((1, masker.n)))[0]), float(np.asarray(explainer.expected_value).reshape(-1)[0])
    return {"base": base, "full": full, "values": values}


def faithfulness(masker: RegionMasker, score, values: np.ndarray, k: int = 3, n_random: int = 20, seed: int = 0) -> dict:
    """Score drop when the k regions with the largest positive contribution are removed, against the mean drop for k random regions."""
    full = float(score([masker.wave])[0])
    pos = np.argsort(-values)[:k]
    pos = pos[values[pos] > 0]
    if len(pos) == 0:
        return {"checked": False, "reason": "no region pushes the score towards fake"}
    keep = np.ones(masker.n)
    keep[pos] = 0
    drop_top = full - float(score([masker.apply(keep)])[0])
    rng = np.random.default_rng(seed)
    rand = []
    for _ in range(n_random):
        keep = np.ones(masker.n)
        keep[rng.choice(masker.n, len(pos), replace=False)] = 0
        rand.append(full - float(score([masker.apply(keep)])[0]))
    drop_random = float(np.mean(rand))
    return {"checked": True, "k": int(len(pos)), "drop_top": round(drop_top, 4), "drop_random": round(drop_random, 4),
            "faithful": bool(drop_top > drop_random)}
