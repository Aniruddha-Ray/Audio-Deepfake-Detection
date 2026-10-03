"""Energy VAD used identically in training and serving.

ASVspoof5 train bonafide clips carry ~0.2-0.26 s of leading/trailing silence against ~0.03-0.05 s
for spoof, so raw clips teach "silence at the edges means real". Training trims both edges of every
clip (both classes); CallSession skips leading silence before speech starts. Internal pauses are kept.
Threshold -45 dBFS sits above both classes' noise floor (p90 -52.5 / -49.2 dBFS) and ~17 dB below
the quietest speech (p5 -27.6 dBFS), measured on 800 ASVspoof5 train clips.
"""

from __future__ import annotations

import numpy as np

from audiodf.config import VadConfig


def frame_db(wave: np.ndarray, frame: int) -> np.ndarray:
    n = len(wave) // frame
    if n == 0:
        return np.zeros(0)
    return 10 * np.log10((wave[: n * frame].reshape(n, frame).astype(np.float64) ** 2).mean(axis=1) + 1e-12)


def speech_bounds(wave: np.ndarray, sample_rate: int, cfg: VadConfig | None = None) -> tuple[int, int] | None:
    """(start, end) sample indices of the audio from the first to the last loud frame, widened by the
    margin; None if no frame is loud enough."""
    cfg = cfg or VadConfig()
    frame = int(cfg.frame_seconds * sample_rate)
    loud = np.nonzero(frame_db(wave, frame) > cfg.threshold_db)[0]
    if not len(loud):
        return None
    margin = int(cfg.margin_seconds * sample_rate)
    return max(0, loud[0] * frame - margin), min(len(wave), (loud[-1] + 1) * frame + margin)


def trim_silence(wave: np.ndarray, sample_rate: int, cfg: VadConfig | None = None) -> np.ndarray:
    """Drop leading and trailing silence; returns an empty array when there is no speech."""
    bounds = speech_bounds(wave, sample_rate, cfg)
    return wave[:0] if bounds is None else wave[bounds[0]:bounds[1]]
