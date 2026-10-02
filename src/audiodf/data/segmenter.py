"""Cut audio into fixed windows exactly as the streaming buffer will at inference time."""

from __future__ import annotations

import numpy as np


def pad_to_length(wave: np.ndarray, length: int) -> np.ndarray:
    """Repeat-pad short audio (same behaviour as training) so at least one window fits."""
    if len(wave) == 0:
        return np.zeros(length, dtype=np.float32)
    if len(wave) >= length:
        return wave
    return np.tile(wave, int(np.ceil(length / len(wave))))[:length]


def segment(wave: np.ndarray, seg_samples: int, hop_samples: int) -> np.ndarray:
    """Windows of seg_samples every hop_samples; a trailing window is end-aligned. -> (n, seg_samples)"""
    wave = pad_to_length(wave, seg_samples)
    starts = list(range(0, len(wave) - seg_samples + 1, hop_samples))
    if (len(wave) - seg_samples) % hop_samples:
        starts.append(len(wave) - seg_samples)
    return np.stack([wave[s:s + seg_samples] for s in starts])
