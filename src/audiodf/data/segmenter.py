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


def prefix_snapshots(wave: np.ndarray, sample_rate: int, seconds, min_samples: int) -> list[np.ndarray]:
    """SVM training views: the first s seconds of the clip for each s, capped by the clip length,
    i.e. the growing buffer CallSession scores live. Buffers shorter than min_samples are
    repeat-padded exactly as serving pads them."""
    lengths = sorted({min(int(s * sample_rate), len(wave)) for s in seconds} - {0})
    return [pad_to_length(wave[:n], min_samples) for n in lengths]


def stream_window_starts(n_samples: int, seg_samples: int, hop_samples: int, horizon_samples: int) -> list[int]:
    """Starts of the RCNN windows CallSession has scored once `horizon_samples` of audio arrived:
    a fixed grid (0, hop, 2*hop, ...) with no end-aligned extra window."""
    n = min(n_samples, horizon_samples)
    return list(range(0, n - seg_samples + 1, hop_samples))


def segment(wave: np.ndarray, seg_samples: int, hop_samples: int) -> np.ndarray:
    """Windows of seg_samples every hop_samples; a trailing window is end-aligned. -> (n, seg_samples)"""
    wave = pad_to_length(wave, seg_samples)
    starts = list(range(0, len(wave) - seg_samples + 1, hop_samples))
    if (len(wave) - seg_samples) % hop_samples:
        starts.append(len(wave) - seg_samples)
    return np.stack([wave[s:s + seg_samples] for s in starts])
