"""Audio I/O shared by training and serving. Output is always mono float32 at the target rate."""

from __future__ import annotations

import io
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def to_mono_target_rate(wave: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    if sr != target_sr:
        g = gcd(sr, target_sr)
        wave = resample_poly(wave, target_sr // g, sr // g).astype(np.float32)
    return np.ascontiguousarray(wave)


def load_audio(source, target_sr: int) -> np.ndarray:
    """source: file path or bytes (wav/flac/ogg)."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    wave, sr = sf.read(source, dtype="float32", always_2d=False)
    return to_mono_target_rate(wave, sr, target_sr)


def pcm16_to_float(data: bytes) -> np.ndarray:
    """Little-endian signed 16-bit mono PCM -> float32 in [-1, 1]."""
    usable = len(data) - (len(data) % 2)
    return np.frombuffer(data[:usable], dtype="<i2").astype(np.float32) / 32768.0


def float_to_pcm16(wave: np.ndarray) -> bytes:
    return (np.clip(wave, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
