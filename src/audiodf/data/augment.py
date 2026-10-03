"""Codec / telephony-channel augmentation, applied in-process through libsndfile (no ffmpeg).

ASVspoof5 train/dev contain no codec-processed audio, while ~75% of eval (and every real phone call)
is codec-processed, equally for bonafide and spoof. The augmenter therefore never sees the label:
callers apply it with the same probability to both classes. Augmenting one class only would teach
the model that "codec" predicts the label.

Covered families: Opus (wide/narrowband), MP3, Vorbis, G.711 mu/A-law, plain narrowband (8 kHz).
Not covered (no in-process encoder): AMR, Speex, EnCodec, AAC/M4A, Bluetooth device channels.
"""

from __future__ import annotations

import io

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

CODECS = ("opus_wb", "opus_nb", "mp3", "vorbis", "g711_ulaw", "g711_alaw", "narrowband")
_SR = 16000
# libsndfile compression_level ranges chosen to land near ASVspoof5 eval bitrates (measured on 2 s clips):
# opus 0.88-1.0 ~ 9-30 kbps (eval C01/C08: 4-30), mp3 0-0.99 ~ 16-68 kbps (MP3 rejects 1.0), vorbis ~35-54 kbps.
LEVELS = {"opus_wb": (0.88, 1.0), "opus_nb": (0.9, 1.0), "mp3": (0.0, 0.99), "vorbis": (0.5, 1.0)}


def _roundtrip(wave: np.ndarray, sr: int, fmt: str, subtype: str, level: float | None) -> np.ndarray:
    buf = io.BytesIO()
    kwargs = {} if level is None else {"compression_level": float(level)}
    sf.write(buf, wave, sr, format=fmt, subtype=subtype, **kwargs)
    buf.seek(0)
    out, _ = sf.read(buf, dtype="float32", always_2d=False)
    return out


def _to_8k(wave: np.ndarray) -> np.ndarray:
    return resample_poly(wave, 1, 2).astype(np.float32)


def _to_16k(wave: np.ndarray) -> np.ndarray:
    return resample_poly(wave, 2, 1).astype(np.float32)


def apply_codec(wave: np.ndarray, codec: str, level: float | None = None) -> np.ndarray:
    """Encode+decode 16 kHz mono float audio. level: libsndfile compression level (higher = lower
    bitrate), clamped to the codec's range; None = middle of the range. Output keeps the input length."""
    if codec in LEVELS:
        lo, hi = LEVELS[codec]
        level = (lo + hi) / 2 if level is None else min(max(level, lo), hi)
    wave = np.clip(np.asarray(wave, dtype=np.float32), -1.0, 1.0)
    if codec == "opus_wb":
        out = _roundtrip(wave, _SR, "OGG", "OPUS", level)
    elif codec == "opus_nb":
        out = _to_16k(_roundtrip(_to_8k(wave), 8000, "OGG", "OPUS", level))
    elif codec == "mp3":
        out = _roundtrip(wave, _SR, "MP3", "MPEG_LAYER_III", level)
    elif codec == "vorbis":
        out = _roundtrip(wave, _SR, "OGG", "VORBIS", level)
    elif codec in ("g711_ulaw", "g711_alaw"):
        out = _to_16k(_roundtrip(_to_8k(wave), 8000, "WAV", "ULAW" if codec == "g711_ulaw" else "ALAW", None))
    elif codec == "narrowband":
        out = _to_16k(_to_8k(wave))
    else:
        raise ValueError(f"unknown codec {codec!r}; expected one of {CODECS}")
    n = len(wave)
    out = out[:n] if len(out) >= n else np.pad(out, (0, n - len(out)))
    return np.clip(out, -1.0, 1.0).astype(np.float32)


class CodecAugmenter:
    """With probability p, pass the audio through a random codec at a random quality.
    Takes no label on purpose: both classes must be augmented identically."""

    def __init__(self, p: float = 0.5, codecs=CODECS):
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0, 1]")
        self.p, self.codecs = p, tuple(codecs)

    def __call__(self, wave: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, str]:
        if self.p == 0.0 or rng.random() >= self.p:
            return wave, "-"
        codec = self.codecs[rng.integers(len(self.codecs))]
        level = rng.uniform(*LEVELS[codec]) if codec in LEVELS else None
        return apply_codec(wave, codec, level), codec
