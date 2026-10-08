"""Speech-to-text of the caller's audio with Whisper, run locally (the transcript never leaves the machine). The weights of openai/whisper-small are
the ones already cached for the Whisper branch. A bank's telephony keeps the caller's leg separate; the whole input is treated as the caller."""

from __future__ import annotations

import numpy as np

MODEL = "openai/whisper-small"
_PIPES: dict = {}


def transcribe(wave: np.ndarray, sr: int = 16000, model: str = MODEL, language: str | None = None) -> list[tuple[float, str]]:
    """[(start seconds, text)] segments of the audio. language=None lets Whisper detect it (Hindi / English / mixed)."""
    import torch
    from transformers import pipeline

    key = (model, torch.cuda.is_available())
    if key not in _PIPES:
        _PIPES[key] = pipeline("automatic-speech-recognition", model=model, device=0 if key[1] else -1, chunk_length_s=30)
    kwargs = {"task": "transcribe"} | ({"language": language} if language else {})
    out = _PIPES[key]({"raw": np.asarray(wave, dtype=np.float32), "sampling_rate": sr}, return_timestamps=True, generate_kwargs=kwargs)
    chunks = out.get("chunks") or [{"timestamp": (0.0, None), "text": out.get("text", "")}]
    return [(float(c["timestamp"][0] or 0.0), c["text"].strip()) for c in chunks if c["text"].strip()]
