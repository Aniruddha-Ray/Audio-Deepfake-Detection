"""RCNN branch features: one Log-Mel spectrogram per FIXED 2 s window (n_mels x frames).

Unlike the SVM branch this works on short windows, mirroring the Kafka chunk buffer, and
standardises each spectrogram on its own (zero mean, unit variance).
"""

from __future__ import annotations

import numpy as np
import torch
import torchaudio.transforms as AT

from audiodf.config import AudioConfig, RcnnFeatureConfig

FEATURE_VERSION = 1  # bump on any change to the Log-Mel definition; checked when artifacts load


class RcnnFeatureExtractor:
    def __init__(self, audio: AudioConfig | None = None, cfg: RcnnFeatureConfig | None = None,
                 segment_samples: int = 32000):
        self.audio = audio or AudioConfig()
        self.cfg = cfg or RcnnFeatureConfig()
        self.n_frames = segment_samples // self.audio.hop_length
        self._mel = AT.MelSpectrogram(
            sample_rate=self.audio.sample_rate, n_fft=self.audio.n_fft,
            hop_length=self.audio.hop_length, win_length=self.audio.win_length,
            n_mels=self.cfg.n_mels, power=2.0,
        )

    @property
    def shape(self) -> tuple[int, int]:
        return (self.cfg.n_mels, self.n_frames)

    @torch.no_grad()
    def extract(self, segments: np.ndarray) -> np.ndarray:
        """segments: (n, segment_samples) -> (n, n_mels, n_frames) float32."""
        spec = torch.log(self._mel(torch.from_numpy(np.ascontiguousarray(segments, dtype=np.float32)))
                         + self.cfg.log_offset)[:, :, :self.n_frames]
        mean = spec.mean(dim=(1, 2), keepdim=True)
        std = spec.std(dim=(1, 2), keepdim=True)
        return ((spec - mean) / (std + 1e-6)).numpy()
