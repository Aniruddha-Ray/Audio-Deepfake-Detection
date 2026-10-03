"""SVM branch features: ONE fixed-size vector per whole audio buffer (up to the 10 s window).

MFCC, LFCC and MGDCC (each with deltas and delta-deltas), summarised over time by mean and std.
Unlike the RCNN branch this sees the whole buffer at once and does no per-window normalisation;
standardisation happens inside the SVM pipeline (StandardScaler). The SVM degrades when given
short 2 s segments (eval EER 8.6% -> 16.9% in our tests), so never feed it segments.
"""

from __future__ import annotations

import numpy as np
import torch
import torchaudio.functional as AF
import torchaudio.transforms as AT
from scipy import signal as sp_signal
from scipy.fftpack import dct

from audiodf.config import AudioConfig, SvmFeatureConfig
from audiodf.data.segmenter import pad_to_length

# Bump whenever the feature definition changes; artifacts record it and refuse to load on mismatch.
# v2: log floor at the STFT magnitude of a 1-LSB 16-bit signal (~1e-6) instead of 1e-10, so exact
# digital silence (VoIP DTX, padded TTS output) no longer produces log values near -23 that swamp
# the mean/std statistics.
FEATURE_VERSION = 2
MAG_FLOOR = 1e-6


class SvmFeatureExtractor:
    def __init__(self, audio: AudioConfig | None = None, cfg: SvmFeatureConfig | None = None):
        self.audio = audio or AudioConfig()
        self.cfg = cfg or SvmFeatureConfig()
        self._mfcc = AT.MFCC(
            sample_rate=self.audio.sample_rate,
            n_mfcc=self.cfg.n_mfcc,
            melkwargs={"n_fft": self.audio.n_fft, "hop_length": self.audio.hop_length,
                       "win_length": self.audio.win_length, "n_mels": self.cfg.mfcc_n_mels},
        )

    @property
    def dim(self) -> int:
        return self.cfg.dim

    @staticmethod
    def _with_deltas(x: torch.Tensor) -> torch.Tensor:
        d1 = AF.compute_deltas(x)
        return torch.cat([x, d1, AF.compute_deltas(d1)], dim=1)

    def _cepstra(self, waves: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.audio
        _, _, stft = sp_signal.stft(waves, fs=a.sample_rate, window="hann", nperseg=a.win_length,
                                    noverlap=a.win_length - a.hop_length, nfft=a.n_fft, axis=-1)
        mag = np.abs(stft)
        n = self.cfg.n_cepstra
        lfcc = dct(np.log(mag + MAG_FLOOR), type=2, axis=1, norm="ortho")[:, :n]
        phase = np.angle(stft)
        mgd = np.diff(phase, axis=1, prepend=phase[:, :1]) * mag ** self.cfg.mgd_alpha
        mgdcc = dct(np.log(np.abs(mgd) + MAG_FLOOR ** self.cfg.mgd_alpha), type=2, axis=1, norm="ortho")[:, :n]
        return torch.from_numpy(lfcc).float(), torch.from_numpy(mgdcc).float()

    @torch.no_grad()
    def extract_batch(self, waves: np.ndarray) -> np.ndarray:
        """waves: (n, samples) float32, equal length -> (n, dim)."""
        waves = np.ascontiguousarray(waves, dtype=np.float32)
        mfcc = self._with_deltas(self._mfcc(torch.from_numpy(waves)))
        lfcc, mgdcc = self._cepstra(waves)
        parts = []
        for feat in (mfcc, self._with_deltas(lfcc), self._with_deltas(mgdcc)):
            parts += [feat.mean(dim=2), feat.std(dim=2)]
        return np.nan_to_num(torch.cat(parts, dim=1).numpy())

    def extract(self, wave: np.ndarray) -> np.ndarray:
        """One buffer (1-D) -> (dim,)."""
        wave = pad_to_length(np.asarray(wave, dtype=np.float32), self.audio.win_length * 4)
        return self.extract_batch(wave[None])[0]
