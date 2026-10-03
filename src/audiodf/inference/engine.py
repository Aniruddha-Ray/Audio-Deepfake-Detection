"""Shared, stateless scoring engine: the two branches keep separate preprocessing, then fuse."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import numpy as np
import torch

from audiodf.artifacts import load_bundle
from audiodf.config import Settings
from audiodf.data.audio import fill_digital_silence
from audiodf.data.vad import trim_silence
from audiodf.features.rcnn_features import RcnnFeatureExtractor
from audiodf.features.svm_features import SvmFeatureExtractor
from audiodf.models.ensemble import fuse
from audiodf.models.rcnn import RCNN, predict_spoof_proba as rcnn_predict
from audiodf.models.svm import predict_spoof_proba as svm_predict
from audiodf.risk import RiskEngine


@dataclass(frozen=True)
class Verdict:
    label: str
    fake_probability: float
    svm_probability: float
    rcnn_probability: float
    risk_level: str
    action: str
    segments_scored: int
    audio_seconds: float
    final: bool
    latency_ms: float
    call_id: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class DetectionEngine:
    def __init__(self, svm, rcnn: RCNN, settings: Settings, device: str | torch.device = "cpu"):
        self.settings, self.device = settings, torch.device(device)
        self.svm = svm
        self.rcnn = rcnn.to(self.device).eval()
        self.svm_features = SvmFeatureExtractor(settings.audio, settings.svm_features)
        self.rcnn_features = RcnnFeatureExtractor(settings.audio, settings.rcnn_features, settings.segment_samples)
        self.risk = RiskEngine(settings.risk)

    @classmethod
    def from_artifacts(cls, settings: Settings, device: str | torch.device | None = None) -> "DetectionEngine":
        """The fusion weight and risk thresholds tuned during training live in the bundle and override
        the settings, so a model always runs with the operating point it was tuned for."""
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        bundle = load_bundle(settings, device)
        settings = copy.deepcopy(settings)
        settings.ensemble.svm_weight = bundle.manifest["svm_weight"]
        settings.risk.high = bundle.manifest["risk"]["high"]
        settings.risk.medium = bundle.manifest["risk"]["medium"]
        engine = cls(bundle.svm, bundle.rcnn, settings, device)
        engine.warmup()
        return engine

    def warmup(self) -> None:
        """Pay one-time CUDA/library start-up cost before the first real call (noise loud enough
        to pass the VAD, so every stage actually runs)."""
        n = self.settings.segment_samples + self.settings.segment_hop_samples
        self.predict_waveform((np.random.default_rng(0).standard_normal(n) * 0.05).astype(np.float32))

    def svm_probability(self, wave: np.ndarray) -> float:
        """SVM branch: one feature vector over the whole buffer (never a 2 s segment)."""
        return float(svm_predict(self.svm, self.svm_features.extract(wave)[None])[0])

    def rcnn_probabilities(self, segments: np.ndarray) -> np.ndarray:
        """RCNN branch: one probability per fixed 2 s window, shape (n,)."""
        return rcnn_predict(self.rcnn, self.rcnn_features.extract(segments), self.device)

    def make_verdict(self, svm_p: float, rcnn_window_probs, audio_seconds: float, final: bool,
                     latency_ms: float, call_id: str | None = None) -> Verdict:
        rcnn_p = float(np.mean(rcnn_window_probs))
        fake_p = float(fuse(svm_p, rcnn_p, self.settings.ensemble.svm_weight))
        risk = self.risk.classify(fake_p)
        return Verdict(label="fake" if fake_p >= 0.5 else "real", fake_probability=round(fake_p, 5),
                       svm_probability=round(svm_p, 5), rcnn_probability=round(rcnn_p, 5),
                       risk_level=risk.level, action=risk.action, segments_scored=len(rcnn_window_probs),
                       audio_seconds=round(audio_seconds, 2), final=final, latency_ms=round(latency_ms, 2),
                       call_id=call_id)

    def predict_waveform(self, wave: np.ndarray, call_id: str | None = None) -> Verdict | None:
        """Offline scoring of a whole recording, trimmed at both ends like training clips, then run
        through the same session logic as live audio. None when the recording contains no speech."""
        from audiodf.inference.session import CallSession

        session = CallSession(self, call_id)
        wave = fill_digital_silence(np.asarray(wave, dtype=np.float32))
        session.push(trim_silence(wave, self.settings.audio.sample_rate, self.settings.vad))
        return session.finalize()
