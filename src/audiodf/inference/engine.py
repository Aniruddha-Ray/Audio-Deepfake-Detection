"""Shared, stateless scoring engine: each branch keeps its own preprocessing, then the branches fuse.

SVM: hand-crafted features over the whole buffer. RCNN: Log-Mel of each 2 s window. WavLM and Whisper: the raw 2 s window.
Only branches with a non-zero fusion weight are run.
"""

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
from audiodf.models.rcnn import predict_spoof_proba as rcnn_predict
from audiodf.models.svm import predict_spoof_proba as svm_predict
from audiodf.risk import RiskEngine

WINDOW_BRANCHES = ("rcnn", "wavlm", "whisper")


@dataclass(frozen=True)
class Verdict:
    label: str
    fake_probability: float
    svm_probability: float | None  # None when the branch is not part of the fused model
    rcnn_probability: float | None
    risk_level: str
    action: str
    segments_scored: int
    audio_seconds: float
    final: bool
    latency_ms: float
    call_id: str | None = None
    wavlm_probability: float | None = None
    whisper_probability: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class DetectionEngine:
    def __init__(self, models: dict, settings: Settings, device: str | torch.device = "cpu"):
        """models: branch name -> model; settings.ensemble.weights says how they fuse."""
        self.settings, self.device = settings, torch.device(device)
        self.weights = {b: w for b, w in settings.ensemble.weights.items() if w > 0}
        missing = [b for b in self.weights if b not in models]
        if missing:
            raise ValueError(f"fusion weights name branches without a model: {missing}")
        self.models = {b: m for b, m in models.items() if b in self.weights}
        for name in WINDOW_BRANCHES:
            if name in self.models:
                self.models[name] = self.models[name].to(self.device).eval()
        self.window_branches = tuple(b for b in WINDOW_BRANCHES if b in self.models)
        self.svm_features = SvmFeatureExtractor(settings.audio, settings.svm_features)
        self.rcnn_features = RcnnFeatureExtractor(settings.audio, settings.rcnn_features, settings.segment_samples)
        self.risk = RiskEngine(settings.risk)

    @property
    def uses_svm(self) -> bool:
        return "svm" in self.models

    @classmethod
    def from_artifacts(cls, settings: Settings, device: str | torch.device | None = None) -> "DetectionEngine":
        """The fusion weights and risk thresholds tuned during training live in the bundle and override
        the settings, so a model always runs with the operating point it was tuned for."""
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        bundle = load_bundle(settings, device, only_weighted=True)
        settings = copy.deepcopy(settings)
        settings.ensemble.weights = bundle.weights
        settings.risk.high = bundle.manifest["risk"]["high"]
        settings.risk.medium = bundle.manifest["risk"]["medium"]
        engine = cls(bundle.models, settings, device)
        engine.warmup()
        return engine

    def warmup(self) -> None:
        """Pay one-time CUDA/library start-up cost before the first real call (noise loud enough
        to pass the VAD, so every stage actually runs)."""
        n = self.settings.segment_samples + self.settings.segment_hop_samples
        self.predict_waveform((np.random.default_rng(0).standard_normal(n) * 0.05).astype(np.float32))

    def svm_probability(self, wave: np.ndarray) -> float:
        """SVM branch: one feature vector over the whole buffer (never a 2 s segment)."""
        return float(svm_predict(self.models["svm"], self.svm_features.extract(wave)[None])[0])

    def rcnn_probabilities(self, segments: np.ndarray) -> np.ndarray:
        """RCNN branch: one probability per fixed 2 s window, shape (n,)."""
        return rcnn_predict(self.models["rcnn"], self.rcnn_features.extract(segments), self.device)

    def wavlm_probabilities(self, segments: np.ndarray) -> np.ndarray:
        """WavLM branch: one probability per raw 2 s window, shape (n,)."""
        from audiodf.models.wavlm import predict_spoof_proba as wavlm_predict

        return wavlm_predict(self.models["wavlm"], segments, self.device)

    def whisper_probabilities(self, segments: np.ndarray) -> np.ndarray:
        """Whisper-encoder branch: one probability per raw 2 s window, shape (n,)."""
        from audiodf.models.whisper import predict_spoof_proba as whisper_predict

        return whisper_predict(self.models["whisper"], segments, self.device)

    def window_probabilities(self, segments: np.ndarray) -> dict:
        """Every active window branch on the same (n, samples) windows -> {branch: (n,)}."""
        score = {"rcnn": self.rcnn_probabilities, "wavlm": self.wavlm_probabilities,
                 "whisper": self.whisper_probabilities}
        return {b: score[b](segments) for b in self.window_branches}

    def make_verdict(self, svm_p: float | None, window_probs: dict, audio_seconds: float, final: bool,
                     latency_ms: float, call_id: str | None = None) -> Verdict:
        """window_probs: {branch: per-window P(spoof) list}; each branch is averaged over its windows."""
        probs = {b: float(np.mean(v)) for b, v in window_probs.items()}
        if svm_p is not None:
            probs["svm"] = svm_p
        fake_p = float(fuse(probs, self.weights))
        risk = self.risk.classify(fake_p)
        rounded = {b: (round(probs[b], 5) if b in probs else None) for b in ("svm", "rcnn", "wavlm", "whisper")}
        n_windows = max((len(v) for v in window_probs.values()), default=0)
        return Verdict(label="fake" if fake_p >= 0.5 else "real", fake_probability=round(fake_p, 5),
                       svm_probability=rounded["svm"], rcnn_probability=rounded["rcnn"],
                       wavlm_probability=rounded["wavlm"], whisper_probability=rounded["whisper"], risk_level=risk.level, action=risk.action,
                       segments_scored=n_windows, audio_seconds=round(audio_seconds, 2), final=final,
                       latency_ms=round(latency_ms, 2), call_id=call_id)

    def predict_waveform(self, wave: np.ndarray, call_id: str | None = None) -> Verdict | None:
        """Offline scoring of a whole recording, trimmed at both ends like training clips, then run
        through the same session logic as live audio. None when the recording contains no speech."""
        from audiodf.inference.session import CallSession

        session = CallSession(self, call_id)
        wave = fill_digital_silence(np.asarray(wave, dtype=np.float32))
        session.push(trim_silence(wave, self.settings.audio.sample_rate, self.settings.vad))
        return session.finalize()
