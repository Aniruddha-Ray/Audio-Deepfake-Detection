"""Central settings. Defaults mirror the validated experiment; override with a YAML file."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160


@dataclass
class SegmentConfig:
    seconds: float = 2.0
    hop_seconds: float = 1.0


@dataclass
class SvmFeatureConfig:
    n_mfcc: int = 13
    mfcc_n_mels: int = 26
    n_cepstra: int = 20
    mgd_alpha: float = 0.4

    @property
    def dim(self) -> int:
        # mean+std of (static, delta, delta-delta) for MFCC, LFCC and MGDCC
        return 2 * 3 * (self.n_mfcc + 2 * self.n_cepstra)


@dataclass
class RcnnFeatureConfig:
    n_mels: int = 64
    log_offset: float = 1e-2


@dataclass
class RcnnModelConfig:
    hidden: int = 128
    bidirectional: bool = True
    dropout: float = 0.2


@dataclass
class RcnnTrainConfig:
    epochs: int = 20
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 0


@dataclass
class SvmModelConfig:
    C: float = 10.0
    gamma: str = "scale"
    class_weight: str = "balanced"


@dataclass
class EnsembleConfig:
    # 0.7/0.3 gave the best eval EER (5.7%); weights tuned on dev made eval worse (see README).
    svm_weight: float = 0.7


@dataclass
class RiskConfig:
    high: float = 0.80
    medium: float = 0.50


@dataclass
class StreamConfig:
    window_seconds: float = 10.0
    chunk_seconds: float = 0.5
    session_idle_seconds: float = 30.0
    max_sessions: int = 1000


@dataclass
class KafkaConfig:
    bootstrap_servers: str = "localhost:9092"
    audio_topic: str = "deepfake-detection-audio-chunks"
    result_topic: str = "deepfake-detection-verdicts"
    group_id: str = "deepfake-detector"


@dataclass
class PathsConfig:
    data_root: str = str(REPO_ROOT / "dataset" / "LA" / "LA")
    cache_dir: str = str(Path.home() / ".cache" / "audiodf")
    artifacts_dir: str = str(REPO_ROOT / "artifacts")
    results_dir: str = str(REPO_ROOT / "results")


@dataclass
class Settings:
    audio: AudioConfig = field(default_factory=AudioConfig)
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    svm_features: SvmFeatureConfig = field(default_factory=SvmFeatureConfig)
    rcnn_features: RcnnFeatureConfig = field(default_factory=RcnnFeatureConfig)
    rcnn_model: RcnnModelConfig = field(default_factory=RcnnModelConfig)
    rcnn_train: RcnnTrainConfig = field(default_factory=RcnnTrainConfig)
    svm_model: SvmModelConfig = field(default_factory=SvmModelConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)
    kafka: KafkaConfig = field(default_factory=KafkaConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    @property
    def segment_samples(self) -> int:
        return int(self.segment.seconds * self.audio.sample_rate)

    @property
    def segment_hop_samples(self) -> int:
        return int(self.segment.hop_seconds * self.audio.sample_rate)

    @property
    def segment_frames(self) -> int:
        return self.segment_samples // self.audio.hop_length

    @property
    def window_samples(self) -> int:
        return int(self.stream.window_seconds * self.audio.sample_rate)

    def to_dict(self) -> dict:
        return asdict(self)


def _merge(obj, overrides: dict) -> None:
    for key, value in overrides.items():
        if not hasattr(obj, key):
            raise KeyError(f"unknown setting: {key}")
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
        else:
            setattr(obj, key, type(current)(value) if current is not None else value)


def load_settings(path: str | Path | None = None) -> Settings:
    """Defaults, then YAML (path or $AUDIODF_CONFIG), then $AUDIODF_DATA / $AUDIODF_ARTIFACTS."""
    settings = Settings()
    path = path or os.environ.get("AUDIODF_CONFIG")
    if path:
        with open(path) as f:
            _merge(settings, yaml.safe_load(f) or {})
    if os.environ.get("AUDIODF_DATA"):
        settings.paths.data_root = os.environ["AUDIODF_DATA"]
    if os.environ.get("AUDIODF_ARTIFACTS"):
        settings.paths.artifacts_dir = os.environ["AUDIODF_ARTIFACTS"]
    if os.environ.get("KAFKA_BOOTSTRAP_SERVERS"):
        settings.kafka.bootstrap_servers = os.environ["KAFKA_BOOTSTRAP_SERVERS"]
    for name in ("data_root", "cache_dir", "artifacts_dir", "results_dir"):
        setattr(settings.paths, name, os.path.expanduser(getattr(settings.paths, name)))
    return settings
