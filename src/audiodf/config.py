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
    epochs: int = 10
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
class WavlmConfig:
    pretrained: bool = True  # False builds the architecture without downloading weights (tests only)
    finetune_top: int = 4  # top transformer layers that adapt; the rest of WavLM-Base+ stays frozen
    hidden: int = 128
    dropout: float = 0.2
    epochs: int = 8
    batch_size: int = 32  # batch 64 thrashes the 4.3 GB GPU (measured 84 vs 249 windows/s)
    head_lr: float = 1e-3
    backbone_lr: float = 2e-5
    weight_decay: float = 1e-4
    seed: int = 0


@dataclass
class EnsembleConfig:
    # Which branches to train and fuse. WavLM alone since run 4 (ASV5 eval 5.43% alone vs 5.51% fused with the
    # ~29% SVM/RCNN branches, and ~57 ms less compute per verdict); SVM and RCNN stay available.
    branches: tuple = ("wavlm",)
    # Training tunes these on the held-out set and stores them in the model bundle, which wins at serving time.
    weights: dict = field(default_factory=lambda: {"wavlm": 1.0})


@dataclass
class RiskConfig:
    high: float = 0.80
    medium: float = 0.50
    # Training sets high/medium so that this share of bonafide dev clips scores at or above each
    # (a 1% block rate and a 10% verify rate on genuine speech); change these to change the policy.
    block_fpr: float = 0.01
    verify_fpr: float = 0.10


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
class VadConfig:
    threshold_db: float = -45.0
    frame_seconds: float = 0.025
    margin_seconds: float = 0.05


@dataclass
class DataConfig:
    dataset: str = "asv5"
    # Training pool as "dataset:split". Test splits (asv5:eval, asv19:eval) are refused.
    train_splits: tuple = ("asv5:train", "asv5:dev", "asv19:train", "asv19:dev")
    # Tuning set: these attacks (dataset-prefixed) plus bonafide clips of `holdout_speaker_frac` of the bonafide
    # speakers in `holdout_speaker_source` are removed from the pool and used only for tuning. With no held-out
    # attacks, tuning falls back to ASV5 dev, which then must not be in the training pool.
    holdout_attacks: tuple = ("asv5:A10", "asv5:A12", "asv5:A15")
    holdout_speaker_source: str = "asv5:dev"
    holdout_speaker_frac: float = 0.25
    # SVM sees growing buffers, as CallSession scores them live (never isolated 2 s crops).
    svm_snapshot_seconds: tuple = (2.0, 4.0, 6.0, 8.0, 10.0)
    # Random 2 s windows drawn per utterance per epoch, so long clips don't dominate training.
    rcnn_windows_per_utt: int = 1
    svm_train_utts: int = 20000  # clips (each gives up to 5 snapshot vectors); exact RBF scales ~n^1.5
    tune_utts: int = 12000  # dev clips used for per-epoch checks, SVM calibration and fusion weight
    eval_utts: int = 60000  # test clips scored per report (0 = the whole split; ~35 clips/s for the SVM)
    # Real-codec copies rendered with ffmpeg (data/ffmpeg_codecs.py) for this label-blind share of training
    # clips; the in-process simulated codecs below apply only to clips without a copy. About 0.5 + 0.5 * 0.4
    # = 70% of training clips end up codec-processed, near ASV5 eval's ~75%.
    ffmpeg_codecs: bool = True
    render_frac: float = 0.5
    tune_render_frac: float = 0.75  # tuning clips with a real-codec copy (replaces tune_aug_p when enabled)
    # Applied to bonafide and spoof with the same probability; never conditioned on the label.
    codec_aug_p: float = 0.4
    # Dev tuning clips get codec augmentation at about the rate ASV5 eval has (~75% codec-processed),
    # so epoch choice, calibration, fusion weight and thresholds are picked for codec'd audio, not clean.
    tune_aug_p: float = 0.75


@dataclass
class PathsConfig:
    data_root: str = str(REPO_ROOT / "dataset" / "LA" / "LA")
    asv5_root: str = str(REPO_ROOT / "dataset5")
    asv21_root: str = str(REPO_ROOT / "dataset21")  # ASVspoof 2021 LA eval: telephony test set (data/asv21.py)
    calls_root: str = str(REPO_ROOT / "dataset_calls")  # simulated VoIP calls from ASV5 eval clips (data/voip_sim.py)
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
    wavlm: WavlmConfig = field(default_factory=WavlmConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)
    kafka: KafkaConfig = field(default_factory=KafkaConfig)
    data: DataConfig = field(default_factory=DataConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    def dataset_root(self, dataset: str | None = None) -> str:
        dataset = dataset or self.data.dataset
        return {"asv19": self.paths.data_root, "asv5": self.paths.asv5_root, "asv21": self.paths.asv21_root,
                "calls": self.paths.calls_root}[dataset]

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
    if os.environ.get("AUDIODF_ASV5"):
        settings.paths.asv5_root = os.environ["AUDIODF_ASV5"]
    if os.environ.get("AUDIODF_ASV21"):
        settings.paths.asv21_root = os.environ["AUDIODF_ASV21"]
    if os.environ.get("AUDIODF_CALLS"):
        settings.paths.calls_root = os.environ["AUDIODF_CALLS"]
    if os.environ.get("AUDIODF_ARTIFACTS"):
        settings.paths.artifacts_dir = os.environ["AUDIODF_ARTIFACTS"]
    if os.environ.get("KAFKA_BOOTSTRAP_SERVERS"):
        settings.kafka.bootstrap_servers = os.environ["KAFKA_BOOTSTRAP_SERVERS"]
    for name in ("data_root", "asv5_root", "asv21_root", "calls_root", "cache_dir", "artifacts_dir", "results_dir"):
        setattr(settings.paths, name, os.path.expanduser(getattr(settings.paths, name)))
    return settings
