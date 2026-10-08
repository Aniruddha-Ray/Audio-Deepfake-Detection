"""Versioned model bundle: svm.joblib + rcnn.pt (+ wavlm.pt) + bundle.json (branches, shapes, fusion weights,
risk thresholds, training metrics)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import torch

from audiodf import __version__
from audiodf.config import Settings
from audiodf.features import rcnn_features, svm_features
from audiodf.models.ensemble import weights_from_manifest
from audiodf.models.rcnn import load_rcnn
from audiodf.models.svm import load_svm, save_svm

SVM_FILE, RCNN_FILE, WAVLM_FILE, MANIFEST_FILE = "svm.joblib", "rcnn.pt", "wavlm.pt", "bundle.json"
WHISPER_FILE = "whisper.pt"
BRANCH_FILES = {"svm": SVM_FILE, "rcnn": RCNN_FILE, "wavlm": WAVLM_FILE, "whisper": WHISPER_FILE}
WAVLM_INPUT_VERSION = 1  # raw 16 kHz waveform windows, no normalisation


@dataclass
class ModelBundle:
    models: dict  # branch name -> model, only the branches the bundle was trained with
    manifest: dict
    weights: dict = field(default_factory=dict)

    @property
    def svm(self):
        return self.models.get("svm")

    @property
    def rcnn(self):
        return self.models.get("rcnn")

    @property
    def wavlm(self):
        return self.models.get("wavlm")

    @property
    def whisper(self):
        return self.models.get("whisper")


def feature_versions(branches=("svm", "rcnn")) -> dict:
    known = {"svm": svm_features.FEATURE_VERSION, "rcnn": rcnn_features.FEATURE_VERSION,
             "wavlm": WAVLM_INPUT_VERSION}
    if "whisper" in branches:  # imported only when used: transformers is an optional dependency
        from audiodf.models.whisper import FEATURE_VERSION as whisper_version

        known["whisper"] = whisper_version
    return {b: known[b] for b in branches}


def bundle_dir(settings: Settings) -> Path:
    return Path(settings.paths.artifacts_dir)


def write_manifest(settings: Settings, models: dict, metrics: dict | None = None) -> None:
    """Saves the SVM (window branches are already saved by their training checkpoints) and the manifest.
    settings.ensemble.weights must hold the tuned fusion weights."""
    out = bundle_dir(settings)
    branches = list(models)
    manifest = {
        "version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "branches": branches,
        "feature_versions": feature_versions(branches),
        "fusion_weights": dict(settings.ensemble.weights),
        "segment_seconds": settings.segment.seconds,
        "segment_hop_seconds": settings.segment.hop_seconds,
        "risk": {"high": settings.risk.high, "medium": settings.risk.medium},
        "metrics": metrics or {},
    }
    if "svm" in models:
        save_svm(models["svm"], out / SVM_FILE)
        manifest["svm_feature_dim"] = int(models["svm"].n_features_in_)
    if "rcnn" in models:
        manifest["rcnn_input_shape"] = list(models["rcnn"].input_shape)
        manifest["rcnn_bidirectional"] = models["rcnn"].cfg.bidirectional
    if "wavlm" in models:
        from audiodf.models.wavlm import BACKBONE

        manifest["wavlm_backbone"] = BACKBONE
        manifest["wavlm_finetune_top"] = models["wavlm"].finetune_top
    if "whisper" in models:
        manifest["whisper_backbone"] = models["whisper"].backbone
        manifest["whisper_finetune_top"] = models["whisper"].finetune_top
    (out / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))


def load_bundle(settings: Settings, device: str | torch.device = "cpu", only_weighted: bool = False) -> ModelBundle:
    """only_weighted: skip loading branches whose tuned fusion weight is 0 (serving does not need them)."""
    out = bundle_dir(settings)
    if not (out / MANIFEST_FILE).exists():
        raise FileNotFoundError(f"missing {MANIFEST_FILE} in {out}; run `audiodf train` first")
    manifest = json.loads((out / MANIFEST_FILE).read_text())
    branches = manifest.get("branches", ["svm", "rcnn"])  # bundles from runs 1-3 had no branch list
    weights = weights_from_manifest(manifest)
    if only_weighted:
        branches = [b for b in branches if weights.get(b, 0) > 0]
    missing = [BRANCH_FILES[b] for b in branches if not (out / BRANCH_FILES[b]).exists()]
    if missing:
        raise FileNotFoundError(f"missing {missing} in {out}; run `audiodf train` first")
    expected = feature_versions(branches)
    stored = {b: manifest.get("feature_versions", {}).get(b) for b in branches}
    if stored != expected:
        raise ValueError(f"artifacts in {out} were built with feature versions {stored}, "
                         f"this code computes {expected}; retrain")
    models = {}
    if "svm" in branches:
        models["svm"] = load_svm(out / SVM_FILE)
        if models["svm"].n_features_in_ != settings.svm_features.dim:
            raise ValueError(f"SVM expects {models['svm'].n_features_in_} features, "
                             f"config produces {settings.svm_features.dim}")
    if "rcnn" in branches:
        models["rcnn"] = load_rcnn(out / RCNN_FILE, device)
        expect_shape = (settings.rcnn_features.n_mels, settings.segment_frames)
        if tuple(models["rcnn"].input_shape) != expect_shape:
            raise ValueError(f"RCNN expects {tuple(models['rcnn'].input_shape)} input, config produces {expect_shape}")
    if "wavlm" in branches:
        from audiodf.models.wavlm import load_wavlm

        models["wavlm"] = load_wavlm(out / WAVLM_FILE, device, settings.wavlm.pretrained)
    if "whisper" in branches:
        from audiodf.models.whisper import load_whisper

        models["whisper"] = load_whisper(out / WHISPER_FILE, device, settings.whisper.pretrained)
    return ModelBundle(models, manifest, weights)
