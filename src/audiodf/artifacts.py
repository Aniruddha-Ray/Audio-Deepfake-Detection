"""Versioned model bundle: svm.joblib + rcnn.pt + bundle.json (shapes and training metrics)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
from sklearn.pipeline import Pipeline

from audiodf import __version__
from audiodf.config import Settings
from audiodf.features import rcnn_features, svm_features
from audiodf.models.rcnn import RCNN, load_rcnn
from audiodf.models.svm import load_svm, save_svm

SVM_FILE, RCNN_FILE, MANIFEST_FILE = "svm.joblib", "rcnn.pt", "bundle.json"


@dataclass
class ModelBundle:
    svm: Pipeline
    rcnn: RCNN
    manifest: dict


def feature_versions() -> dict:
    return {"svm": svm_features.FEATURE_VERSION, "rcnn": rcnn_features.FEATURE_VERSION}


def bundle_dir(settings: Settings) -> Path:
    return Path(settings.paths.artifacts_dir)


def write_manifest(settings: Settings, svm: Pipeline, rcnn: RCNN, metrics: dict | None = None) -> None:
    out = bundle_dir(settings)
    save_svm(svm, out / SVM_FILE)
    manifest = {
        "version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feature_versions": feature_versions(),
        "svm_feature_dim": int(svm.n_features_in_),
        "rcnn_input_shape": list(rcnn.input_shape),
        "rcnn_bidirectional": rcnn.cfg.bidirectional,
        "segment_seconds": settings.segment.seconds,
        "segment_hop_seconds": settings.segment.hop_seconds,
        "svm_weight": settings.ensemble.svm_weight,
        "risk": {"high": settings.risk.high, "medium": settings.risk.medium},
        "metrics": metrics or {},
    }
    (out / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))


def load_bundle(settings: Settings, device: str | torch.device = "cpu") -> ModelBundle:
    out = bundle_dir(settings)
    missing = [f for f in (SVM_FILE, RCNN_FILE, MANIFEST_FILE) if not (out / f).exists()]
    if missing:
        raise FileNotFoundError(f"missing {missing} in {out}; run `audiodf train` first")
    manifest = json.loads((out / MANIFEST_FILE).read_text())
    if manifest.get("feature_versions") != feature_versions():
        raise ValueError(f"artifacts in {out} were built with feature versions "
                         f"{manifest.get('feature_versions')}, this code computes {feature_versions()}; retrain")
    svm = load_svm(out / SVM_FILE)
    rcnn = load_rcnn(out / RCNN_FILE, device)
    expect_shape = (settings.rcnn_features.n_mels, settings.segment_frames)
    if svm.n_features_in_ != settings.svm_features.dim:
        raise ValueError(f"SVM expects {svm.n_features_in_} features, config produces {settings.svm_features.dim}")
    if tuple(rcnn.input_shape) != expect_shape:
        raise ValueError(f"RCNN expects {tuple(rcnn.input_shape)} input, config produces {expect_shape}")
    return ModelBundle(svm, rcnn, manifest)
