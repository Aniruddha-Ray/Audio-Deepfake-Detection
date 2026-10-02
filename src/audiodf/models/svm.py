"""Classical branch: StandardScaler + RBF SVM over the whole-buffer feature vector."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from audiodf.config import SvmModelConfig


def build_svm(cfg: SvmModelConfig | None = None) -> Pipeline:
    cfg = cfg or SvmModelConfig()
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=cfg.C, gamma=cfg.gamma, class_weight=cfg.class_weight,
                    probability=True, random_state=0)),
    ])


def predict_spoof_proba(model: Pipeline, features: np.ndarray, n_jobs: int = 1, chunk: int = 4000) -> np.ndarray:
    """P(spoof) for (n, dim) features; column 1 is spoof because classes are [0, 1]."""
    features = np.atleast_2d(features)
    if n_jobs == 1 or len(features) <= chunk:
        return model.predict_proba(features)[:, 1]
    parts = Parallel(n_jobs=n_jobs)(delayed(model.predict_proba)(features[i:i + chunk])
                                    for i in range(0, len(features), chunk))
    return np.concatenate(parts)[:, 1]


def save_svm(model: Pipeline, path: str | Path) -> None:
    joblib.dump(model, path)


def load_svm(path: str | Path) -> Pipeline:
    return joblib.load(path)
