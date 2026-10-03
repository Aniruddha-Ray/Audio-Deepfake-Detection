"""Classical branch: StandardScaler + RBF SVM over the growing-buffer feature vector.

The SVM is trained without its built-in probability (internal 5-fold Platt, ~10x slower to fit) and
calibrated afterwards on held-out dev data, so its P(spoof) reflects data the SVM did not train on.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from audiodf.config import SvmModelConfig


def build_svm(cfg: SvmModelConfig | None = None) -> Pipeline:
    cfg = cfg or SvmModelConfig()
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=cfg.C, gamma=cfg.gamma, class_weight=cfg.class_weight)),
    ])


class CalibratedSvm:
    """Fitted scaler+SVM plus a 1-D logistic (Platt) map from decision value to P(spoof)."""

    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        self.calibrator: LogisticRegression | None = None

    @property
    def n_features_in_(self) -> int:
        return int(self.pipeline.n_features_in_)

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        return self.pipeline.decision_function(np.atleast_2d(features))

    def fit_calibrator(self, features: np.ndarray, labels: np.ndarray) -> "CalibratedSvm":
        self.calibrator = LogisticRegression(C=1e6, max_iter=1000).fit(
            self.decision_function(features)[:, None], labels)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.calibrator is None:
            raise RuntimeError("SVM is not calibrated; call fit_calibrator on held-out data first")
        return self.calibrator.predict_proba(self.decision_function(features)[:, None])


def predict_spoof_proba(model: CalibratedSvm, features: np.ndarray, n_jobs: int = 1, chunk: int = 4000) -> np.ndarray:
    """P(spoof) for (n, dim) features; column 1 is spoof because classes are [0, 1]."""
    features = np.atleast_2d(features)
    if n_jobs == 1 or len(features) <= chunk:
        return model.predict_proba(features)[:, 1]
    parts = Parallel(n_jobs=n_jobs)(delayed(model.predict_proba)(features[i:i + chunk])
                                    for i in range(0, len(features), chunk))
    return np.concatenate(parts)[:, 1]


def save_svm(model: CalibratedSvm, path: str | Path) -> None:
    joblib.dump(model, path)


def load_svm(path: str | Path) -> CalibratedSvm:
    return joblib.load(path)
