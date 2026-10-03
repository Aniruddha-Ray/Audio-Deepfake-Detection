"""Fixtures: a tiny trained engine built from synthetic audio, so tests need no dataset or GPU."""

import numpy as np
import pytest
import torch

from audiodf.config import Settings
from audiodf.features.svm_features import SvmFeatureExtractor
from audiodf.inference.engine import DetectionEngine
from audiodf.models.rcnn import RCNN
from audiodf.models.svm import CalibratedSvm, build_svm

SR = 16000


def tone(seconds: float, freq: float = 220.0, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * freq * t) + noise * rng.standard_normal(len(t))).astype(np.float32)


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings()


@pytest.fixture(scope="session")
def engine(settings) -> DetectionEngine:
    torch.manual_seed(0)
    fx = SvmFeatureExtractor(settings.audio, settings.svm_features)
    X, y = [], []
    for i in range(24):
        spoof = i % 2
        X.append(fx.extract(tone(3, 220 + 40 * i, noise=0.02 if spoof else 0.3, seed=i)))
        y.append(spoof)
    svm = CalibratedSvm(build_svm(settings.svm_model).fit(np.stack(X), np.array(y)))
    svm.fit_calibrator(np.stack(X), np.array(y))
    rcnn = RCNN(settings.rcnn_features.n_mels, settings.segment_frames, settings.rcnn_model)
    return DetectionEngine(svm, rcnn, settings, "cpu")
