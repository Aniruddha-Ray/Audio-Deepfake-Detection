"""Weighted fusion of the two branches' P(spoof)."""

from __future__ import annotations

import numpy as np


def fuse(svm_prob, rcnn_prob, svm_weight: float = 0.7):
    if not 0.0 <= svm_weight <= 1.0:
        raise ValueError("svm_weight must be in [0, 1]")
    return svm_weight * np.asarray(svm_prob) + (1.0 - svm_weight) * np.asarray(rcnn_prob)
