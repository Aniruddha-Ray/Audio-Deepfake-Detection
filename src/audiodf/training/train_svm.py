"""SVM branch training: growing-buffer snapshots of a stratified utterance subset, codec-augmented."""

from __future__ import annotations

import time

from audiodf.config import Settings
from audiodf.data.prepare import SplitIndex, build_svm_snapshots, stratified_subset
from audiodf.models.svm import CalibratedSvm, build_svm


def train_svm(train: SplitIndex, settings: Settings, workers: int, n_utts: int | None = None,
              log=print) -> CalibratedSvm:
    n_utts = n_utts or settings.data.svm_train_utts
    utts = stratified_subset(train, n_utts)
    data = build_svm_snapshots(train, utts, settings, settings.data.codec_aug_p, f"train{len(utts)}",
                               workers, log=log)
    t0 = time.time()
    pipeline = build_svm(settings.svm_model).fit(data["x"], data["label"])
    log(f"  SVM fit on {len(data['x'])} snapshot vectors from {len(utts)} clips in {time.time() - t0:.0f}s")
    return CalibratedSvm(pipeline)
