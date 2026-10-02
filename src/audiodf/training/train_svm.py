"""SVM branch training: whole-utterance feature vectors, no segmentation."""

from __future__ import annotations

from sklearn.pipeline import Pipeline

from audiodf.config import Settings
from audiodf.data.cache import SplitData
from audiodf.models.svm import build_svm


def train_svm(train: SplitData, settings: Settings) -> Pipeline:
    return build_svm(settings.svm_model).fit(train.utt_svm, train.utt_label)
