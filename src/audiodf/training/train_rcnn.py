"""RCNN branch training (Log-Mel windows); the loop is shared with WavLM in window_trainer.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from audiodf.config import Settings
from audiodf.data.prepare import RcnnWindowDataset, SplitIndex
from audiodf.models.rcnn import RCNN, load_rcnn, save_rcnn
from audiodf.training.window_trainer import train_window_model


def train_rcnn(train: SplitIndex, dev: SplitIndex, tune_utts: np.ndarray, settings: Settings,
               device: torch.device, ckpt_path: Path, workers: int, log=print,
               dev_aug_p: float | None = None) -> tuple[RCNN, list[dict]]:
    """dev_aug_p: simulated codec rate for the per-epoch tuning check (0 when tuning clips already carry
    real-codec copies); defaults to settings.data.tune_aug_p."""
    cfg = settings.rcnn_train
    torch.manual_seed(cfg.seed)
    model = RCNN(settings.rcnn_features.n_mels, settings.segment_frames, settings.rcnn_model).to(device)
    return train_window_model(
        model, [{"params": list(model.parameters()), "lr": cfg.lr}], RcnnWindowDataset, train, dev, tune_utts,
        settings, device, ckpt_path, workers, log,
        settings.data.tune_aug_p if dev_aug_p is None else dev_aug_p,
        epochs=cfg.epochs, batch_size=cfg.batch_size, weight_decay=cfg.weight_decay, seed=cfg.seed,
        save=save_rcnn, load=lambda p: load_rcnn(p, device), score_batch=256)
