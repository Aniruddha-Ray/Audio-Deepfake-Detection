"""Whisper-encoder branch training: raw 2 s windows, top encoder layers fine-tuned slowly, new head trained fast; the loop is the
one shared with the WavLM branch (window_trainer.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from audiodf.config import Settings
from audiodf.data.prepare import SplitIndex, WaveWindowDataset
from audiodf.models.whisper import WhisperDetector, load_whisper, save_whisper
from audiodf.training.window_trainer import train_window_model


def train_whisper(train: SplitIndex, dev: SplitIndex, tune_utts: np.ndarray, settings: Settings,
                  device: torch.device, ckpt_path: Path, workers: int, log=print,
                  dev_aug_p: float = 0.0) -> tuple[WhisperDetector, list[dict]]:
    cfg = settings.whisper
    torch.manual_seed(cfg.seed)
    model = WhisperDetector(cfg.backbone, cfg.finetune_top, cfg.hidden, cfg.dropout, cfg.pretrained,
                            cfg.encoder_config or None).to(device)
    return train_window_model(
        model, model.param_groups(cfg.head_lr, cfg.backbone_lr), WaveWindowDataset, train, dev, tune_utts,
        settings, device, ckpt_path, workers, log, dev_aug_p,
        epochs=cfg.epochs, batch_size=cfg.batch_size, weight_decay=cfg.weight_decay, seed=cfg.seed,
        save=save_whisper, load=lambda p: load_whisper(p, device, cfg.pretrained), score_batch=cfg.batch_size,
        max_grad_norm=5.0, keep_epochs=cfg.keep_epochs)
