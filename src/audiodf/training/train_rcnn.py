"""RCNN branch training: random 2 s windows read from the FLAC files, codec-augmented, one-cycle LR,
class-weighted BCE; the checkpoint with the best dev EER (at the 10 s decision) is kept."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from audiodf.config import Settings
from audiodf.data.prepare import RandomWindowSampler, RcnnWindowDataset, SplitIndex, _worker_init
from audiodf.evaluation.metrics import compute_metrics
from audiodf.evaluation.stream_eval import score_rcnn
from audiodf.models.rcnn import RCNN, load_rcnn, save_rcnn


def train_rcnn(train: SplitIndex, dev: SplitIndex, tune_utts: np.ndarray, settings: Settings,
               device: torch.device, ckpt_path: Path, workers: int, log=print) -> tuple[RCNN, list[dict]]:
    cfg = settings.rcnn_train
    torch.manual_seed(cfg.seed)
    model = RCNN(settings.rcnn_features.n_mels, settings.segment_frames, settings.rcnn_model).to(device)

    sampler = RandomWindowSampler(train, settings.segment_samples, settings.data.rcnn_windows_per_utt,
                                  settings.window_samples, cfg.seed)
    loader = DataLoader(RcnnWindowDataset(train, settings, settings.data.codec_aug_p, cfg.seed),
                        batch_size=cfg.batch_size, sampler=sampler, num_workers=workers, drop_last=True,
                        worker_init_fn=_worker_init, prefetch_factor=4 if workers else None,
                        persistent_workers=workers > 0)
    labels = train.label[train.has_speech]
    n_spoof = int(labels.sum())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(labels) - n_spoof) / n_spoof, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.epochs * len(loader))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    best, history = float("inf"), []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        sampler.set_epoch(epoch)
        t0, total, seen = time.time(), 0.0, 0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = loss_fn(model(x).float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(y)
            seen += len(y)
        horizon = score_rcnn(model, dev, tune_utts, settings, device, workers,
                             grid=(settings.stream.window_seconds,), augment_p=settings.data.tune_aug_p)
        eer = compute_metrics(dev.label[tune_utts], horizon[:, 0])["eer_pct"]
        history.append({"epoch": epoch, "loss": round(total / seen, 5), "dev_eer_pct": eer,
                        "minutes": round((time.time() - t0) / 60, 1)})
        log(f"  epoch {epoch}/{cfg.epochs}: {history[-1]}")
        if eer < best:
            best = eer
            save_rcnn(model, ckpt_path)
    return load_rcnn(ckpt_path, device), history
