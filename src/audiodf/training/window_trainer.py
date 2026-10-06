"""Shared loop for the window branches (RCNN, WavLM): random 2 s windows read from the FLAC files (or their codec
copies), codec-augmented, one-cycle LR per parameter group, class-weighted BCE; after every epoch the branch is
scored on the tuning set at the 10 s decision and the best epoch is kept."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from audiodf.config import Settings
from audiodf.data.impairments import LossAugmenter, LossConfig
from audiodf.data.prepare import RandomWindowSampler, SplitIndex, _worker_init
from audiodf.evaluation.metrics import compute_metrics
from audiodf.evaluation.stream_eval import score_window_branch


def train_window_model(model: nn.Module, param_groups: list[dict], dataset_cls, train: SplitIndex, dev: SplitIndex,
                       tune_utts: np.ndarray, settings: Settings, device: torch.device, ckpt_path: Path,
                       workers: int, log, dev_aug_p: float, *, epochs: int, batch_size: int, weight_decay: float,
                       seed: int, save, load, score_batch: int, max_grad_norm: float | None = None):
    sampler = RandomWindowSampler(train, settings.segment_samples, settings.data.rcnn_windows_per_utt,
                                  settings.window_samples, seed)
    loss_aug = LossAugmenter(LossConfig(p=settings.data.loss_p)) if settings.data.loss_p > 0 else None
    loader = DataLoader(dataset_cls(train, settings, settings.data.codec_aug_p, seed, loss=loss_aug),
                        batch_size=batch_size, sampler=sampler, num_workers=workers, drop_last=True,
                        worker_init_fn=_worker_init, prefetch_factor=4 if workers else None,
                        # Not persistent: kept alive, they doubled the worker count during each epoch's dev
                        # scoring, and run 2 died with a MemoryError there (~3-5 GB free RAM on this machine).
                        persistent_workers=False)
    labels = train.label[train.has_speech]
    n_spoof = int(labels.sum())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(labels) - n_spoof) / n_spoof, device=device))
    opt = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in param_groups],
                                                total_steps=epochs * len(loader))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    params = [p for g in param_groups for p in g["params"]]

    best, history = float("inf"), []
    for epoch in range(1, epochs + 1):
        model.train()
        sampler.set_epoch(epoch)
        t0, total, seen = time.time(), 0.0, 0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = loss_fn(model(x).float(), y)
            scaler.scale(loss).backward()
            if max_grad_norm:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(y)
            seen += len(y)
        horizon = score_window_branch(model, dev, tune_utts, settings, device, workers,
                                      grid=(settings.stream.window_seconds,), augment_p=dev_aug_p,
                                      dataset_cls=dataset_cls, batch_size=score_batch)
        eer = compute_metrics(dev.label[tune_utts], horizon[:, 0])["eer_pct"]
        history.append({"epoch": epoch, "loss": round(total / seen, 5), "dev_eer_pct": eer,
                        "minutes": round((time.time() - t0) / 60, 1)})
        log(f"  epoch {epoch}/{epochs}: {history[-1]}")
        if eer < best:
            best = eer
            save(model, ckpt_path)
    return load(ckpt_path), history
