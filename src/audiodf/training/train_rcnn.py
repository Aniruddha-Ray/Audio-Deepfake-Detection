"""RCNN branch training: 2 s Log-Mel windows, one-cycle LR, class-weighted BCE, best-dev-EER checkpoint."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from audiodf.config import Settings
from audiodf.data.cache import SplitData
from audiodf.evaluation.metrics import compute_metrics
from audiodf.models.rcnn import RCNN, load_rcnn, predict_spoof_proba, save_rcnn


def _batches(split: SplitData, order: np.ndarray, batch_size: int):
    for i in range(0, len(order), batch_size):
        idx = np.sort(order[i:i + batch_size])  # sorted reads are much faster on the memmap
        yield torch.from_numpy(split.logmel[idx].astype(np.float32)).unsqueeze(1), idx


def train_rcnn(train: SplitData, dev: SplitData, settings: Settings, device: torch.device,
               ckpt_path: Path, log=print) -> tuple[RCNN, list[dict]]:
    cfg = settings.rcnn_train
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = RCNN(settings.rcnn_features.n_mels, settings.segment_frames, settings.rcnn_model).to(device)

    labels = torch.from_numpy(train.seg_label.astype(np.float32))
    n_spoof = int(labels.sum())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(labels) - n_spoof) / n_spoof, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * int(np.ceil(train.n_seg / cfg.batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=steps)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    best, history = float("inf"), []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for x, idx in _batches(train, rng.permutation(train.n_seg), cfg.batch_size):
            x, y = x.to(device, non_blocking=True), labels[idx].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = loss_fn(model(x).float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(idx)
        dev_scores = dev.segments_to_utterances(predict_spoof_proba(model, dev.logmel, device))
        dev_eer = compute_metrics(dev.utt_label, dev_scores)["eer_pct"]
        history.append({"epoch": epoch, "loss": round(total / train.n_seg, 5),
                        "dev_utt_eer_pct": dev_eer, "sec": round(time.time() - t0)})
        log(f"  epoch {epoch}/{cfg.epochs}: {history[-1]}")
        if dev_eer < best:
            best = dev_eer
            save_rcnn(model, ckpt_path)
    return load_rcnn(ckpt_path, device), history
