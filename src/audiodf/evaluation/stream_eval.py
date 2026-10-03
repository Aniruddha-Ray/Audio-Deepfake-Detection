"""Score clips the way a live call is scored, and report what a deployment cares about.

For every clip: after speech onset, the SVM scores the buffer so far (a growing prefix) and the RCNN
scores each completed 2 s window (1 s hop); the verdict at time T fuses the SVM prefix with the mean
of the RCNN windows finished by T. Reported: EER at the 10 s decision horizon, EER vs time-to-decision
(2/4/6/8/10 s), per-attack EER, and per-codec EER (ASV5 eval carries 11 codec conditions).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from audiodf.config import Settings
from audiodf.data.prepare import RcnnWindowDataset, SplitIndex, _worker_init, build_svm_snapshots, stream_windows
from audiodf.evaluation.metrics import compute_metrics, per_attack_eer, per_group_eer
from audiodf.models.ensemble import fuse
from audiodf.models.rcnn import RCNN
from audiodf.models.svm import CalibratedSvm, predict_spoof_proba

EPS_S = 1e-3


@dataclass
class StreamScores:
    utts: np.ndarray  # clip indices into the SplitIndex
    label: np.ndarray
    attack: np.ndarray
    codec: np.ndarray
    grid: tuple  # seconds after speech onset
    svm: np.ndarray  # (n, len(grid)) P(spoof) from the buffer available at each time
    rcnn: np.ndarray  # (n, len(grid)) mean P(spoof) of the windows finished by each time

    def fused(self, svm_weight: float) -> np.ndarray:
        return fuse(self.svm, self.rcnn, svm_weight)


def svm_at_times(utt: np.ndarray, seconds: np.ndarray, probs: np.ndarray, utts: np.ndarray, grid) -> np.ndarray:
    """P(spoof) of the longest snapshot not exceeding each time T (the first one if all are longer).
    Snapshot lengths are capped by clip length, so a clip shorter than T simply uses all its audio."""
    order = np.lexsort((seconds, utt))
    utt_s, sec_s, p_s = utt[order], seconds[order], probs[order]
    starts = np.flatnonzero(np.r_[True, utt_s[1:] != utt_s[:-1]])
    ends = np.r_[starts[1:], len(utt_s)]
    out = np.empty((len(starts), len(grid)))
    for k, t in enumerate(grid):
        csum = np.cumsum(sec_s <= t + EPS_S)  # ok rows are a prefix of each clip's rows (sorted by length)
        n_ok = csum[ends - 1] - np.where(starts > 0, csum[starts - 1], 0)
        out[:, k] = p_s[starts + np.maximum(n_ok - 1, 0)]
    return out[np.searchsorted(utt_s[starts], utts)]


def rcnn_at_times(upos: np.ndarray, end_s: np.ndarray, probs: np.ndarray, n: int, grid) -> np.ndarray:
    """Mean window P(spoof) over windows ending by each T; before the first window ends, the first window."""
    first = probs[np.unique(upos, return_index=True)[1]]
    out = np.empty((n, len(grid)))
    for k, t in enumerate(grid):
        ok = end_s <= t + EPS_S
        cnt = np.bincount(upos[ok], minlength=n)
        total = np.bincount(upos[ok], weights=probs[ok], minlength=n)
        out[:, k] = np.where(cnt > 0, total / np.maximum(cnt, 1), first)
    return out


@torch.no_grad()
def score_windows(model: RCNN, idx: SplitIndex, keys: list, settings: Settings, device, workers: int,
                  augment_p: float = 0.0, batch_size: int = 256) -> np.ndarray:
    loader = DataLoader(RcnnWindowDataset(idx, settings, augment_p), batch_size=batch_size, sampler=keys,
                        num_workers=workers, worker_init_fn=_worker_init, prefetch_factor=8 if workers else None)
    model.eval()
    out = []
    for x, _ in loader:
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            out.append(torch.sigmoid(model(x.to(device, non_blocking=True)).float()).cpu().numpy())
    return np.concatenate(out)


def score_rcnn(model: RCNN, idx: SplitIndex, utts: np.ndarray, settings: Settings, device, workers: int,
               grid=None, augment_p: float = 0.0) -> np.ndarray:
    grid = grid or settings.data.svm_snapshot_seconds
    keys = stream_windows(idx, utts, settings)
    probs = score_windows(model, idx, keys, settings, device, workers, augment_p)
    pos = {int(u): i for i, u in enumerate(utts)}
    upos = np.array([pos[u] for u, _ in keys])
    rel = np.array([s - idx.speech_start[u] for u, s in keys])
    return rcnn_at_times(upos, (rel + settings.segment_samples) / settings.audio.sample_rate, probs, len(utts), grid)


def score_svm(svm: CalibratedSvm, idx: SplitIndex, utts: np.ndarray, settings: Settings, workers: int,
              tag: str, log=print, augment_p: float = 0.0) -> np.ndarray:
    data = build_svm_snapshots(idx, utts, settings, augment_p, tag, workers, log=log)
    probs = predict_spoof_proba(svm, data["x"], n_jobs=8)
    return svm_at_times(data["utt"], data["seconds"], probs, utts, settings.data.svm_snapshot_seconds)


def score_split(svm: CalibratedSvm, rcnn: RCNN, idx: SplitIndex, utts: np.ndarray, settings: Settings,
                device, workers: int, tag: str, log=print, augment_p: float = 0.0) -> StreamScores:
    grid = tuple(settings.data.svm_snapshot_seconds)
    return StreamScores(utts, idx.label[utts].astype(np.int64), idx.attack[utts], idx.codec[utts], grid,
                        score_svm(svm, idx, utts, settings, workers, tag, log, augment_p),
                        score_rcnn(rcnn, idx, utts, settings, device, workers, augment_p=augment_p))


def summarize(scores: StreamScores, svm_weight: float) -> dict:
    """Metrics at the decision horizon (last grid time) plus the time-to-decision curve."""
    fused = scores.fused(svm_weight)
    y, last = scores.label, -1
    branch = {"svm": scores.svm[:, last], "rcnn": scores.rcnn[:, last], "fused": fused[:, last]}
    out = {"n": len(y), "spoof_fraction": round(float(y.mean()), 4), "svm_weight": svm_weight,
           "at_horizon": {k: compute_metrics(y, v) for k, v in branch.items()},
           "per_attack_eer_pct": {k: per_attack_eer(y, v, scores.attack) for k, v in branch.items()},
           "time_to_decision_eer_pct": {
               name: {f"{t:g}s": compute_metrics(y, arr[:, i])["eer_pct"] for i, t in enumerate(scores.grid)}
               for name, arr in (("svm", scores.svm), ("rcnn", scores.rcnn), ("fused", fused))}}
    if len(set(scores.codec)) > 1:
        out["per_codec_eer_pct"] = {k: per_group_eer(y, v, scores.codec) for k, v in branch.items()}
    return out
