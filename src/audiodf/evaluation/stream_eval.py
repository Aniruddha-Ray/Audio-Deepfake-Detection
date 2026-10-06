"""Score clips the way a live call is scored, and report what a deployment cares about.

For every clip: after speech onset, the SVM scores the buffer so far (a growing prefix) and each window branch
(RCNN on Log-Mel, WavLM on the raw waveform) scores each completed 2 s window (1 s hop); the verdict at time T
fuses the SVM prefix with the mean of each window branch's windows finished by T. Reported: EER at the 10 s
decision horizon, EER vs time-to-decision (2/4/6/8/10 s), per-attack EER, and per-codec EER (ASV5 eval carries
11 codec conditions).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader

from audiodf.config import Settings
from audiodf.data.prepare import (RcnnWindowDataset, SplitIndex, WaveWindowDataset, _worker_init,
                                  build_svm_snapshots, stream_windows)
from audiodf.evaluation.metrics import compute_metrics, per_attack_eer, per_group_eer
from audiodf.models.ensemble import fuse
from audiodf.models.svm import CalibratedSvm, predict_spoof_proba

EPS_S = 1e-3
# Window branches: the dataset that feeds them and the scoring batch size (WavLM: 4.3 GB VRAM at 32).
WINDOW_BRANCHES = {"rcnn": (RcnnWindowDataset, 256), "wavlm": (WaveWindowDataset, 32)}


@dataclass
class StreamScores:
    utts: np.ndarray  # clip indices into the SplitIndex
    label: np.ndarray
    attack: np.ndarray
    codec: np.ndarray
    grid: tuple  # seconds after speech onset
    # branch name -> (n, len(grid)) P(spoof): the SVM from the buffer available at each time, window branches the
    # mean of the windows finished by each time
    branches: dict = field(default_factory=dict)

    def fused(self, weights: dict) -> np.ndarray:
        return fuse(self.branches, weights)


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
    """Mean window P(spoof) over windows ending by each T; before the first window ends, the first window.
    Used for every window branch."""
    first = probs[np.unique(upos, return_index=True)[1]]
    out = np.empty((n, len(grid)))
    for k, t in enumerate(grid):
        ok = end_s <= t + EPS_S
        cnt = np.bincount(upos[ok], minlength=n)
        total = np.bincount(upos[ok], weights=probs[ok], minlength=n)
        out[:, k] = np.where(cnt > 0, total / np.maximum(cnt, 1), first)
    return out


@torch.no_grad()
def score_windows(model, idx: SplitIndex, keys: list, settings: Settings, device, workers: int,
                  augment_p: float = 0.0, batch_size: int = 256, dataset_cls=RcnnWindowDataset) -> np.ndarray:
    loader = DataLoader(dataset_cls(idx, settings, augment_p), batch_size=batch_size, sampler=keys,
                        num_workers=workers, worker_init_fn=_worker_init, prefetch_factor=8 if workers else None)
    model.eval()
    out = []
    for x, _ in loader:
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            out.append(torch.sigmoid(model(x.to(device, non_blocking=True)).float()).cpu().numpy())
    return np.concatenate(out)


def score_window_branch(model, idx: SplitIndex, utts: np.ndarray, settings: Settings, device, workers: int,
                        grid=None, augment_p: float = 0.0, dataset_cls=RcnnWindowDataset,
                        batch_size: int = 256) -> np.ndarray:
    grid = grid or settings.data.svm_snapshot_seconds
    keys = stream_windows(idx, utts, settings)
    probs = score_windows(model, idx, keys, settings, device, workers, augment_p, batch_size, dataset_cls)
    pos = {int(u): i for i, u in enumerate(utts)}
    upos = np.array([pos[u] for u, _ in keys])
    rel = np.array([s - idx.speech_start[u] for u, s in keys])
    return rcnn_at_times(upos, (rel + settings.segment_samples) / settings.audio.sample_rate, probs, len(utts), grid)


def score_rcnn(model, idx: SplitIndex, utts: np.ndarray, settings: Settings, device, workers: int,
               grid=None, augment_p: float = 0.0) -> np.ndarray:
    return score_window_branch(model, idx, utts, settings, device, workers, grid, augment_p)


def score_svm(svm: CalibratedSvm, idx: SplitIndex, utts: np.ndarray, settings: Settings, workers: int,
              tag: str, log=print, augment_p: float = 0.0) -> np.ndarray:
    data = build_svm_snapshots(idx, utts, settings, augment_p, tag, workers, log=log)
    probs = predict_spoof_proba(svm, data["x"], n_jobs=8)
    return svm_at_times(data["utt"], data["seconds"], probs, utts, settings.data.svm_snapshot_seconds)


def score_split(models: dict, idx: SplitIndex, utts: np.ndarray, settings: Settings, device, workers: int,
                tag: str, log=print, augment_p: float = 0.0) -> StreamScores:
    """models: branch name -> trained model ("svm", "rcnn", "wavlm"); every branch given is scored."""
    grid = tuple(settings.data.svm_snapshot_seconds)
    branches = {}
    for name, model in models.items():
        log(f"  scoring {name} on {len(utts)} clips")
        if name == "svm":
            branches[name] = score_svm(model, idx, utts, settings, workers, tag, log, augment_p)
        else:
            dataset_cls, batch = WINDOW_BRANCHES[name]
            branches[name] = score_window_branch(model, idx, utts, settings, device, workers, grid, augment_p,
                                                 dataset_cls, batch)
    return StreamScores(utts, idx.label[utts].astype(np.int64), idx.attack[utts], idx.codec[utts], grid, branches)


def operating_point_report(scores: StreamScores, weights: dict, risk: dict) -> dict:
    """What the stored risk thresholds do on this data, at the decision horizon: the share of genuine clips flagged
    (verify / block) and of fakes caught, overall and per codec condition. `risk` = {"high": block, "medium": verify}.
    Only meaningful for the weights the thresholds were tuned with."""
    from audiodf.calibrate import threshold_rates

    out = threshold_rates(scores.fused(weights)[:, -1], scores.label, scores.codec, risk)
    if "per_group" in out:
        out["per_codec"] = out.pop("per_group")
    return out


def write_scores(scores: StreamScores, weights: dict, path) -> None:
    """One row per clip: ids, label, attack, codec condition, and every branch's (and the fused) P(spoof) at each
    time-to-decision, so any further breakdown (e.g. by transmission route) needs no re-scoring."""
    import csv

    cols = {**scores.branches, "fused": scores.fused(weights)}
    head = ["clip_index", "label", "attack", "codec"] + [f"{b}_{t:g}s" for b in cols for t in scores.grid]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(head)
        for i, u in enumerate(scores.utts):
            w.writerow([int(u), int(scores.label[i]), scores.attack[i], scores.codec[i]]
                       + [f"{cols[b][i, k]:.5f}" for b in cols for k in range(len(scores.grid))])


def summarize(scores: StreamScores, weights: dict) -> dict:
    """Metrics at the decision horizon (last grid time) plus the time-to-decision curve."""
    fused = scores.fused(weights)
    y, last = scores.label, -1
    curves = {**scores.branches, "fused": fused}
    branch = {k: v[:, last] for k, v in curves.items()}
    out = {"n": len(y), "spoof_fraction": round(float(y.mean()), 4), "fusion_weights": dict(weights),
           "at_horizon": {k: compute_metrics(y, v) for k, v in branch.items()},
           "per_attack_eer_pct": {k: per_attack_eer(y, v, scores.attack) for k, v in branch.items()},
           "time_to_decision_eer_pct": {
               name: {f"{t:g}s": compute_metrics(y, arr[:, i])["eer_pct"] for i, t in enumerate(scores.grid)}
               for name, arr in curves.items()}}
    if len(set(scores.codec)) > 1:
        out["per_codec_eer_pct"] = {k: per_group_eer(y, v, scores.codec) for k, v in branch.items()}
    return out
