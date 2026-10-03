"""End-to-end training on ASVspoof5: audit -> index -> train both branches -> tune on dev -> test -> bundle.

Data discipline: models fit on train only; everything tuned (RCNN epoch, SVM calibration, fusion weight,
risk thresholds) uses a stratified subset of dev, whose attacks differ from train's; test sets (ASV5
eval, and ASV2019 eval as a cross-dataset check) are scored once at the end and never feed back.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch

from audiodf.artifacts import RCNN_FILE, bundle_dir, load_bundle, write_manifest
from audiodf.config import Settings
from audiodf.data.integrity import audit_dataset
from audiodf.data.prepare import SplitIndex, build_index, build_svm_snapshots, stratified_subset
from audiodf.data.protocol import ASV5_SPLITS, ASV19_SPLITS
from audiodf.evaluation.metrics import compute_metrics
from audiodf.evaluation.stream_eval import StreamScores, score_split, summarize
from audiodf.training.train_rcnn import train_rcnn
from audiodf.training.train_svm import train_svm

TABLES = {"asv5": ASV5_SPLITS, "asv19": ASV19_SPLITS}


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _has_audio(settings: Settings, dataset: str, split: str) -> bool:
    return (Path(settings.dataset_root(dataset)) / TABLES[dataset][split][1]).is_dir()


def tune_fusion(scores: StreamScores) -> tuple[float, dict]:
    """SVM weight minimising the mean fused EER over every time-to-decision (2..10 s): a live call
    is decided at each of them, not only at the end. Weights within `tie` EER points of the best are
    treated as equal and the one closest to 0.5 wins, so the system doesn't lean on one branch
    without evidence (dev has few unseen attacks, and dev-tuned weights generalised badly before)."""
    grid = np.round(np.linspace(0.0, 1.0, 21), 2)
    mean_eer = {}
    for w in grid:
        fused = scores.fused(float(w))
        mean_eer[float(w)] = float(np.mean([compute_metrics(scores.label, fused[:, k])["eer_pct"]
                                            for k in range(fused.shape[1])]))
    tie = 0.05
    near_best = [w for w, v in mean_eer.items() if v <= min(mean_eer.values()) + tie]
    best = min(near_best, key=lambda w: abs(w - 0.5))
    return best, {f"{w:.2f}": round(v, 3) for w, v in mean_eer.items()}


def operating_points(fused: np.ndarray, y: np.ndarray, settings: Settings) -> tuple[float, float, list[dict]]:
    """Risk thresholds from the dev score distribution of bonafide clips (see RiskConfig)."""
    bona, spoof = fused[y == 0], fused[y == 1]
    thresholds = {f: float(np.quantile(bona, 1 - f)) for f in (settings.risk.block_fpr, settings.risk.verify_fpr)}
    table = [{"bonafide_flagged": f, "threshold": round(t, 5), "spoof_caught": round(float((spoof >= t).mean()), 4)}
             for f, t in sorted(thresholds.items())]
    return thresholds[settings.risk.block_fpr], thresholds[settings.risk.verify_fpr], table


def _subset(idx: SplitIndex, n: int, seed: int = 0) -> np.ndarray:
    ok = int(idx.has_speech.sum())
    return np.nonzero(idx.has_speech)[0] if not n or n >= ok else stratified_subset(idx, n, seed)


def evaluate_on(svm, rcnn, settings: Settings, dataset: str, split: str, n_utts: int, device, workers: int,
                limit: int | None = None, log=print) -> dict:
    idx = build_index(settings, dataset, split, limit, workers, log, available_only=True)  # test splits may be partial
    utts = _subset(idx, n_utts)
    log(f"  scoring {dataset}/{split}: {len(utts)} of {len(idx)} clips")
    scores = score_split(svm, rcnn, idx, utts, settings, device, workers, f"{split}{len(utts)}", log)
    return summarize(scores, settings.ensemble.svm_weight)


def _print_summary(name: str, rep: dict, log) -> None:
    h = rep["at_horizon"]
    log(f"  {name:12s} n={rep['n']:6d} | EER at 10 s: svm {h['svm']['eer_pct']:6.2f}%  rcnn {h['rcnn']['eer_pct']:6.2f}%  "
        f"fused {h['fused']['eer_pct']:6.2f}%")
    t = rep["time_to_decision_eer_pct"]["fused"]
    log(f"  {'':12s} fused EER by seconds of speech: " + "  ".join(f"{k} {v:.2f}%" for k, v in t.items()))


def run_training(settings: Settings, limit: int | None = None, workers: int = 8, log=print) -> dict:
    device = pick_device()
    out = bundle_dir(settings)
    out.mkdir(parents=True, exist_ok=True)
    log(f"device={device} artifacts={out}")

    sources = tuple(settings.data.train_datasets)
    log(f"[1/6] integrity audit (training on: {' + '.join(sources)})")
    for dataset in dict.fromkeys(("asv5",) + sources):  # ASV5 dev is always the tuning set
        audit = audit_dataset(settings, dataset, sample_n=500, silence_n=100)
        if not audit["ok"]:
            raise RuntimeError(f"{dataset} data integrity audit failed:\n  " + "\n  ".join(audit["errors"]))
        for finding in audit["findings"]:
            log(f"  finding ({dataset}): {finding}")

    log("[2/6] index clips (decode once, record VAD speech bounds)")
    parts = [build_index(settings, "asv5", "train", limit, workers, log)] if "asv5" in sources else []
    if "asv19" in sources:
        parts += [build_index(settings, "asv19", split, limit, workers, log) for split in ("train", "dev")]
    train = parts[0] if len(parts) == 1 else SplitIndex.concat(parts)
    dev = build_index(settings, "asv5", "dev", limit, workers, log)
    log(f"  training clips: {len(train)} ({int(train.label.sum())} spoof, {int((1 - train.label).sum())} bonafide)")
    tune = _subset(dev, min(settings.data.tune_utts, len(dev)))

    log("[3/6] SVM branch: growing-buffer snapshots, codec-augmented")
    svm = train_svm(train, settings, workers, n_utts=min(settings.data.svm_train_utts, len(train)), log=log)

    log("[4/6] RCNN branch: random 2 s windows read from FLAC, codec-augmented")
    rcnn, history = train_rcnn(train, dev, tune, settings, device, out / RCNN_FILE, workers, log)

    log("[5/6] tune on dev (codec-augmented): SVM calibration, fusion weight, risk thresholds")
    aug = settings.data.tune_aug_p  # label-blind codec augmentation at about eval's codec rate
    snaps = build_svm_snapshots(dev, tune, settings, aug, f"dev{len(tune)}", workers, log=log)
    svm.fit_calibrator(snaps["x"], snaps["label"])
    dev_scores = score_split(svm, rcnn, dev, tune, settings, device, workers, f"dev{len(tune)}", log, aug)
    weight, weight_curve = tune_fusion(dev_scores)
    settings.ensemble.svm_weight = weight
    high, medium, table = operating_points(dev_scores.fused(weight)[:, -1], dev_scores.label, settings)
    settings.risk.high, settings.risk.medium = high, medium
    log(f"  svm weight {weight:.2f}; risk thresholds: verify >= {medium:.4f}, block >= {high:.4f}")
    report = {"rcnn_history": history, "svm_weight": weight, "svm_weight_mean_eer_curve": weight_curve,
              "risk": {"high": high, "medium": medium, "operating_points_dev": table},
              "dev_tuning_subset": summarize(dev_scores, weight), "test": {}}
    _print_summary("dev (tuned)", report["dev_tuning_subset"], log)

    log("[6/6] test sets (scored once, never tuned on)")
    for dataset, split in (("asv5", "eval"), ("asv19", "eval")):
        if not _has_audio(settings, dataset, split):
            log(f"  skipping {dataset}/{split}: audio not found under {settings.dataset_root(dataset)}")
            continue
        rep = evaluate_on(svm, rcnn, settings, dataset, split, settings.data.eval_utts, device, workers, limit, log)
        report["test"][f"{dataset}/{split}"] = rep
        _print_summary(f"{dataset}/{split}", rep, log)

    write_manifest(settings, svm, rcnn, {"svm_weight": weight, "dev_eer_pct": report["dev_tuning_subset"][
        "at_horizon"]["fused"]["eer_pct"], "test_eer_pct": {k: v["at_horizon"]["fused"]["eer_pct"]
                                                              for k, v in report["test"].items()}})
    results = Path(settings.paths.results_dir)
    results.mkdir(parents=True, exist_ok=True)
    path = results / f"training_report{f'_limit{limit}' if limit else ''}.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    log(f"report: {path}")
    return report


def evaluate_artifacts(settings: Settings, dataset: str, split: str, n_utts: int, workers: int,
                       limit: int | None = None, log=print) -> dict:
    """Score a saved bundle on any dataset/split with its own tuned fusion weight."""
    device = pick_device()
    bundle = load_bundle(settings, device)
    settings = copy.deepcopy(settings)
    settings.ensemble.svm_weight = bundle.manifest["svm_weight"]
    rep = evaluate_on(bundle.svm, bundle.rcnn, settings, dataset, split, n_utts, device, workers, limit, log)
    _print_summary(f"{dataset}/{split}", rep, log)
    return rep
