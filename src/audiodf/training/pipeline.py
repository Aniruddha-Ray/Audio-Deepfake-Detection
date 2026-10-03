"""End-to-end training: audit -> index -> train the branches -> tune on the held-out set -> test -> bundle.

Data discipline: models fit on the training pool only; everything tuned (best epoch per window branch, SVM
calibration, fusion weights, risk thresholds) uses the held-out tuning set, whose attacks and speakers are not
trained on; test sets (ASV5 eval, and ASV2019 eval as a cross-dataset check) are scored once at the end and never
feed back.
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from audiodf.artifacts import BRANCH_FILES, bundle_dir, load_bundle, write_manifest
from audiodf.config import Settings
from audiodf.data.integrity import audit_dataset
from audiodf.data.ffmpeg_codecs import render_copies
from audiodf.data.prepare import SplitIndex, build_index, build_svm_snapshots, holdout_split, stratified_subset
from audiodf.data.protocol import ASV5_SPLITS, ASV19_SPLITS
from audiodf.evaluation.metrics import compute_metrics
from audiodf.evaluation.stream_eval import StreamScores, score_split, summarize
from audiodf.models.ensemble import BRANCHES, simplex_grid
from audiodf.models.rcnn import load_rcnn
from audiodf.models.svm import load_svm
from audiodf.training.train_rcnn import train_rcnn
from audiodf.training.train_svm import train_svm

TABLES = {"asv5": ASV5_SPLITS, "asv19": ASV19_SPLITS}


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _has_audio(settings: Settings, dataset: str, split: str) -> bool:
    return (Path(settings.dataset_root(dataset)) / TABLES[dataset][split][1]).is_dir()


def _weights_key(weights: dict) -> str:
    return " ".join(f"{b}={w:.2f}" for b, w in weights.items())


def tune_fusion(scores: StreamScores, step: float = 0.05, tie: float = 0.05) -> tuple[dict, dict]:
    """Fusion weights (on a `step` simplex grid over the scored branches) minimising the mean fused EER over every
    time-to-decision (2..10 s): a live call is decided at each of them, not only at the end. Weightings within
    `tie` EER points of the best are treated as equal and the one closest to equal weights wins, so the system
    doesn't lean on one branch without evidence (the tuning set has few unseen attacks, and tuned weights
    generalised badly before). Returns (weights, {weighting: mean EER} for the 20 best and each single branch)."""
    branches = list(scores.branches)
    mean_eer = {}
    for weights in simplex_grid(branches, step):
        fused = scores.fused(weights)
        mean_eer[_weights_key(weights)] = (weights, float(np.mean(
            [compute_metrics(scores.label, fused[:, k])["eer_pct"] for k in range(fused.shape[1])])))
    floor = min(v for _, v in mean_eer.values())
    uniform = 1 / len(branches)
    near_best = [w for w, v in mean_eer.values() if v <= floor + tie]
    best = min(near_best, key=lambda w: sum((x - uniform) ** 2 for x in w.values()))
    ranked = sorted(mean_eer.items(), key=lambda kv: kv[1][1])
    curve = {k: round(v, 3) for k, (_, v) in ranked[:20]}
    curve.update({k: round(v, 3) for k, (w, v) in mean_eer.items() if max(w.values()) == 1.0})
    return best, curve


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


def evaluate_on(models: dict, settings: Settings, dataset: str, split: str, n_utts: int, device, workers: int,
                limit: int | None = None, log=print) -> dict:
    idx = build_index(settings, dataset, split, limit, workers, log, available_only=True)  # test splits may be partial
    utts = _subset(idx, n_utts)
    log(f"  scoring {dataset}/{split}: {len(utts)} of {len(idx)} clips")
    scores = score_split(models, idx, utts, settings, device, workers, f"{split}{len(utts)}", log)
    return summarize(scores, settings.ensemble.weights)


def _print_summary(name: str, rep: dict, log) -> None:
    h = rep["at_horizon"]
    log(f"  {name:12s} n={rep['n']:6d} | EER at 10 s: "
        + "  ".join(f"{b} {v['eer_pct']:6.2f}%" for b, v in h.items()))
    t = rep["time_to_decision_eer_pct"]["fused"]
    log(f"  {'':12s} fused EER by seconds of speech: " + "  ".join(f"{k} {v:.2f}%" for k, v in t.items()))


def training_splits(settings: Settings) -> list[tuple[str, str]]:
    """Parse and check data.train_splits: test splits are refused, and ASV5 dev may only be trained on
    when attacks are held out for tuning (otherwise it is the tuning set)."""
    splits = []
    for item in settings.data.train_splits:
        dataset, _, split = item.partition(":")
        if dataset not in TABLES or split not in ("train", "dev"):
            raise ValueError(f"train_splits entry {item!r}: expected asv5|asv19 : train|dev (eval splits are tests)")
        splits.append((dataset, split))
    if not settings.data.holdout_attacks and ("asv5", "dev") in splits:
        raise ValueError("asv5:dev is the tuning set when no attacks are held out; remove it from train_splits "
                         "or set data.holdout_attacks")
    return splits


def build_training_pool(settings: Settings, splits, limit, workers, log) -> tuple[SplitIndex, SplitIndex, str]:
    """(train, tuning, description). With held-out attacks the tuning set is cut out of the pool; otherwise
    it is ASV5 dev."""
    pool = SplitIndex.concat([build_index(settings, d, s, limit, workers, log) for d, s in splits])
    held = tuple(settings.data.holdout_attacks)
    if not held:
        return pool, build_index(settings, "asv5", "dev", limit, workers, log), "ASV5 dev (all attacks)"
    train_mask, tune_mask, n_spk = holdout_split(pool, held, settings.data.holdout_speaker_source,
                                                 settings.data.holdout_speaker_frac)
    tune = pool.subset(tune_mask, "tune")
    desc = (f"held-out attacks {', '.join(held)} + bonafide of {n_spk} held-out "
            f"{settings.data.holdout_speaker_source} speakers: {len(tune)} clips "
            f"({int(tune.label.sum())} spoof, {int((1 - tune.label).sum())} bonafide)")
    return pool.subset(train_mask, "train"), tune, desc


def _reuse(name: str, path: Path, out: Path, device, log, settings: Settings):
    """Load an already-trained branch (e.g. from an earlier run on the same pool) and copy it into the bundle."""
    target = out / BRANCH_FILES[name]
    if name != "svm" and Path(path).resolve() != target.resolve():  # the SVM is re-saved after calibration
        shutil.copyfile(path, target)
    log(f"  reusing {name} from {path} (training skipped)")
    history = [{"reused_checkpoint": str(path)}]
    if name == "svm":
        return load_svm(path), history
    if name == "rcnn":
        return load_rcnn(target, device), history
    from audiodf.models.wavlm import load_wavlm

    return load_wavlm(target, device, settings.wavlm.pretrained), history


def run_training(settings: Settings, limit: int | None = None, workers: int = 8, log=print,
                 checkpoints: dict | None = None) -> dict:
    """Trains the branches in settings.ensemble.branches. checkpoints: {branch: path} of already-trained branches
    to reuse instead of training (e.g. the best epoch of an interrupted run, or the SVM and RCNN of an earlier
    run on the same pool when only a new branch is added); every other stage runs as usual. A reused SVM is
    re-calibrated on the tuning set."""
    checkpoints = {k: v for k, v in (checkpoints or {}).items() if v}
    branches = [b for b in BRANCHES if b in settings.ensemble.branches]
    unknown = (set(settings.ensemble.branches) - set(BRANCHES)) | (set(checkpoints) - set(branches))
    if unknown or not branches:
        raise ValueError(f"branches {sorted(settings.ensemble.branches)} / checkpoints {sorted(checkpoints)}: "
                         f"expected a non-empty subset of {BRANCHES}, with checkpoints only for trained branches")
    device = pick_device()
    out = bundle_dir(settings)
    out.mkdir(parents=True, exist_ok=True)
    log(f"device={device} artifacts={out} branches={','.join(branches)}")

    splits = training_splits(settings)
    log(f"[1/6] integrity audit (training pool: {', '.join(f'{d}:{s}' for d, s in splits)})")
    for dataset in dict.fromkeys(["asv5"] + [d for d, _ in splits]):  # ASV5 always provides tuning or test data
        audit = audit_dataset(settings, dataset, sample_n=500, silence_n=100)
        if not audit["ok"]:
            raise RuntimeError(f"{dataset} data integrity audit failed:\n  " + "\n  ".join(audit["errors"]))
        for finding in audit["findings"]:
            log(f"  finding ({dataset}): {finding}")

    log("[2/6] index clips (decode once, record VAD speech bounds) and split off the tuning set")
    train, dev, tuning_set = build_training_pool(settings, splits, limit, workers, log)
    log(f"  training clips: {len(train)} ({int(train.label.sum())} spoof, {int((1 - train.label).sum())} bonafide), "
        f"{len(set(train.attack) - {'-'})} attack systems")
    log(f"  tuning set: {tuning_set}")
    tune = _subset(dev, min(settings.data.tune_utts, len(dev)))
    tune_aug = settings.data.tune_aug_p  # simulated codecs for tuning clips, unless real-codec copies replace them
    if settings.data.ffmpeg_codecs:
        log("  rendering real-codec copies (ffmpeg; label-blind; cached on disk)")
        train = render_copies(train, np.nonzero(train.has_speech)[0], settings, settings.data.render_frac,
                              seed=1, log=log)
        dev = render_copies(dev, tune, settings, settings.data.tune_render_frac, seed=2, log=log)
        tune_aug = 0.0
        log(f"  real-codec copies: {int(train.rendered.sum())} training clips, {int(dev.rendered[tune].sum())} of "
            f"{len(tune)} tuning clips")

    models, histories = {}, {}
    log("[3/6] branches")
    for name in branches:
        if name in checkpoints:
            models[name], histories[name] = _reuse(name, checkpoints[name], out, device, log, settings)
        elif name == "svm":
            log("  SVM: growing-buffer snapshots, codec-augmented")
            models[name] = train_svm(train, settings, workers, n_utts=min(settings.data.svm_train_utts, len(train)),
                                     log=log)
            histories[name] = []
        elif name == "rcnn":
            log("  RCNN: random 2 s Log-Mel windows read from FLAC, codec-augmented; best tuning epoch kept")
            models[name], histories[name] = train_rcnn(train, dev, tune, settings, device, out / BRANCH_FILES[name],
                                                       workers, log, tune_aug)
        else:
            from audiodf.training.train_wavlm import train_wavlm

            log(f"  WavLM: random 2 s raw windows, top {settings.wavlm.finetune_top} layers fine-tuned; "
                f"best tuning epoch kept")
            models[name], histories[name] = train_wavlm(train, dev, tune, settings, device,
                                                        out / BRANCH_FILES[name], workers, log, tune_aug)

    log("[4/6] tune on the held-out set (codec-processed): SVM calibration, fusion weights, risk thresholds")
    if "svm" in models:
        snaps = build_svm_snapshots(dev, tune, settings, tune_aug, f"tune{len(tune)}", workers, log=log)
        models["svm"].fit_calibrator(snaps["x"], snaps["label"])
    dev_scores = score_split(models, dev, tune, settings, device, workers, f"tune{len(tune)}", log, tune_aug)
    weights, weight_curve = tune_fusion(dev_scores)
    settings.ensemble.weights = weights
    high, medium, table = operating_points(dev_scores.fused(weights)[:, -1], dev_scores.label, settings)
    settings.risk.high, settings.risk.medium = high, medium
    log(f"  fusion weights {_weights_key(weights)}; risk thresholds: verify >= {medium:.4f}, block >= {high:.4f}")
    report = {"training_pool": [f"{d}:{s}" for d, s in splits], "training_clips": len(train),
              "tuning_set": tuning_set, "branches": branches, "histories": histories, "fusion_weights": weights,
              "fusion_mean_eer_curve": weight_curve,
              "risk": {"high": high, "medium": medium, "operating_points_tuning": table},
              "tuning_subset": summarize(dev_scores, weights), "test": {}}
    _print_summary("tuning set", report["tuning_subset"], log)

    log("[5/6] test sets (scored once, never tuned on)")
    for dataset, split in (("asv5", "eval"), ("asv19", "eval")):
        if not _has_audio(settings, dataset, split):
            log(f"  skipping {dataset}/{split}: audio not found under {settings.dataset_root(dataset)}")
            continue
        rep = evaluate_on(models, settings, dataset, split, settings.data.eval_utts, device, workers, limit, log)
        report["test"][f"{dataset}/{split}"] = rep
        _print_summary(f"{dataset}/{split}", rep, log)

    log("[6/6] save the bundle")
    write_manifest(settings, models, {"training_pool": report["training_pool"], "tuning_set": tuning_set,
                                      "tuning_eer_pct": report["tuning_subset"]["at_horizon"]["fused"]["eer_pct"],
                                      "test_eer_pct": {k: v["at_horizon"]["fused"]["eer_pct"]
                                                       for k, v in report["test"].items()}})
    results = Path(settings.paths.results_dir)
    results.mkdir(parents=True, exist_ok=True)
    path = results / f"training_report{f'_limit{limit}' if limit else ''}.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    log(f"report: {path}")
    return report


def evaluate_artifacts(settings: Settings, dataset: str, split: str, n_utts: int, workers: int,
                       limit: int | None = None, log=print) -> dict:
    """Score a saved bundle on any dataset/split with its own tuned fusion weights (every branch is scored, so
    the report shows each one, including branches the fusion gave no weight)."""
    device = pick_device()
    bundle = load_bundle(settings, device)
    settings = copy.deepcopy(settings)
    settings.ensemble.weights = bundle.weights
    rep = evaluate_on(bundle.models, settings, dataset, split, n_utts, device, workers, limit, log)
    _print_summary(f"{dataset}/{split}", rep, log)
    return rep
