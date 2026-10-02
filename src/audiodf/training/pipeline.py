"""End-to-end offline pipeline: cache features -> train both branches -> evaluate ensemble -> save bundle."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import torch

from audiodf.artifacts import RCNN_FILE, bundle_dir, write_manifest
from audiodf.config import Settings
from audiodf.data.cache import SplitData, load_or_build
from audiodf.evaluation.metrics import compute_metrics, per_attack_eer
from audiodf.models.ensemble import fuse
from audiodf.models.rcnn import RCNN, load_rcnn, predict_spoof_proba as rcnn_predict
from audiodf.models.svm import predict_spoof_proba as svm_predict
from audiodf.training.train_rcnn import train_rcnn
from audiodf.training.train_svm import train_svm


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(svm, rcnn: RCNN, splits: dict[str, SplitData], settings: Settings, device) -> dict:
    """Utterance-level metrics per split. SVM scores the whole utterance; RCNN averages its 2 s windows."""
    report = {}
    for name in ("dev", "eval"):
        if name not in splits:
            continue
        d = splits[name]
        svm_p = svm_predict(svm, d.utt_svm, n_jobs=8)
        seg_p = rcnn_predict(rcnn, d.logmel, device)
        rcnn_p = d.segments_to_utterances(seg_p)
        fused = fuse(svm_p, rcnn_p, settings.ensemble.svm_weight)
        scores = {"svm": svm_p, "rcnn": rcnn_p, "ensemble": fused}
        report[name] = {k: compute_metrics(d.utt_label, v) for k, v in scores.items()}
        report[name]["rcnn_segment_level"] = compute_metrics(d.seg_label, seg_p)
        if name == "eval":
            report["eval_per_attack_eer_pct"] = {k: per_attack_eer(d.utt_label, v, d.utt_attack)
                                                 for k, v in scores.items()}
    return report


def run_training(settings: Settings, limit: int | None = None, workers: int = 8,
                 rcnn_checkpoint: Path | None = None, log=print) -> dict:
    device = pick_device()
    out = bundle_dir(settings)
    out.mkdir(parents=True, exist_ok=True)
    log(f"device={device} artifacts={out}")

    log("[1/4] features (SVM and RCNN extractors, cached)")
    splits = load_or_build(settings, limit, workers)

    log("[2/4] SVM branch: whole-utterance vectors")
    t0 = time.time()
    svm = train_svm(splits["train"], settings)
    log(f"  trained in {time.time() - t0:.0f}s")

    log("[3/4] RCNN branch: 2 s Log-Mel windows")
    history = []
    if rcnn_checkpoint:
        shutil.copy(rcnn_checkpoint, out / RCNN_FILE)
        rcnn = load_rcnn(out / RCNN_FILE, device, settings.rcnn_features.n_mels, settings.segment_frames,
                         settings.rcnn_model)
        log(f"  reused checkpoint {rcnn_checkpoint}")
    else:
        rcnn, history = train_rcnn(splits["train"], splits["dev"], settings, device, out / RCNN_FILE, log)

    log("[4/4] evaluation")
    report = evaluate(svm, rcnn, splits, settings, device)
    report["rcnn_history"] = history
    report["svm_weight"] = settings.ensemble.svm_weight
    write_manifest(settings, svm, rcnn, {k: report[k] for k in ("dev", "eval") if k in report})

    results = Path(settings.paths.results_dir)
    results.mkdir(parents=True, exist_ok=True)
    (results / f"training_report{f'_limit{limit}' if limit else ''}.json").write_text(json.dumps(report, indent=2))
    for split in ("dev", "eval"):
        if split in report:
            log(f"\n{split.upper()} (utterance level)")
            for name in ("svm", "rcnn", "ensemble"):
                m = report[split][name]
                log(f"  {name:9s} EER {m['eer_pct']:7.3f}%  AUC {m['auc']:.5f}  acc@EER {m['acc_at_eer_thr']:.5f}")
    return report
