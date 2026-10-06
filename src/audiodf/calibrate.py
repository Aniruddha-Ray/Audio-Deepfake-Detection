"""Set the verify / block thresholds on phone-channel data and check them on other speakers, then bundle one branch.

Thresholds are set with the same policy as in training (RiskConfig: `block_fpr` / `verify_fpr` of genuine clips
flagged, pooled over channels), but on audio that went through real phone channels. The scored clips are split **by
speaker** into two halves; thresholds come from half A and are checked on half B, so no voice is used for both.

  python -m audiodf calibrate --scores results/evaluate_asv21_eval_run4_wavlm_scores.csv \
      --protocol dataset21/ASVspoof2021.LA.eval.tsv --src-bundle artifacts_run4_wavlm --out-bundle artifacts_run4_wavlm_only
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

MANIFEST, WAVLM_FILE = "bundle.json", "wavlm.pt"


def speaker_halves(speakers: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks (A, B): the speakers are shuffled and split in two, so no speaker is in both."""
    names = np.array(sorted(set(speakers)))
    half = set(np.random.default_rng(seed).permutation(names)[: len(names) // 2])
    a = np.array([s in half for s in speakers])
    return a, ~a


def policy_thresholds(scores: np.ndarray, labels: np.ndarray, block_fpr: float = 0.01,
                      verify_fpr: float = 0.10) -> dict:
    """{"high": block, "medium": verify}: the scores that `block_fpr` / `verify_fpr` of genuine clips reach or exceed."""
    bona = np.asarray(scores)[np.asarray(labels) == 0]
    if len(bona) < 10 / min(block_fpr, verify_fpr):
        raise ValueError(f"{len(bona)} genuine clips are too few to place a {min(block_fpr, verify_fpr):.0%} threshold")
    return {"high": float(np.quantile(bona, 1 - block_fpr)), "medium": float(np.quantile(bona, 1 - verify_fpr))}


def threshold_rates(p: np.ndarray, y: np.ndarray, groups: np.ndarray | None, risk: dict) -> dict:
    """Genuine clips flagged and fakes caught at the verify ("medium") and block ("high") thresholds, overall and per
    group (e.g. codec condition)."""

    def rates(mask):
        bona, spoof = p[mask & (y == 0)], p[mask & (y == 1)]
        return {name: {"threshold": round(float(risk[key]), 5),
                       "bonafide_flagged": round(float((bona >= risk[key]).mean()), 4) if len(bona) else None,
                       "spoof_caught": round(float((spoof >= risk[key]).mean()), 4) if len(spoof) else None}
                for name, key in (("verify", "medium"), ("block", "high"))}

    out = {"overall": rates(np.ones(len(y), dtype=bool))}
    if groups is not None and len(set(groups)) > 1:
        out["per_group"] = {str(g): rates(groups == g) for g in sorted(set(groups))}
    return out


def calibrate(scores: np.ndarray, labels: np.ndarray, speakers: np.ndarray, groups: np.ndarray | None = None,
              block_fpr: float = 0.01, verify_fpr: float = 0.10, seed: int = 0, stored: dict | None = None) -> dict:
    """Thresholds from half A, rates on both halves (and, for reference, at `stored` thresholds on half B)."""
    a, b = speaker_halves(speakers, seed)
    risk = policy_thresholds(scores[a], labels[a], block_fpr, verify_fpr)
    sub = lambda m: None if groups is None else groups[m]  # noqa: E731
    out = {"policy": {"block_fpr": block_fpr, "verify_fpr": verify_fpr}, "seed": seed, "risk": risk,
           "half_a": {"clips": int(a.sum()), "speakers": len(set(speakers[a])),
                      **threshold_rates(scores[a], labels[a], sub(a), risk)},
           "half_b": {"clips": int(b.sum()), "speakers": len(set(speakers[b])),
                      **threshold_rates(scores[b], labels[b], sub(b), risk)}}
    if stored:
        out["half_b_at_stored_thresholds"] = threshold_rates(scores[b], labels[b], sub(b), stored)
    return out


def repeated_check(scores: np.ndarray, labels: np.ndarray, speakers: np.ndarray, n: int = 20, block_fpr: float = 0.01,
                   verify_fpr: float = 0.10, seed: int = 0) -> dict:
    """How well thresholds set on half the speakers hold on the other half, over `n` random speaker splits: the mean and
    spread (std, min, max) of the genuine clips flagged and fakes caught on the held-out half. One split is one draw; the
    spread says how much the rates move with which speakers happen to be in the calibration half."""
    rows = []
    for k in range(n):
        a, b = speaker_halves(speakers, seed + k)
        risk = policy_thresholds(scores[a], labels[a], block_fpr, verify_fpr)
        r = threshold_rates(scores[b], labels[b], None, risk)["overall"]
        rows.append([r["verify"]["bonafide_flagged"], r["verify"]["spoof_caught"],
                     r["block"]["bonafide_flagged"], r["block"]["spoof_caught"]])
    rows = np.array(rows)
    names = ("verify_bonafide_flagged", "verify_spoof_caught", "block_bonafide_flagged", "block_spoof_caught")
    return {"splits": n, **{name: {"mean": round(float(rows[:, j].mean()), 4), "std": round(float(rows[:, j].std()), 4),
                                   "min": round(float(rows[:, j].min()), 4), "max": round(float(rows[:, j].max()), 4)}
                            for j, name in enumerate(names)}}


def make_wavlm_bundle(src_dir: str | Path, dst_dir: str | Path, risk: dict, provenance: dict) -> Path:
    """A model bundle with only the WavLM branch of `src_dir` (fusion weight 1) and the given risk thresholds.
    The source bundle is not touched."""
    src, dst = Path(src_dir), Path(dst_dir)
    manifest = json.loads((src / MANIFEST).read_text())
    if "wavlm" not in manifest.get("branches", []):
        raise ValueError(f"{src} has no WavLM branch (branches: {manifest.get('branches')})")
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src / WAVLM_FILE, dst / WAVLM_FILE)
    keep = {k: v for k, v in manifest.items() if k.startswith("wavlm_") or k in (
        "version", "segment_seconds", "segment_hop_seconds")}
    out = {**keep, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "branches": ["wavlm"],
           "feature_versions": {"wavlm": manifest["feature_versions"]["wavlm"]}, "fusion_weights": {"wavlm": 1.0},
           "risk": {"high": risk["high"], "medium": risk["medium"]},
           "metrics": {"source_bundle": str(src), "source_metrics": manifest.get("metrics", {}),
                       "threshold_calibration": provenance}}
    (dst / MANIFEST).write_text(json.dumps(out, indent=2))
    return dst
