"""Weighted fusion of per-branch P(spoof). Branches: "svm", "rcnn", "wavlm" (any subset)."""

from __future__ import annotations

import itertools

import numpy as np

BRANCHES = ("svm", "rcnn", "wavlm")


def fuse(probs: dict, weights: dict):
    """sum_b weights[b] * probs[b]; weights must be non-negative and sum to 1."""
    if any(w < 0 for w in weights.values()) or abs(sum(weights.values()) - 1.0) > 1e-6:
        raise ValueError(f"fusion weights must be non-negative and sum to 1, got {weights}")
    missing = [b for b, w in weights.items() if w > 0 and b not in probs]
    if missing:
        raise ValueError(f"no scores for weighted branches {missing}")
    return sum(w * np.asarray(probs[b], dtype=float) for b, w in weights.items() if w > 0)


def weights_from_manifest(manifest: dict) -> dict:
    """Fusion weights stored in a model bundle (bundles from runs 1-3 stored only the SVM weight)."""
    if "fusion_weights" in manifest:
        return dict(manifest["fusion_weights"])
    return {"svm": manifest["svm_weight"], "rcnn": round(1.0 - manifest["svm_weight"], 6)}


def simplex_grid(branches, step: float = 0.05) -> list[dict]:
    """Every weighting of `branches` on a `step` grid that sums to 1."""
    n = round(1 / step)
    out = []
    for parts in itertools.product(range(n + 1), repeat=len(branches) - 1):
        if sum(parts) <= n:
            ws = [p / n for p in parts] + [(n - sum(parts)) / n]
            out.append({b: round(w, 6) for b, w in zip(branches, ws)})
    return out
