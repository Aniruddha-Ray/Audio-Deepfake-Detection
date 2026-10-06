"""Paired comparison of two runs scored on the same clips (`audiodf evaluate --save-scores` CSVs).

EER of each run and a paired bootstrap of their difference: genuine and fake clips are resampled with replacement (the
same clips for both runs), so the interval reflects test-sample noise on this clip set. It does not cover run-to-run
(training seed) noise, which is not measured.

  python -m audiodf.evaluation.compare_runs --first run4=a_scores.csv --second run5=b_scores.csv --column wavlm_10s
"""

from __future__ import annotations

import argparse
import csv
import json

import numpy as np
from sklearn.metrics import roc_curve


def eer_pct(y: np.ndarray, s: np.ndarray) -> float:
    """EER in percent, defined exactly as in evaluation/metrics.compute_metrics."""
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float(100 * (fpr[i] + fnr[i]) / 2)


def load(path, column: str) -> dict:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {"clip": np.array([int(r["clip_index"]) for r in rows]), "y": np.array([int(r["label"]) for r in rows]),
            "s": np.array([float(r[column]) for r in rows]), "group": np.array([r["codec"] for r in rows])}


def paired_bootstrap(y: np.ndarray, first: np.ndarray, second: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict:
    """EER(second) - EER(first) with a 95% interval; positive means `first` is better (lower EER)."""
    rng = np.random.default_rng(seed)
    bona, spoof = np.nonzero(y == 0)[0], np.nonzero(y == 1)[0]
    diffs = np.empty(n_boot)
    for k in range(n_boot):
        i = np.concatenate([rng.choice(bona, len(bona)), rng.choice(spoof, len(spoof))])
        diffs[k] = eer_pct(y[i], second[i]) - eer_pct(y[i], first[i])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"eer_first": round(eer_pct(y, first), 3), "eer_second": round(eer_pct(y, second), 3),
            "diff_second_minus_first": round(eer_pct(y, second) - eer_pct(y, first), 3),
            "ci95": [round(float(lo), 3), round(float(hi), 3)], "share_first_better": round(float((diffs > 0).mean()), 4),
            "n_boot": n_boot}


def compare(first_csv, second_csv, column: str, n_boot: int = 1000, seed: int = 0) -> dict:
    a, b = load(first_csv, column), load(second_csv, column)
    if not np.array_equal(a["clip"], b["clip"]) or not np.array_equal(a["y"], b["y"]):
        raise ValueError("the two score files are not for the same clips; a paired comparison needs identical clips")
    out = {"n": len(a["y"]), **paired_bootstrap(a["y"], a["s"], b["s"], n_boot, seed), "per_group": {}}
    for g in sorted(set(a["group"])):
        m = a["group"] == g
        if min((a["y"][m] == 0).sum(), (a["y"][m] == 1).sum()) >= 20:
            out["per_group"][g] = {"n": int(m.sum()), "eer_first": round(eer_pct(a["y"][m], a["s"][m]), 3),
                                   "eer_second": round(eer_pct(b["y"][m], b["s"][m]), 3)}
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--first", required=True, metavar="NAME=SCORES.csv")
    ap.add_argument("--second", required=True, metavar="NAME=SCORES.csv")
    ap.add_argument("--column", default="fused_10s")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    (n1, p1), (n2, p2) = args.first.split("=", 1), args.second.split("=", 1)
    rep = {"first": n1, "second": n2, "column": args.column, **compare(p1, p2, args.column, args.n_boot)}
    text = json.dumps(rep, indent=2)
    print(text)
    if args.out:
        open(args.out, "w").write(text)


if __name__ == "__main__":
    main()
