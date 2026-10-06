"""Per-channel breakdown of `audiodf evaluate --save-scores` output on ASVspoof 2021 LA (real telephony channels).

Joins each scores CSV to the protocol file (transmission route, phase, trim) and reports, for each group of clips:
  eer       EER inside the group (what the model separates when the threshold is picked for that channel)
  flagged   genuine clips scored at or above the *global* EER threshold (the single threshold a deployed system has)
  caught    fakes scored at or above that same threshold
Large gaps between `eer` and flagged/caught mean the score shifts with the channel, which a single deployed threshold
cannot absorb. Several runs are shown side by side. Test-data analysis only: nothing here feeds back into training.

  python -m audiodf.evaluation.channel_report --protocol dataset21/ASVspoof2021.LA.eval.tsv \
      --runs run5=results/evaluate_asv21_eval_run5_scores.csv run4=results/evaluate_asv21_eval_run4_wavlm_scores.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from audiodf.data import asv21, voip_sim
from audiodf.evaluation.metrics import compute_metrics

# protocol layouts this tool reads (kind -> (columns, default groups))
KINDS = {"asv21": (asv21.COLUMNS, ("codec", "transmission", "phase", "trim")),
         "calls": (voip_sim.CALL_COLUMNS, ("codec", "noise", "snr_bin", "loss"))}


def load_protocol(path, columns=asv21.COLUMNS) -> dict:
    """Protocol rows as column arrays, in file order (= clip_index order of the evaluation)."""
    with open(path) as f:
        rows = [line.split() for line in f if line.strip()]
    bad = [r for r in rows if len(r) != len(columns)]
    if bad:
        raise ValueError(f"{path}: expected {len(columns)} columns, got {len(bad[0])} in {bad[0]!r}")
    cols = {c: np.array([r[i] for r in rows]) for i, c in enumerate(columns)}
    if "codec" in cols and "transmission" in cols:
        cols["codec/transmission"] = np.char.add(np.char.add(cols["codec"], "/"), cols["transmission"])
    cols["label"] = (cols["key"] == "spoof").astype(int)
    return cols


def load_scores(path, protocol: dict, column: str) -> tuple[np.ndarray, np.ndarray]:
    """(row indices into the protocol, scores in `column`); checks that every clip's label matches the protocol."""
    idx, label, score = [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            idx.append(int(row["clip_index"]))
            label.append(int(row["label"]))
            score.append(float(row[column]))
    idx = np.array(idx)
    if not np.array_equal(protocol["label"][idx], np.array(label)):
        raise ValueError(f"{path}: clip labels do not match the protocol; was it scored on another protocol file?")
    return idx, np.array(score)


def breakdown(protocol: dict, idx: np.ndarray, score: np.ndarray, groups, min_class: int = 20) -> dict:
    y = protocol["label"][idx]
    overall = compute_metrics(y, score)
    thr = overall["eer_threshold"]
    out = {"overall": {"n": len(y), "eer": overall["eer_pct"], "auc": overall["auc"], "global_threshold": thr}}
    for g in groups:
        vals = protocol["attack"][idx] if g == "attack" else protocol[g][idx]
        res = {}
        for v in sorted(set(vals)):
            m = vals == v
            if g == "attack" and v == "bonafide":
                continue
            if g == "attack":  # one attack against all genuine clips, as in the per-attack numbers elsewhere
                m = m | (y == 0)
            yy, ss = y[m], score[m]
            if min((yy == 0).sum(), (yy == 1).sum()) < min_class:
                continue
            res[str(v)] = {"n": int(m.sum()), "eer": compute_metrics(yy, ss)["eer_pct"],
                           "flagged": round(float((ss[yy == 0] >= thr).mean()), 4),
                           "caught": round(float((ss[yy == 1] >= thr).mean()), 4)}
        out[g] = res
    return out


def compare(runs: dict, protocol_path, column: str = "fused_10s", kind: str = "asv21", groups=None) -> dict:
    columns, default_groups = KINDS[kind]
    protocol = load_protocol(protocol_path, columns)
    return {name: breakdown(protocol, *load_scores(path, protocol, column), groups or default_groups)
            for name, path in runs.items()}


def format_table(report: dict, group: str) -> str:
    names = list(report)
    keys = sorted(set().union(*(report[n][group] for n in names)))
    lines = [f"| {group} | " + " | ".join(f"{n} EER / flagged / caught" for n in names) + " |",
             "|---|" + "---|" * len(names)]
    for k in keys:
        cells = []
        for n in names:
            r = report[n][group].get(k)
            cells.append(f"{r['eer']:.2f}% / {100 * r['flagged']:.1f}% / {100 * r['caught']:.1f}% (n={r['n']})"
                         if r else "-")
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", required=True,
                    help="ASVspoof2021.LA.eval.tsv (`audiodf prepare21`) or ASV5.eval.calls.tsv (`audiodf render-calls`)")
    ap.add_argument("--kind", choices=sorted(KINDS), default="asv21", help="which protocol layout (default asv21)")
    ap.add_argument("--runs", nargs="+", required=True, metavar="NAME=SCORES.csv")
    ap.add_argument("--column", default="fused_10s", help="score column, e.g. fused_10s, wavlm_4s (default fused_10s)")
    ap.add_argument("--out", help="write the full breakdown as JSON")
    ap.add_argument("--groups", nargs="+", help="protocol columns to break down by (default depends on --kind); "
                    "'attack' is always available")
    args = ap.parse_args(argv)
    runs = dict(r.split("=", 1) for r in args.runs)
    report = compare(runs, args.protocol, args.column, args.kind, args.groups)
    groups = args.groups or KINDS[args.kind][1]
    for n, r in report.items():
        o = r["overall"]
        print(f"{n}: n={o['n']} EER {o['eer']:.2f}% AUC {o['auc']:.4f} (global threshold {o['global_threshold']})")
    for g in groups:
        print("\n" + format_table(report, g))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\nreport: {args.out}")


if __name__ == "__main__":
    main()
