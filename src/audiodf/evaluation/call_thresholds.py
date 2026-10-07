"""Verify / block thresholds for call audio: set on one half of the ASV5 eval speakers, reported on the other.

The speakers are split once by a hash of their name (`voip_sim.speaker_half`), so the split is the same in every call set.
Thresholds come from the genuine calls of half A in the calibration sets (pooled); every report uses clips of half B only
(call sets) or other speakers entirely (ASVspoof 2021), so nothing reported was used to set anything. For each genuine-flag
budget the report gives the threshold and, on every test set, the share of genuine calls flagged and fakes caught; the
operating points themselves are a product decision.

  python -m audiodf call-thresholds --cal ../dataset_calls_cal_v1/ASV5.eval.calls.tsv,../results/x_scores.csv ... \
      --test v1=../dataset_calls/ASV5.eval.calls.tsv,../results/y_scores.csv ... --stored ../artifacts --tag run7
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from audiodf.calibrate import policy_thresholds, threshold_rates
from audiodf.data import asv21, voip_sim
from audiodf.evaluation.channel_report import load_protocol, load_scores
from audiodf.evaluation.metrics import compute_metrics

BUDGETS = (0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20)  # share of genuine calls the threshold may flag
GROUPS = {"calls": ("loss", "noise", "codec"), "asv21": ("transmission", "codec")}


@dataclass
class ScoredSet:
    name: str
    kind: str  # "calls" or "asv21"
    protocol: dict
    rows: np.ndarray  # protocol row of each score
    score: np.ndarray

    @property
    def y(self) -> np.ndarray:
        return self.protocol["label"][self.rows]

    def col(self, name: str) -> np.ndarray:
        return self.protocol[name][self.rows]


def load_set(name: str, protocol_path: str, scores_path: str, column: str = "wavlm_10s") -> ScoredSet:
    with open(protocol_path) as f:
        width = len(f.readline().split())
    kind = "calls" if width == len(voip_sim.CALL_COLUMNS) else "asv21"
    protocol = load_protocol(protocol_path, voip_sim.CALL_COLUMNS if kind == "calls" else asv21.COLUMNS)
    rows, score = load_scores(scores_path, protocol, column)
    return ScoredSet(name, kind, protocol, rows, score)


def _rates(s: np.ndarray, y: np.ndarray, thr: float) -> dict:
    return {"genuine_flagged": round(float((s[y == 0] >= thr).mean()), 4),
            "fakes_caught": round(float((s[y == 1] >= thr).mean()), 4)}


def call_thresholds(cal: list[ScoredSet], tests: list[ScoredSet], stored: dict | None = None,
                    budgets=BUDGETS, block_fpr: float = 0.01, verify_fpr: float = 0.10) -> dict:
    """Thresholds from half A of the `cal` sets (pooled), reported on half B of call test sets and on all of an ASVspoof
    2021 test set. `stored`: the bundle's current {"high", "medium"} thresholds, reported alongside."""
    s_parts, y_parts, ignored = [], [], 0
    for c in cal:
        a = voip_sim.speaker_half(c.col("speaker")) if c.kind == "calls" else np.ones(len(c.score), dtype=bool)
        ignored += int((~a).sum())
        s_parts.append(c.score[a])
        y_parts.append(c.y[a])
    cal_s, cal_y = np.concatenate(s_parts), np.concatenate(y_parts)
    genuine = cal_s[cal_y == 0]
    curve_thr = {b: float(np.quantile(genuine, 1 - b)) for b in budgets}
    proposed = policy_thresholds(cal_s, cal_y, block_fpr, verify_fpr)
    out = {"calibration": {"sets": [c.name for c in cal], "genuine": int((cal_y == 0).sum()), "fakes": int((cal_y == 1).sum()),
                           "ignored_half_b_clips": ignored,
                           "curve": [{"budget": b, "threshold": round(t, 6), **_rates(cal_s, cal_y, t)}
                                     for b, t in curve_thr.items()]},
           "proposed": {"block_fpr": block_fpr, "verify_fpr": verify_fpr, **proposed},
           "stored": stored, "tests": {}}
    for t in tests:
        keep = ~voip_sim.speaker_half(t.col("speaker")) if t.kind == "calls" else np.ones(len(t.score), dtype=bool)
        s, y = t.score[keep], t.y[keep]
        rep = {"kind": t.kind, "speakers": "half B" if t.kind == "calls" else "all (none in calibration)",
               "genuine": int((y == 0).sum()), "fakes": int((y == 1).sum()), "eer_pct": compute_metrics(y, s)["eer_pct"],
               "curve": [{"budget": b, "threshold": round(thr, 6), **_rates(s, y, thr)} for b, thr in curve_thr.items()]}
        for label, risk in (("proposed", proposed), ("stored", stored)):
            if risk is None:
                continue
            rep[label] = {"overall": threshold_rates(s, y, None, risk)["overall"]}
            for g in GROUPS[t.kind]:
                rep[label][g] = threshold_rates(s, y, t.col(g)[keep], risk).get("per_group", {})
        out["tests"][t.name] = rep
    return out


def summary_lines(rep: dict) -> list[str]:
    """A compact text table: per budget, genuine flagged / fakes caught on every test set."""
    names = list(rep["tests"])
    lines = [f"calibration: {rep['calibration']['genuine']} genuine + {rep['calibration']['fakes']} fake calls "
             f"(half A of {', '.join(rep['calibration']['sets'])})",
             "budget  threshold  | " + " | ".join(f"{n} flagged / caught" for n in names)]
    for k, point in enumerate(rep["calibration"]["curve"]):
        cells = [f"{rep['tests'][n]['curve'][k]['genuine_flagged']:6.1%} / {rep['tests'][n]['curve'][k]['fakes_caught']:6.1%}"
                 for n in names]
        lines.append(f"{point['budget']:6.1%}  {point['threshold']:.6f} | " + " | ".join(cells))
    if rep["stored"]:
        for key, name in (("medium", "verify"), ("high", "block")):
            cells = [f"{rep['tests'][n]['stored']['overall'][name]['bonafide_flagged']:6.1%} / "
                     f"{rep['tests'][n]['stored']['overall'][name]['spoof_caught']:6.1%}" for n in names]
            lines.append(f"stored {name:6s} {rep['stored'][key]:.6f} | " + " | ".join(cells))
    return lines
