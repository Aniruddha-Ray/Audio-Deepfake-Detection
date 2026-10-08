"""Reproduces the intent evaluation of audit.md phase 30 (rules) and measures the LLM path the same way when a provider and key are given.

  python -m audiodf.intent.evaluate                      # rules only
  python -m audiodf.intent.evaluate --provider groq      # rules + bounded LLM (needs AUDIODF_LLM_API_KEY; rate limits apply)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SETS = {"BothBosu test": ROOT / "dataset_intent" / "bothbosu_scam-dialogue_test.csv",
        "shakeleoatmeal test (independent)": ROOT / "dataset_intent" / "shakeleoatmeal_test.parquet"}


def _load(path: Path):
    import pandas as pd

    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def evaluate(provider: str = "template", pause_s: float = 0.0) -> dict:
    from sklearn.metrics import roc_auc_score

    from audiodf.intent.call import text_intent

    out = {}
    for name, path in SETS.items():
        d = _load(path)
        y = d["label"].astype(int).values
        res = []
        for dialogue in d["dialogue"]:
            res.append(text_intent(dialogue, provider))
            if pause_s:
                time.sleep(pause_s)  # free tiers limit requests per minute
        s = np.array([r["score"] for r in res])
        lv = np.array([r["level"] for r in res])
        out[name] = {"n": int(len(y)), "auc": float(roc_auc_score(y, s)),
                     "caution_or_high": {"scam": float(np.mean(lv[y == 1] != "none")), "normal": float(np.mean(lv[y == 0] != "none"))},
                     "high": {"scam": float(np.mean(lv[y == 1] == "high")), "normal": float(np.mean(lv[y == 0] == "high"))},
                     "llm_failed": int(sum(1 for r in res if isinstance(r.get("llm"), dict) and r["llm"].get("used") is False))}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", default="template", choices=["template", "groq", "gemini", "openrouter"])
    ap.add_argument("--pause", type=float, default=0.0, help="seconds between LLM requests (free-tier rate limits)")
    ap.add_argument("--out", default=str(ROOT / "results" / "intent_eval.json"))
    args = ap.parse_args()
    rep = evaluate(args.provider, args.pause)
    for name, r in rep.items():
        print(f"{name}: AUC {r['auc']:.3f} | caution or high: scam {r['caution_or_high']['scam']:.1%}, normal {r['caution_or_high']['normal']:.1%} | "
              f"high: scam {r['high']['scam']:.1%}, normal {r['high']['normal']:.1%}" + (f" | LLM failed on {r['llm_failed']}" if r["llm_failed"] else ""))
    Path(args.out).write_text(json.dumps({"provider": args.provider, "sets": rep}, indent=1))


if __name__ == "__main__":
    main()
