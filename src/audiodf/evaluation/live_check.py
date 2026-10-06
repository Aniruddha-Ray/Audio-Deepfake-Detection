"""Does the live path agree with the batched scoring? Stream clips through CallSession in 0.5 s chunks.

Evaluations score clips in batches (`stream_windows` + `score_windows`); a deployed system runs `CallSession` on audio
chunks as they arrive (VAD gate, rolling buffer, 2 s windows every 1 s, one verdict per completed window). This streams
a sample of clips through the real session, takes the first verdict that covers 10 s of speech (the same horizon as the
batched EER), and compares it with the batched score of the same clip. It also times the verdicts.
"""

from __future__ import annotations

import csv
import time

import numpy as np

from audiodf.data.audio import load_audio
from audiodf.data.prepare import build_index
from audiodf.evaluation.compare_runs import eer_pct
from audiodf.inference.engine import DetectionEngine
from audiodf.inference.session import CallSession


def live_check(settings, dataset: str, split: str, scores_csv, column: str = "fused_10s", n: int = 400,
               seed: int = 0, log=print) -> dict:
    """settings.paths.artifacts_dir is the bundle to serve; scores_csv must come from scoring the same bundle."""
    idx = build_index(settings, dataset, split, available_only=True, log=log)
    with open(scores_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    rng = np.random.default_rng(seed)
    pick = []
    for label in ("0", "1"):  # half genuine, half fake
        pool = [r for r in rows if r["label"] == label]
        pick += [pool[i] for i in rng.choice(len(pool), min(n // 2, len(pool)), replace=False)]
    engine = DetectionEngine.from_artifacts(settings)
    sr, chunk = settings.audio.sample_rate, int(settings.stream.chunk_seconds * settings.audio.sample_rate)
    live, batched, label, latency, n_verdicts = [], [], [], [], []
    t0 = time.time()
    for r in pick:
        i = int(r["clip_index"])
        wave = load_audio(idx.path(i), sr)
        session, first_final, count = CallSession(engine, str(idx.utt_id[i])), None, 0
        for s in range(0, len(wave), chunk):
            v = session.push(wave[s:s + chunk])
            if v is not None:
                count += 1
                latency.append(v.latency_ms)
                if v.final and first_final is None:
                    first_final = v  # first verdict covering a full 10 s window
        v = first_final or session.finalize()
        if v is None:  # no speech at all
            continue
        live.append(v.fake_probability)
        batched.append(float(r[column]))
        label.append(int(r["label"]))
        n_verdicts.append(count)
    live, batched, label = np.array(live), np.array(batched), np.array(label)
    diff = np.abs(live - batched)
    out = {"clips": len(live), "seconds": round(time.time() - t0, 1),
           "eer_live_pct": round(eer_pct(label, live), 3), "eer_batched_pct": round(eer_pct(label, batched), 3),
           "abs_diff": {"median": round(float(np.median(diff)), 5), "p95": round(float(np.percentile(diff, 95)), 5),
                        "max": round(float(diff.max()), 5)},
           "share_within_0.01": round(float((diff <= 0.01).mean()), 4),
           "same_side_of_0.5": round(float(((live >= 0.5) == (batched >= 0.5)).mean()), 4),
           "verdict_latency_ms": {"p50": round(float(np.percentile(latency, 50)), 1),
                                  "p95": round(float(np.percentile(latency, 95)), 1),
                                  "max": round(float(np.max(latency)), 1)},
           "verdicts_per_call_median": float(np.median(n_verdicts))}
    log(f"  live path vs batched on {out['clips']} clips: EER {out['eer_live_pct']}% vs {out['eer_batched_pct']}%, "
        f"|diff| median {out['abs_diff']['median']}, p95 {out['abs_diff']['p95']}; verdict latency p50 "
        f"{out['verdict_latency_ms']['p50']} ms")
    return out
