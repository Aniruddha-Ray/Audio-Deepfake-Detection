"""Serving cost of the final model: latency and throughput per device and precision, calls per server, and whether faster variants
change any decision.

A live call needs one 2 s window scored per second (1 s hop), so the number of calls one device can serve is about its window throughput
(windows / s). Variants: GPU float32, GPU float16 (autocast, as served), CPU float32, CPU int8 (dynamic quantisation of the Linear layers).
Decision check: the clip scores of each variant against float32 on real calls, and how often the verify / block action differs.

  python -m audiodf.evaluation.serving_cost --clips 300 --out ../results/serving_cost.json
"""

from __future__ import annotations

import argparse
import copy
import json
import time

import numpy as np
import torch

from audiodf.config import load_settings


def _time(fn, x, repeat: int) -> float:
    fn(x)
    if x.is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn(x)
    if x.is_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeat


def variants(model) -> dict:
    """name -> (forward function, device)."""
    out = {}
    if torch.cuda.is_available():
        gpu = copy.deepcopy(model).cuda().eval()

        def fp32(x, m=gpu):
            with torch.no_grad():
                return torch.sigmoid(m(x).float())

        def fp16(x, m=gpu):
            with torch.no_grad(), torch.autocast("cuda"):
                return torch.sigmoid(m(x).float())

        out["GPU float32"], out["GPU float16 (served)"] = (fp32, "cuda"), (fp16, "cuda")
    cpu = copy.deepcopy(model).cpu().eval()

    def cpu32(x, m=cpu):
        with torch.no_grad():
            return torch.sigmoid(m(x).float())

    q = torch.ao.quantization.quantize_dynamic(copy.deepcopy(model).cpu().eval(), {torch.nn.Linear}, dtype=torch.qint8)

    def cpu8(x, m=q):
        with torch.no_grad():
            return torch.sigmoid(m(x).float())

    out["CPU float32"], out["CPU int8 (dynamic)"] = (cpu32, "cpu"), (cpu8, "cpu")
    return out


def call_windows(settings, n_clips: int, seed: int = 0):
    """Windows of real calls (call set v1, the first 10 s of speech of n_clips random clips) and the clip each belongs to."""
    from audiodf.data.prepare import WaveWindowDataset, build_index, stream_windows

    idx = build_index(settings, "calls", "eval", None, 0, lambda *_: None, available_only=True)
    utts = np.sort(np.random.default_rng(seed).choice(np.nonzero(idx.has_speech)[0], n_clips, replace=False))
    keys = stream_windows(idx, utts, settings)
    ds = WaveWindowDataset(idx, settings, 0.0, 0)
    return np.stack([ds.read_window(i, s) for i, s in keys]), np.array([i for i, _ in keys]), idx.label[utts], utts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", type=int, default=300, help="real calls for the decision check")
    ap.add_argument("--out", default="../results/serving_cost.json")
    args = ap.parse_args()
    s = load_settings()
    from audiodf.artifacts import load_bundle

    bundle = load_bundle(s, "cpu")
    model = bundle.models["wavlm"]
    verify, block = bundle.manifest["risk"]["medium"], bundle.manifest["risk"]["high"]
    win, owner, labels, utts = call_windows(s, args.clips)
    rep = {"cpu_threads": torch.get_num_threads(), "verify": verify, "block": block, "variants": {}}
    ref = None
    for name, (fn, dev) in variants(model).items():
        timing = {}
        for b in ((1, 8, 32) if dev == "cuda" else (1, 8)):
            x = torch.from_numpy(win[:b]).to(dev)
            sec = _time(fn, x, 20 if dev == "cuda" else 3)
            timing[f"batch {b}"] = {"ms_per_batch": round(1000 * sec, 1), "windows_per_s": round(b / sec, 1)}
        scores = []
        bs = 32 if dev == "cuda" else 8
        for i in range(0, len(win), bs):
            scores.append(fn(torch.from_numpy(win[i:i + bs]).to(dev)).cpu().numpy().reshape(-1))
        clip = np.bincount(np.searchsorted(utts, owner), weights=np.concatenate(scores)) / np.bincount(np.searchsorted(utts, owner))
        if ref is None:
            ref = clip
        act = lambda p: np.where(p >= block, 2, np.where(p >= verify, 1, 0))  # noqa: E731
        best = max(timing.values(), key=lambda t: t["windows_per_s"])["windows_per_s"]
        rep["variants"][name] = {"timing": timing, "calls_per_device": int(best), "max_abs_score_diff_vs_first": float(np.abs(clip - ref).max()),
                                 "actions_changed_vs_first": int((act(clip) != act(ref)).sum()), "clips": int(len(clip))}
        print(f"{name:22s} {timing} -> ~{int(best)} live calls per device | actions changed vs {list(rep['variants'])[0]}: "
              f"{rep['variants'][name]['actions_changed_vs_first']} of {len(clip)} (max score diff {rep['variants'][name]['max_abs_score_diff_vs_first']:.4f})")
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=1)
    print(f"report: {args.out}")


if __name__ == "__main__":
    main()
