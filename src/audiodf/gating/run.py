"""Driver of the channel-quality-gated fusion experiment (new_plan.md 7.3v). Commands (python -m audiodf.gating.run <command>):

  tuning    score the held-out tuning set with both models of a bundle, compute its channel features and known impairments
  fit       fit the channel-quality estimator on training copies (known echo / noise from the impairment plan)
  features  channel features of the clips of a scores CSV (a test set): the same clips, the same 10 s of speech
  eval      choose the gate on the tuning set, apply it to every test set, check the rule

Nothing here is tuned on a test set: the estimator is fitted on training clips, the gate parameters on the tuning set.
"""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

from audiodf.config import load_settings
from audiodf.data.impairments import build_kit, impair_plan
from audiodf.gating.channel_quality import NOISY_SNR_DB, SECONDS, SR, ChannelQualityModel, channel_features, leak_auc

TEST_SETS = {  # name -> (scores CSV of the run-9 bundle, features file)
    "asv5": "evaluate_asv5_eval_run9_scores.csv", "asv19": "evaluate_asv19_eval_run9_scores.csv",
    "asv21": "evaluate_asv21_eval_run9_scores.csv", "v1": "evaluate_calls_eval_run9_calls_scores.csv",
    "babble": "evaluate_calls_eval_run9_babble_scores.csv", "v2": "evaluate_calls_eval_run9_v2_scores.csv"}


def read_window(idx, i: int, seconds: float = SECONDS) -> np.ndarray:
    """The first `seconds` of speech of clip i (its rendered copy if it has one), as the models see it."""
    with sf.SoundFile(idx.path(i)) as f:
        f.seek(min(int(idx.speech_start[i]), max(f.frames - 1, 0)))
        x = f.read(int(seconds * SR), dtype="float32", always_2d=True)
    return x.mean(axis=1)


def extract_features(idx, utts, workers: int = 8, log=print) -> np.ndarray:
    log(f"  channel features of {len(utts)} clips")
    with ThreadPoolExecutor(workers) as pool:
        return np.stack(list(pool.map(lambda i: channel_features(read_window(idx, int(i))), utts)))


def plan_labels(idx, utts, cfg, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(echo, snr_db) of each clip from the impairment plan; a clip without a rendered copy has neither. Label-blind."""
    echo, snr = np.zeros(len(utts), dtype=bool), np.full(len(utts), np.nan)
    for k, i in enumerate(utts):
        if idx.rendered[i]:
            plan = impair_plan(str(idx.utt_id[i]), seed, cfg)
            echo[k] = plan.reverb
            snr[k] = plan.snr_db if plan.noise != "none" else np.nan
    return echo, snr


def training_copies(settings, workers: int, log=print):
    """The training pool and the tuning set with their rendered copies, exactly as run_training builds them (cached on disk)."""
    from audiodf.data.ffmpeg_codecs import render_copies
    from audiodf.training.pipeline import _subset, build_training_pool, training_splits

    train, dev, _ = build_training_pool(settings, training_splits(settings), None, workers, log)
    tune = _subset(dev, min(settings.data.tune_utts, len(dev)))
    kit = build_kit(settings, train)
    train = render_copies(train, np.nonzero(train.has_speech)[0], settings, settings.data.render_frac, seed=1, log=log, impair=kit)
    dev = render_copies(dev, tune, settings, settings.data.tune_render_frac, seed=2, log=log, impair=kit)
    return train, dev, tune, kit


def _settings(args):
    settings = load_settings(args.config)
    if getattr(args, "bundle", None):
        settings.paths.artifacts_dir = args.bundle
    return settings


def cmd_tuning(args):
    from audiodf.artifacts import load_bundle
    from audiodf.evaluation.stream_eval import score_split
    from audiodf.training.pipeline import pick_device

    settings = _settings(args)
    device = pick_device()
    bundle = load_bundle(settings, device)
    models = {b: bundle.models[b] for b in ("wavlm", "whisper")}
    train, dev, tune, kit = training_copies(settings, args.workers)
    scores = score_split(models, dev, tune, settings, device, args.workers, f"tune{len(tune)}", print, 0.0)
    feats = extract_features(dev, tune, args.workers)
    echo, snr = plan_labels(dev, tune, kit.cfg, seed=2)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, utts=tune, label=scores.label, attack=scores.attack, grid=np.array(scores.grid),
             wavlm=scores.branches["wavlm"], whisper=scores.branches["whisper"], feats=feats, echo=echo, snr=snr)
    print(f"tuning set: {len(tune)} clips -> {args.out}")


def cmd_fit(args):
    settings = _settings(args)
    train, dev, tune, kit = training_copies(settings, args.workers)
    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(np.nonzero(train.has_speech)[0], args.n, replace=False))
    x = extract_features(train, pick, args.workers)
    echo, snr = plan_labels(train, pick, kit.cfg, seed=1)
    print(f"training clips for the estimator: {len(pick)} ({echo.mean():.0%} echo, {(np.nan_to_num(snr, nan=99) < NOISY_SNR_DB).mean():.0%} "
          f"noisy), {int(train.rendered[pick].sum())} with a rendered copy")
    model = ChannelQualityModel().fit(x, echo, snr)
    model.save(args.out)
    t = np.load(args.tuning)
    pe, pn, q = model.predict(t["feats"])
    noisy = np.nan_to_num(t["snr"], nan=99) < NOISY_SNR_DB
    from sklearn.metrics import roc_auc_score
    rep = {"echo_accuracy": float(((pe > .5) == t["echo"]).mean()), "echo_auc": float(roc_auc_score(t["echo"], pe)),
           "noisy_accuracy": float(((pn > .5) == noisy).mean()), "noisy_auc": float(roc_auc_score(noisy, pn)),
           "tuning_clips": int(len(q)), "mean_q_clean": float(q[~t["echo"] & ~noisy].mean()),
           "mean_q_degraded": float(q[t["echo"] | noisy].mean()), "leak_auc_tuning": leak_auc(q, t["label"])}
    print(json.dumps(rep, indent=1))
    Path(args.out).with_suffix(".json").write_text(json.dumps(rep, indent=1))


def cmd_features(args):
    from audiodf.data.prepare import build_index

    settings = _settings(args)
    with open(args.scores, newline="") as f:
        rows = list(csv.DictReader(f))
    utts = np.array([int(r["clip_index"]) for r in rows])
    idx = build_index(settings, args.dataset, args.split, None, args.workers, print, available_only=True)
    x = extract_features(idx, utts, args.workers)
    np.savez(args.out, clip_index=utts, label=np.array([int(r["label"]) for r in rows]), feats=x)
    print(f"{len(utts)} clips -> {args.out}")


def _eer(y, s):
    from audiodf.evaluation.metrics import compute_metrics

    return compute_metrics(y, s)["eer_pct"]


def cmd_eval(args):
    from audiodf.evaluation.channel_report import KINDS, load_protocol
    from audiodf.evaluation.compare_runs import load, paired_bootstrap
    from audiodf.gating.fusion import gate_weight, gated_fuse, tune_gate

    res = Path(args.results)
    model = ChannelQualityModel.load(args.estimator)
    t = np.load(args.tuning)
    _, _, q_tune = model.predict(t["feats"])
    (w_min, w_max), table = tune_gate(t["wavlm"], t["whisper"], q_tune, t["label"])
    print(f"gate chosen on the tuning set: w_min={w_min:g} w_max={w_max:g}\n  mean EER by setting: {table}")
    out = {"gate": {"w_min": w_min, "w_max": w_max, "tuning_table": table}, "sets": {}}
    data = {}
    for name, csvname in TEST_SETS.items():
        W, H, F = (load(str(res / csvname), c) for c in ("wavlm_10s", "whisper_10s", "fused_10s"))
        z = np.load(res / "gating" / f"features_{name}.npz")
        assert np.array_equal(z["clip_index"], W["clip"]), f"{name}: features are not of the clips of the scores file"
        pe, pn, q = model.predict(z["feats"])
        y = W["y"]
        g = gated_fuse(W["s"], H["s"], q, w_min, w_max)
        w = gate_weight(q, w_min, w_max)
        bs = paired_bootstrap(y, W["s"], g)
        data[name] = dict(y=y, W=W["s"], H=H["s"], F=F["s"], G=g, q=q, pe=pe, pn=pn, clip=W["clip"])
        out["sets"][name] = {"n": int(len(y)), "eer_wavlm": _eer(y, W["s"]), "eer_whisper": _eer(y, H["s"]), "eer_fixed_65_35": _eer(y, F["s"]),
                             "eer_gated": _eer(y, g), "gated_minus_wavlm": bs["diff_second_minus_first"], "ci95": bs["ci95"],
                             "mean_whisper_weight": float(w.mean()), "share_whisper_skipped": float((w < 0.1).mean()),
                             "leak_auc_q": leak_auc(q, y), "mean_q": float(q.mean())}
    a = data["asv21"]
    for key, b in (("10pct", 0.10), ("1pct", 0.01)):
        out["banking_asv21_fakes_caught_at_" + key + "_of_genuine"] = {
            nm: float((a[c][a["y"] == 1] >= np.quantile(a[c][a["y"] == 0], 1 - b)).mean()) for nm, c in
            (("wavlm", "W"), ("whisper", "H"), ("fixed_65_35", "F"), ("gated", "G"))}
    prot = load_protocol(args.v2_protocol, KINDS["calls"][0])
    d = data["v2"]
    noise = np.asarray(prot["noise"])[d["clip"]]
    snr = np.array([float(s) if s not in ("-", "none") else np.nan for s in np.asarray(prot["snr_db"])[d["clip"]]])
    echo_true = np.array(["+rir" in n for n in noise])
    noisy_true = np.nan_to_num(snr, nan=99) < NOISY_SNR_DB
    from sklearn.metrics import roc_auc_score
    out["q_on_v2_known_conditions"] = {"echo_auc": float(roc_auc_score(echo_true, d["pe"])), "noisy_auc": float(roc_auc_score(noisy_true, d["pn"])),
                                       "echo_accuracy": float(((d["pe"] > .5) == echo_true).mean()),
                                       "noisy_accuracy": float(((d["pn"] > .5) == noisy_true).mean())}
    kind = np.array([n.replace("+rir", "") for n in noise])
    conds = {"v2 overall": np.ones(len(noise), bool), "no noise, no echo": (kind == "none") & ~echo_true,
             "echo, no noise": (kind == "none") & echo_true, "noise, no echo": (kind != "none") & ~echo_true,
             "echo + noise": (kind != "none") & echo_true, "babble": kind == "babble"}
    out["v2_by_condition"] = {n: {k: _eer(d["y"][m], d[c][m]) for k, c in (("wavlm", "W"), ("whisper", "H"), ("fixed", "F"), ("gated", "G"))}
                              | {"n": int(m.sum()), "mean_q": float(d["q"][m].mean())} for n, m in conds.items()}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config")
    common.add_argument("--workers", type=int, default=8)
    p = sub.add_parser("tuning", parents=[common])
    p.add_argument("--bundle", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_tuning)
    p = sub.add_parser("fit", parents=[common])
    p.add_argument("--n", type=int, default=30000)
    p.add_argument("--tuning", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_fit)
    p = sub.add_parser("features", parents=[common])
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="eval")
    p.add_argument("--scores", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_features)
    p = sub.add_parser("eval", parents=[common])
    p.add_argument("--results", default="../results")
    p.add_argument("--tuning", required=True)
    p.add_argument("--estimator", required=True)
    p.add_argument("--v2-protocol", default="../dataset_calls_v2/ASV5.eval.calls.tsv")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_eval)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
