"""Command line: audiodf {audit,prepare,train,evaluate,predict,benchmark,serve,produce,consume}."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audiodf.config import load_settings


def _cmd_audit(args, settings):
    from audiodf.data.integrity import audit_dataset, write_report

    report = audit_dataset(settings, args.dataset, None if args.full else args.sample, args.silence_sample)
    path = write_report(report, settings.paths.results_dir)
    print(f"{report['dataset']}: {'OK' if report['ok'] else 'FAILED'}  (report: {path})")
    for e in report["errors"]:
        print(f"  ERROR   {e}")
    for f in report["findings"]:
        print(f"  finding {f}")
    if not report["ok"]:
        raise SystemExit(1)


def _cmd_prepare(args, settings):
    from pathlib import Path

    from audiodf.data.prepare import build_index
    from audiodf.data.protocol import SPLIT_TABLES

    table = SPLIT_TABLES[args.dataset]
    root = Path(settings.dataset_root(args.dataset))
    for split in args.splits:
        if split not in table:
            continue  # e.g. asv21 has only eval
        audio = root / table[split][1]
        if not audio.is_dir():
            print(f"  skipping {args.dataset}/{split}: audio not downloaded ({audio})")
            continue
        build_index(settings, args.dataset, split, args.limit, args.workers)


def _cmd_train(args, settings):
    from audiodf.training.pipeline import run_training

    if args.epochs:
        settings.rcnn_train.epochs = args.epochs
    if args.svm_utts:
        settings.data.svm_train_utts = args.svm_utts
    if args.eval_utts is not None:
        settings.data.eval_utts = args.eval_utts
    run_training(settings, args.limit, args.workers)


def _cmd_evaluate(args, settings):
    from audiodf.training.pipeline import evaluate_artifacts

    from pathlib import Path

    if args.bundle:
        settings.paths.artifacts_dir = args.bundle
    name = f"evaluate_{args.dataset}_{args.split}{f'_{args.tag}' if args.tag else ''}"
    results = Path(settings.paths.results_dir)
    results.mkdir(parents=True, exist_ok=True)
    rep = evaluate_artifacts(settings, args.dataset, args.split, args.eval_utts, args.workers, args.limit,
                             branches=tuple(args.branches) if args.branches else None,
                             scores_path=results / f"{name}_scores.csv" if args.save_scores else None)
    out = results / f"{name}.json"
    out.write_text(json.dumps(rep, indent=2, default=str))
    print(f"report: {out}")


def _cmd_prepare21(args, settings):
    from audiodf.data.asv21 import extract

    extract(args.parquet_dir or Path(settings.paths.asv21_root) / "data", settings.paths.asv21_root)


def _cmd_render_calls(args, settings):
    from audiodf.data.prepare import build_index
    from audiodf.data.voip_sim import NOISES, SNR_RANGE, BabblePool, render_calls, select_sources

    idx = build_index(settings, "asv5", "eval", None, args.workers, available_only=True)
    sources = select_sources(idx, args.n_genuine, args.n_per_attack, args.seed)
    out = args.out or settings.paths.calls_root
    noises, babble = NOISES, None
    if args.noise == "babble":  # background talkers: genuine clips of other speakers that are not call sources
        noises, babble = ("babble",), BabblePool.from_index(idx, exclude=sources)
    snr = (args.snr_min, args.snr_max) if args.snr_min is not None else SNR_RANGE
    print(f"{len(sources)} clean ASV5 eval clips -> simulated VoIP calls ({'/'.join(noises)} noise, SNR {snr[0]:g}-{snr[1]:g} "
          f"dB) in {out}")
    render_calls(idx, sources, settings, out, args.seed, args.workers, noises=noises, snr_range=snr, babble=babble)


def _cmd_render_calls_v2(args, settings):
    from audiodf.data.calls_v2 import render_calls_v2
    from audiodf.data.noise_bank import NoiseBank
    from audiodf.data.prepare import build_index
    from audiodf.data.voip_sim import BabblePool, select_sources

    idx = build_index(settings, "asv5", "eval", None, args.workers, available_only=True)
    sources = select_sources(idx, args.n_genuine, args.n_per_attack, args.seed)
    bank = NoiseBank.scan(settings.paths.noise_root)
    babble = BabblePool.from_index(idx, exclude=sources)  # background talkers: ASV5 eval speakers, never training ones
    out = args.out or str(Path(settings.paths.calls_root).parent / "dataset_calls_v2")
    print(f"{len(sources)} clean ASV5 eval clips -> v2 calls (held-out noise, real room echo, bursty loss) in {out}")
    render_calls_v2(idx, sources, settings, out, bank, babble, args.seed, args.workers)


def _cmd_prepare_noise(args, settings):
    from audiodf.data.noise_bank import NoiseBank, prepare_parquet_noise

    root = Path(settings.paths.noise_root)
    for folder, out in (("esc50", "esc50_wav"), ("demand", "demand_wav")):
        files = sorted((root / folder).glob("*.parquet"))
        if files:
            prepare_parquet_noise(files, root / out)
    print("noise corpora (files):", NoiseBank.scan(root).corpora())


def _cmd_denoise_calls(args, settings):
    import json as _json

    from audiodf.data.denoise import afftdn_filter, denoise_calls

    info = denoise_calls(args.src or settings.paths.calls_root, args.dst, afftdn_filter(args.nr, args.nf), args.workers)
    out = Path(settings.paths.results_dir) / f"denoise_{args.tag}.json"
    out.write_text(_json.dumps(info, indent=2))
    print(f"report: {out}")


def _cmd_calibrate(args, settings):
    from audiodf.calibrate import calibrate, make_wavlm_bundle, policy_thresholds, repeated_check
    from audiodf.evaluation.channel_report import KINDS, load_protocol, load_scores

    protocol = load_protocol(args.protocol, KINDS[args.kind][0])
    clips, scores = load_scores(args.scores, protocol, args.column)
    stored = json.loads((Path(args.stored) / "bundle.json").read_text())["risk"] if args.stored else None
    rep = calibrate(scores, protocol["label"][clips], protocol["speaker"][clips], protocol["codec"][clips],
                    settings.risk.block_fpr, settings.risk.verify_fpr, args.seed, stored)
    if args.repeat:
        rep["repeated_speaker_splits"] = repeated_check(scores, protocol["label"][clips], protocol["speaker"][clips],
                                                        args.repeat, settings.risk.block_fpr, settings.risk.verify_fpr,
                                                        args.seed)
    rep["source"] = {"scores": str(args.scores), "protocol": str(args.protocol), "column": args.column}
    rep["risk_all_speakers"] = policy_thresholds(scores, protocol["label"][clips], settings.risk.block_fpr,
                                                 settings.risk.verify_fpr)
    if args.out_bundle:
        # Thresholds from half the speakers move with which speakers are in that half (the repeated check shows it),
        # so the shipped bundle can use all of them; the half / half and repeated checks validate the procedure.
        bundle_risk = rep["risk_all_speakers"] if args.thresholds_from == "all" else rep["risk"]
        make_wavlm_bundle(args.src_bundle, args.out_bundle, bundle_risk,
                          {k: rep[k] for k in ("policy", "seed", "source")}
                          | {"thresholds_from": args.thresholds_from, "clips_used": len(scores) if args.thresholds_from
                             == "all" else rep["half_a"]["clips"], "half_a_clips": rep["half_a"]["clips"],
                             "half_b_clips": rep["half_b"]["clips"],
                             "repeated_speaker_splits": rep.get("repeated_speaker_splits")})
        rep["bundle"] = str(args.out_bundle)
        rep["bundle_risk"] = bundle_risk
    out = Path(settings.paths.results_dir) / f"calibration_{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2))
    for half in ("half_a", "half_b"):
        o = rep[half]["overall"]
        print(f"{half} ({rep[half]['clips']} clips, {rep[half]['speakers']} speakers): verify flags "
              f"{o['verify']['bonafide_flagged']:.1%} of genuine, catches {o['verify']['spoof_caught']:.1%}; block flags "
              f"{o['block']['bonafide_flagged']:.1%}, catches {o['block']['spoof_caught']:.1%}")
    if "repeated_speaker_splits" in rep:
        r = rep["repeated_speaker_splits"]
        for k in ("verify", "block"):
            f, c = r[f"{k}_bonafide_flagged"], r[f"{k}_spoof_caught"]
            print(f"{r['splits']} speaker splits, held-out half, {k}: genuine flagged {f['mean']:.1%} +- {f['std']:.1%} "
                  f"(range {f['min']:.1%}-{f['max']:.1%}), fakes caught {c['mean']:.1%} +- {c['std']:.1%}")
    print(f"thresholds (set on half A): verify >= {rep['risk']['medium']:.5f}, block >= {rep['risk']['high']:.5f}\n"
          f"report: {out}")


def _cmd_live_check(args, settings):
    from audiodf.evaluation.live_check import live_check

    if args.bundle:
        settings.paths.artifacts_dir = args.bundle
    rep = live_check(settings, args.dataset, "eval", args.scores, args.column, args.n)
    out = Path(settings.paths.results_dir) / f"live_check_{args.tag}.json"
    out.write_text(json.dumps(rep, indent=2))
    print(f"report: {out}")


def _cmd_predict(args, settings):
    from audiodf.data.audio import load_audio
    from audiodf.inference.engine import DetectionEngine

    engine = DetectionEngine.from_artifacts(settings)
    verdict = engine.predict_waveform(load_audio(args.audio, settings.audio.sample_rate))
    print(json.dumps(verdict.to_dict(), indent=2) if verdict else "no speech detected")


def _cmd_benchmark(args, settings):
    from audiodf.evaluation.benchmark import benchmark
    from audiodf.inference.engine import DetectionEngine

    print(json.dumps(benchmark(DetectionEngine.from_artifacts(settings)), indent=2))


def _cmd_serve(args, settings):
    import os

    import uvicorn

    if args.config:
        os.environ["AUDIODF_CONFIG"] = args.config
    uvicorn.run("audiodf.serving.api:get_app", factory=True, host=args.host, port=args.port)


def _cmd_produce(args, settings):
    from audiodf.streaming.kafka_io import simulate_call

    n = simulate_call(settings, args.audio, args.call_id, args.realtime)
    print(f"sent {n} chunks for call {args.call_id}")


def _cmd_consume(args, settings):
    from audiodf.streaming.kafka_io import run_consumer

    run_consumer(settings, metrics_port=args.metrics_port)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="audiodf")
    p.add_argument("--config", help="YAML settings file (or set AUDIODF_CONFIG)")
    sub = p.add_subparsers(dest="command", required=True)

    def add(name, fn, help_):
        sp = sub.add_parser(name, help=help_)
        sp.set_defaults(fn=fn)
        return sp

    sp = add("train", _cmd_train, "train on ASVspoof5, tune on dev, test, save artifacts")
    sp.add_argument("--limit", type=int, help="utterances per split (smoke test)")
    sp.add_argument("--workers", type=int, default=8)
    sp.add_argument("--epochs", type=int)
    sp.add_argument("--svm-utts", type=int, help="clips the SVM trains on")
    sp.add_argument("--eval-utts", type=int, help="test clips scored per report (0 = whole split)")
    sp = add("evaluate", _cmd_evaluate, "score saved artifacts on a dataset split")
    sp.add_argument("--dataset", choices=["asv5", "asv19", "asv21", "calls"], default="asv5",
                    help="asv21 = ASVspoof 2021 LA eval, real telephony channels (run `prepare21` first); "
                         "calls = simulated VoIP calls from ASV5 eval clips (run `render-calls` first)")
    sp.add_argument("--split", choices=["train", "dev", "eval"], default="eval")
    sp.add_argument("--eval-utts", type=int, default=60000, help="clips to score (0 = whole split)")
    sp.add_argument("--limit", type=int)
    sp.add_argument("--workers", type=int, default=8)
    sp.add_argument("--bundle", help="artifacts folder to score (default: paths.artifacts_dir)")
    sp.add_argument("--branches", nargs="+", choices=["svm", "rcnn", "wavlm"],
                    help="score only these branches of the bundle (e.g. wavlm out of a fused bundle)")
    sp.add_argument("--tag", help="suffix for the report file name, e.g. run4_wavlm")
    sp.add_argument("--save-scores", action="store_true", help="also write every clip's scores to a CSV")
    sp = add("prepare21", _cmd_prepare21, "unpack the ASVspoof 2021 LA eval parquet download into FLAC + protocol")
    sp.add_argument("--parquet-dir", help="folder with test-*.parquet (default: <asv21_root>/data)")
    sp = add("render-calls", _cmd_render_calls, "build simulated VoIP calls (noise, codec, packet loss) from clean "
             "ASV5 eval clips")
    sp.add_argument("--n-genuine", type=int, default=4800)
    sp.add_argument("--n-per-attack", type=int, default=300)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--workers", type=int, default=8)
    sp.add_argument("--out", help="folder for the call set (default: paths.calls_root)")
    sp.add_argument("--noise", choices=["mixed", "babble"], default="mixed",
                    help="mixed = white / pink / brown (first call set); babble = other speakers talking in the background")
    sp.add_argument("--snr-min", type=float, help="lowest SNR in dB of the noisy calls (default 15)")
    sp.add_argument("--snr-max", type=float, help="highest SNR in dB (default 35)")
    sp = add("render-calls-v2", _cmd_render_calls_v2, "build the second call set: held-out noise (ESC-50, eval-speaker "
             "babble), real room echo, long bursty packet loss with an unseen concealment style")
    sp.add_argument("--n-genuine", type=int, default=3200)
    sp.add_argument("--n-per-attack", type=int, default=200)
    sp.add_argument("--seed", type=int, default=2)
    sp.add_argument("--workers", type=int, default=8)
    sp.add_argument("--out", help="folder for the call set (default: dataset_calls_v2 next to dataset_calls)")
    add("prepare-noise", _cmd_prepare_noise, "unpack the ESC-50 and DEMAND parquet downloads in paths.noise_root to 16 kHz FLAC")
    sp = add("denoise-calls", _cmd_denoise_calls, "denoise a simulated call set with ffmpeg afftdn (test of a front end)")
    sp.add_argument("--src", help="call set folder to denoise (default: paths.calls_root)")
    sp.add_argument("--dst", required=True, help="folder for the denoised copy")
    sp.add_argument("--nr", type=float, default=12.0, help="noise reduction in dB (afftdn nr)")
    sp.add_argument("--nf", type=float, default=-45.0, help="noise floor in dB (afftdn nf)")
    sp.add_argument("--workers", type=int, default=8)
    sp.add_argument("--tag", required=True)
    sp = add("calibrate", _cmd_calibrate, "set verify/block thresholds on one half of the speakers of scored phone-"
             "channel clips, check them on the other half, optionally write a WavLM-only bundle")
    sp.add_argument("--scores", required=True, help="per-clip scores CSV from `evaluate --save-scores`")
    sp.add_argument("--protocol", required=True)
    sp.add_argument("--kind", choices=["asv21", "calls"], default="asv21")
    sp.add_argument("--column", default="wavlm_10s")
    sp.add_argument("--seed", type=int, default=0, help="speaker split seed")
    sp.add_argument("--repeat", type=int, default=0, help="also check over this many random speaker splits")
    sp.add_argument("--tag", required=True, help="name of the report file (results/calibration_<tag>.json)")
    sp.add_argument("--stored", help="a bundle folder whose stored thresholds are also checked on half B")
    sp.add_argument("--src-bundle", help="bundle to take the WavLM branch from (with --out-bundle)")
    sp.add_argument("--out-bundle", help="write a WavLM-only bundle with the calibrated thresholds here")
    sp.add_argument("--thresholds-from", choices=["half_a", "all"], default="all",
                    help="thresholds for the bundle: from half A only, or from all speakers (default; more stable)")
    sp = add("live-check", _cmd_live_check, "stream clips through CallSession and compare with batched scores")
    sp.add_argument("--scores", required=True, help="scores CSV of the same bundle on the same dataset")
    sp.add_argument("--dataset", choices=["asv21", "calls"], default="calls")
    sp.add_argument("--bundle", help="bundle to serve (default: paths.artifacts_dir)")
    sp.add_argument("--column", default="fused_10s")
    sp.add_argument("--n", type=int, default=400)
    sp.add_argument("--tag", required=True)
    sp = add("audit", _cmd_audit, "dataset integrity audit (exit 1 on hard errors)")
    sp.add_argument("--dataset", choices=["asv5", "asv19"], default="asv5")
    sp.add_argument("--sample", type=int, default=3000, help="files per split for format/duration checks")
    sp.add_argument("--silence-sample", type=int, default=600, help="files per split decoded for silence checks")
    sp.add_argument("--full", action="store_true", help="check the format of every file")
    sp = add("prepare", _cmd_prepare, "index clips (decode once, record VAD speech bounds)")
    sp.add_argument("--dataset", choices=["asv5", "asv19"], default="asv5")
    sp.add_argument("--splits", nargs="+", default=["train", "dev", "eval"])
    sp.add_argument("--limit", type=int)
    sp.add_argument("--workers", type=int, default=8)
    add("predict", _cmd_predict, "score one audio file").add_argument("audio")
    add("benchmark", _cmd_benchmark, "per-stage latency")
    sp = add("serve", _cmd_serve, "run the FastAPI service")
    sp.add_argument("--host", default="0.0.0.0")
    sp.add_argument("--port", type=int, default=8000)
    sp = add("produce", _cmd_produce, "simulate a call into Kafka")
    sp.add_argument("audio")
    sp.add_argument("--call-id", default="call-1")
    sp.add_argument("--realtime", action="store_true", help="pace chunks at wall-clock speed")
    sp = add("consume", _cmd_consume, "run the Kafka detector worker")
    sp.add_argument("--metrics-port", type=int, default=9100)

    args = p.parse_args(argv)
    args.fn(args, load_settings(args.config))


if __name__ == "__main__":
    main()
