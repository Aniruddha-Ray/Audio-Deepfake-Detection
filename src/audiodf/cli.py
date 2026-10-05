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
    sp.add_argument("--dataset", choices=["asv5", "asv19", "asv21"], default="asv5",
                    help="asv21 = ASVspoof 2021 LA eval, real telephony channels (run `prepare21` first)")
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
