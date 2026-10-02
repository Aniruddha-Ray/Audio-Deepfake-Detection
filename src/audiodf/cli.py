"""Command line: audiodf {extract,train,evaluate,predict,benchmark,serve,produce,consume}."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audiodf.config import load_settings


def _cmd_extract(args, settings):
    from audiodf.data.cache import load_or_build

    load_or_build(settings, args.limit, args.workers)


def _cmd_train(args, settings):
    from audiodf.training.pipeline import run_training

    if args.epochs:
        settings.rcnn_train.epochs = args.epochs
    run_training(settings, args.limit, args.workers, Path(args.rcnn_checkpoint) if args.rcnn_checkpoint else None)


def _cmd_evaluate(args, settings):
    from audiodf.artifacts import load_bundle
    from audiodf.data.cache import load_or_build
    from audiodf.training.pipeline import evaluate, pick_device

    device = pick_device()
    bundle = load_bundle(settings, device)
    splits = load_or_build(settings, args.limit, args.workers, splits=("dev", "eval"))
    print(json.dumps(evaluate(bundle.svm, bundle.rcnn, splits, settings, device), indent=2))


def _cmd_predict(args, settings):
    from audiodf.data.audio import load_audio
    from audiodf.inference.engine import DetectionEngine

    engine = DetectionEngine.from_artifacts(settings)
    wave = load_audio(args.audio, settings.audio.sample_rate)
    print(json.dumps(engine.predict_waveform(wave).to_dict(), indent=2))


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

    for name, fn, help_ in (("extract", _cmd_extract, "build the feature cache"),
                            ("train", _cmd_train, "train SVM + RCNN, evaluate, save artifacts"),
                            ("evaluate", _cmd_evaluate, "evaluate saved artifacts on dev/eval")):
        sp = add(name, fn, help_)
        sp.add_argument("--limit", type=int, help="utterances per split (smoke test)")
        sp.add_argument("--workers", type=int, default=8)
        if name == "train":
            sp.add_argument("--epochs", type=int)
            sp.add_argument("--rcnn-checkpoint", help="reuse this RCNN checkpoint instead of training")
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
