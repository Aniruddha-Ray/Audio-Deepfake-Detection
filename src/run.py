"""One-file entry point: `python run.py` runs the whole training pipeline.

  1. extract features   (SVM vectors + RCNN Log-Mel windows, cached on disk; reused on later runs)
  2. train the SVM      (whole-utterance vectors)
  3. train the RCNN     (CNN-BiLSTM on 2 s windows, best-dev-EER checkpoint)
  4. evaluate           (dev + eval, per-attack EER) and save artifacts/ + results/training_report.json

Optional flags: --epochs N  --rnn {bilstm,lstm}  --limit N (quick smoke test)  --workers N  --config FILE
Serving and Kafka are separate (`python -m audiodf serve|consume`) and are not needed for training.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audiodf.config import load_settings  # noqa: E402
from audiodf.training.pipeline import run_training  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="YAML settings file")
    ap.add_argument("--epochs", type=int, help="RCNN epochs (default from config: 20)")
    ap.add_argument("--rnn", choices=["bilstm", "lstm"], help="recurrent layer (default: bilstm)")
    ap.add_argument("--limit", type=int, help="utterances per split, for a quick smoke test")
    ap.add_argument("--workers", type=int, default=8, help="feature-extraction processes")
    args = ap.parse_args()

    settings = load_settings(args.config)
    if args.epochs:
        settings.rcnn_train.epochs = args.epochs
    if args.rnn:
        settings.rcnn_model.bidirectional = args.rnn == "bilstm"

    protocols = Path(settings.paths.data_root) / "ASVspoof2019_LA_cm_protocols"
    if not protocols.is_dir():
        sys.exit(f"ASVspoof2019 LA data not found at {settings.paths.data_root}\n"
                 f"Set AUDIODF_DATA to the folder that contains ASVspoof2019_LA_cm_protocols/.")

    print(f"data={settings.paths.data_root}\nartifacts={settings.paths.artifacts_dir}\n"
          f"rcnn={'BiLSTM' if settings.rcnn_model.bidirectional else 'LSTM'}, "
          f"epochs={settings.rcnn_train.epochs}, svm_weight={settings.ensemble.svm_weight}")
    run_training(settings, limit=args.limit, workers=args.workers)


if __name__ == "__main__":
    main()
