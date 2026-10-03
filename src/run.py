"""One-file entry point: `python run.py` runs the whole training pipeline on ASVspoof5.

  1. audit      integrity checks; training refuses to start on hard errors
  2. index      decode every clip once, record where speech starts and ends (VAD)
  3. SVM        growing-buffer snapshots (2-10 s), codec-augmented
  4. RCNN       random 2 s windows read from the FLAC files, codec-augmented; best dev epoch kept
  5. tune       on dev only: SVM calibration, fusion weight, risk thresholds
  6. test       ASV5 eval (and ASV2019 eval as a cross-dataset check), scored once
  Saves artifacts/ (models + tuned operating point) and results/training_report.json.

Flags: --epochs N  --svm-utts N  --eval-utts N (0 = whole eval split)  --limit N (smoke test)
       --train-on asv5 asv19 (mixed training)
       --workers N  --config FILE  --rnn {bilstm,lstm}
Serving and Kafka are separate (`python -m audiodf serve|consume`) and not needed for training.
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
    ap.add_argument("--epochs", type=int, help="RCNN epochs (default from config)")
    ap.add_argument("--svm-utts", type=int, help="clips the SVM trains on (default from config)")
    ap.add_argument("--eval-utts", type=int, help="test clips scored per report; 0 = whole split")
    ap.add_argument("--train-on", nargs="+", choices=["asv5", "asv19"],
                    help="training sources (default asv5); 'asv5 asv19' adds ASVspoof2019 train+dev")
    ap.add_argument("--rnn", choices=["bilstm", "lstm"], help="recurrent layer (default: bilstm)")
    ap.add_argument("--limit", type=int, help="utterances per split, for a quick smoke test")
    ap.add_argument("--workers", type=int, default=8, help="data-loading processes")
    args = ap.parse_args()

    settings = load_settings(args.config)
    if args.epochs:
        settings.rcnn_train.epochs = args.epochs
    if args.svm_utts:
        settings.data.svm_train_utts = args.svm_utts
    if args.eval_utts is not None:
        settings.data.eval_utts = args.eval_utts
    if args.train_on:
        settings.data.train_datasets = tuple(args.train_on)
    if args.rnn:
        settings.rcnn_model.bidirectional = args.rnn == "bilstm"

    root = Path(settings.dataset_root("asv5"))
    if not (root / "ASVspoof5.train.tsv").exists() or not (root / "flac_T").is_dir():
        sys.exit(f"ASVspoof5 train/dev data not found at {root}\n"
                 f"Set AUDIODF_ASV5 to the folder containing ASVspoof5.train.tsv and flac_T/, flac_D/.")

    print(f"asv5={root}\nartifacts={settings.paths.artifacts_dir}\n"
          f"rcnn={'BiLSTM' if settings.rcnn_model.bidirectional else 'LSTM'}, epochs={settings.rcnn_train.epochs}, "
          f"svm clips={settings.data.svm_train_utts}, codec aug p={settings.data.codec_aug_p}")
    run_training(settings, limit=args.limit, workers=args.workers)


if __name__ == "__main__":
    main()
