"""One-file entry point: `python run.py` runs the whole training pipeline on ASVspoof5.

  1. audit      integrity checks; training refuses to start on hard errors
  2. index      decode every clip once, record where speech starts and ends (VAD); pool the training
                splits and cut out the tuning set (held-out attacks + held-out speakers)
  3. branches   WavLM (default): random 2 s raw windows, top layers of WavLM-Base+ fine-tuned. Optional: SVM on
                growing-buffer snapshots (2-10 s), RCNN on 2 s Log-Mel windows. All codec-augmented (ffmpeg codecs
                + EnCodec copies rendered offline); best tuning epoch kept.
  4. tune       on the held-out set only: SVM calibration, fusion weights, risk thresholds
  5. test       ASV5 eval (and ASV2019 eval as a cross-dataset check), scored once
  6. save       artifacts/ (models + tuned operating point) and results/training_report.json

Flags: --branches svm rcnn wavlm   --epochs N (RCNN)  --wavlm-epochs N  --svm-utts N
       --eval-utts N (0 = whole eval split)  --limit N (smoke test)
       --train-splits asv5:train asv5:dev asv19:train asv19:dev   --holdout-attacks asv5:A10 asv5:A12 asv5:A15
       --svm-checkpoint F  --rcnn-checkpoint F  --wavlm-checkpoint F  (reuse a trained branch)
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
    ap.add_argument("--branches", nargs="+", choices=["svm", "rcnn", "wavlm"],
                    help="branches to train and fuse (default: wavlm alone)")
    ap.add_argument("--epochs", type=int, help="RCNN epochs (default from config)")
    ap.add_argument("--wavlm-epochs", type=int, help="WavLM epochs (default from config)")
    ap.add_argument("--svm-utts", type=int, help="clips the SVM trains on (default from config)")
    ap.add_argument("--eval-utts", type=int, help="test clips scored per report; 0 = whole split")
    ap.add_argument("--train-splits", nargs="+", metavar="DATASET:SPLIT",
                    help="training pool, e.g. asv5:train asv5:dev asv19:train asv19:dev (the default)")
    ap.add_argument("--holdout-attacks", nargs="*", metavar="DATASET:ATTACK",
                    help="attacks kept out of training for tuning (default asv5:A10 asv5:A12 asv5:A15); "
                         "give none to tune on ASV5 dev instead")
    for name, ext in (("svm", "joblib"), ("rcnn", "pt"), ("wavlm", "pt")):
        ap.add_argument(f"--{name}-checkpoint", type=Path, metavar=f"FILE.{ext}",
                        help=f"reuse this trained {name.upper() if name != 'wavlm' else 'WavLM'} instead of training "
                             f"one (e.g. after an interrupted run, or from an earlier run on the same pool)")
    ap.add_argument("--rnn", choices=["bilstm", "lstm"], help="recurrent layer (default: bilstm)")
    ap.add_argument("--limit", type=int, help="utterances per split, for a quick smoke test")
    ap.add_argument("--workers", type=int, default=8, help="data-loading processes")
    args = ap.parse_args()

    settings = load_settings(args.config)
    if args.branches:
        settings.ensemble.branches = tuple(args.branches)
    if args.epochs:
        settings.rcnn_train.epochs = args.epochs
    if args.wavlm_epochs:
        settings.wavlm.epochs = args.wavlm_epochs
    if args.svm_utts:
        settings.data.svm_train_utts = args.svm_utts
    if args.eval_utts is not None:
        settings.data.eval_utts = args.eval_utts
    if args.train_splits:
        settings.data.train_splits = tuple(args.train_splits)
    if args.holdout_attacks is not None:
        settings.data.holdout_attacks = tuple(args.holdout_attacks)
    if args.rnn:
        settings.rcnn_model.bidirectional = args.rnn == "bilstm"

    root = Path(settings.dataset_root("asv5"))
    if not (root / "ASVspoof5.train.tsv").exists() or not (root / "flac_T").is_dir():
        sys.exit(f"ASVspoof5 train/dev data not found at {root}\n"
                 f"Set AUDIODF_ASV5 to the folder containing ASVspoof5.train.tsv and flac_T/, flac_D/.")

    checkpoints = {"svm": args.svm_checkpoint, "rcnn": args.rcnn_checkpoint, "wavlm": args.wavlm_checkpoint}
    print(f"asv5={root}\nartifacts={settings.paths.artifacts_dir}\n"
          f"pool={' '.join(settings.data.train_splits)}  holdout={' '.join(settings.data.holdout_attacks) or 'none'}\n"
          f"branches={' '.join(settings.ensemble.branches)}  "
          f"reused={' '.join(f'{k}:{v}' for k, v in checkpoints.items() if v) or 'none'}\n"
          f"rcnn={'BiLSTM' if settings.rcnn_model.bidirectional else 'LSTM'}, epochs={settings.rcnn_train.epochs}; "
          f"wavlm epochs={settings.wavlm.epochs}, top {settings.wavlm.finetune_top} layers fine-tuned; "
          f"windows/clip/epoch={settings.data.rcnn_windows_per_utt}, svm clips={settings.data.svm_train_utts}, "
          f"codec aug p={settings.data.codec_aug_p}")
    run_training(settings, limit=args.limit, workers=args.workers, checkpoints=checkpoints)


if __name__ == "__main__":
    main()
