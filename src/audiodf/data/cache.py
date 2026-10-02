"""On-disk feature cache (replaces the notebook's RAM-heavy CSV export).

Per split directory:
  utt_svm.npy      (n_utt, 318)         SVM features, one vector per whole utterance
  logmel.f16       (n_seg, 64, 200)     RCNN features, one spectrogram per 2 s segment (memory-mapped)
  seg_utt.npy      (n_seg,)             utterance index of every segment
  utt_label.npy / utt_attack.npy / meta.json
The two feature sets are produced by separate extractors; the audio is only read once.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from audiodf.config import Settings
from audiodf.data.audio import load_audio
from audiodf.data.protocol import SPLITS, Sample, read_protocol
from audiodf.data.segmenter import segment
from audiodf.features.rcnn_features import RcnnFeatureExtractor
from audiodf.features.svm_features import SvmFeatureExtractor


def cache_tag(settings: Settings, limit: int | None = None) -> str:
    seg, hop = settings.segment.seconds, settings.segment.hop_seconds
    tag = f"seg{seg:g}s_hop{hop:g}s"
    return f"{tag}_limit{limit}" if limit else tag


class _ExtractDataset(Dataset):
    def __init__(self, samples: list[Sample], settings: Settings):
        self.samples, self.settings = samples, settings
        self._svm = self._rcnn = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.settings
        if self._svm is None:
            self._svm = SvmFeatureExtractor(s.audio, s.svm_features)
            self._rcnn = RcnnFeatureExtractor(s.audio, s.rcnn_features, s.segment_samples)
        wave = load_audio(self.samples[i].path, s.audio.sample_rate)
        segs = segment(wave, s.segment_samples, s.segment_hop_samples)
        return {"utt_svm": self._svm.extract(wave),
                "logmel": self._rcnn.extract(segs).astype(np.float16)}


def _worker_init(_):
    torch.set_num_threads(1)


def build_split_cache(samples: list[Sample], out_dir: Path, settings: Settings, workers: int = 8) -> None:
    if (out_dir / "meta.json").exists():
        print(f"  cached: {out_dir}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(_ExtractDataset(samples, settings), batch_size=None, num_workers=workers,
                        worker_init_fn=_worker_init, prefetch_factor=8 if workers else None)
    utt_svm, seg_utt, n_seg, t0 = [], [], 0, time.time()
    with open(out_dir / "logmel.f16", "wb") as f:
        for i, item in enumerate(loader):
            mel = item["logmel"].numpy()
            f.write(mel.tobytes())
            utt_svm.append(item["utt_svm"].numpy())
            seg_utt.append(np.full(len(mel), i, dtype=np.int32))
            n_seg += len(mel)
            if (i + 1) % 2000 == 0:
                print(f"    {i + 1}/{len(samples)} utts, {(i + 1) / (time.time() - t0):.0f} utt/s", flush=True)
    np.save(out_dir / "utt_svm.npy", np.stack(utt_svm))
    np.save(out_dir / "seg_utt.npy", np.concatenate(seg_utt))
    np.save(out_dir / "utt_label.npy", np.array([s.label for s in samples], dtype=np.int8))
    np.save(out_dir / "utt_attack.npy", np.array([s.attack for s in samples]))
    (out_dir / "meta.json").write_text(json.dumps({"n_utt": len(samples), "n_seg": n_seg}))
    print(f"  extracted {len(samples)} utts / {n_seg} segs in {time.time() - t0:.0f}s")


@dataclass
class SplitData:
    utt_svm: np.ndarray
    logmel: np.ndarray
    seg_utt: np.ndarray
    utt_label: np.ndarray
    utt_attack: np.ndarray

    @classmethod
    def load(cls, d: Path, mel_shape: tuple[int, int]) -> "SplitData":
        meta = json.loads((d / "meta.json").read_text())
        return cls(
            utt_svm=np.load(d / "utt_svm.npy"),
            logmel=np.memmap(d / "logmel.f16", dtype=np.float16, mode="r",
                             shape=(meta["n_seg"], *mel_shape)),
            seg_utt=np.load(d / "seg_utt.npy"),
            utt_label=np.load(d / "utt_label.npy").astype(np.int64),
            utt_attack=np.load(d / "utt_attack.npy"),
        )

    @property
    def n_seg(self) -> int:
        return len(self.seg_utt)

    @property
    def seg_label(self) -> np.ndarray:
        return self.utt_label[self.seg_utt]

    def segments_to_utterances(self, seg_scores: np.ndarray) -> np.ndarray:
        """Mean of segment scores per utterance."""
        n = len(self.utt_label)
        return np.bincount(self.seg_utt, weights=seg_scores, minlength=n) / np.bincount(self.seg_utt, minlength=n)


def load_or_build(settings: Settings, limit: int | None = None, workers: int = 8,
                  splits=("train", "dev", "eval")) -> dict[str, SplitData]:
    root = Path(settings.paths.cache_dir) / cache_tag(settings, limit)
    mel_shape = (settings.rcnn_features.n_mels, settings.segment_frames)
    out = {}
    for split in splits:
        print(f" {split}")
        build_split_cache(read_protocol(settings.paths.data_root, split, limit), root / split, settings, workers)
        out[split] = SplitData.load(root / split, mel_shape)
    return out
