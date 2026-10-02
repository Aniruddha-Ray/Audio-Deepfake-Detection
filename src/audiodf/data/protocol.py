"""ASVspoof2019 LA protocol parsing. Label convention everywhere: 1 = spoof (fake), 0 = bonafide."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

SPLITS = {
    "train": ("ASVspoof2019.LA.cm.train.trn.txt", "ASVspoof2019_LA_train"),
    "dev": ("ASVspoof2019.LA.cm.dev.trl.txt", "ASVspoof2019_LA_dev"),
    "eval": ("ASVspoof2019.LA.cm.eval.trl.txt", "ASVspoof2019_LA_eval"),
}


@dataclass(frozen=True)
class Sample:
    path: str
    label: int
    attack: str
    utt_id: str


def read_protocol(data_root: str | Path, split: str, limit: int | None = None) -> list[Sample]:
    data_root = Path(data_root)
    proto, audio_dir = SPLITS[split]
    samples = []
    with open(data_root / "ASVspoof2019_LA_cm_protocols" / proto) as f:
        for line in f:
            _, utt, _, attack, key = line.split()  # attack is column 4; "-" for bonafide
            samples.append(Sample(str(data_root / audio_dir / "flac" / f"{utt}.flac"),
                                  int(key == "spoof"), attack, utt))
    if limit and limit < len(samples):
        keep = np.sort(np.random.default_rng(0).choice(len(samples), limit, replace=False))
        samples = [samples[i] for i in keep]
    return samples
