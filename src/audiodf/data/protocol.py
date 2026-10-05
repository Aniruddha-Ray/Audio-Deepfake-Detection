"""Protocol parsing for ASVspoof2019 LA, ASVspoof5 (Track 1) and ASVspoof 2021 LA eval (a test-only telephony set).

Label convention everywhere: 1 = spoof (fake), 0 = bonafide. Bonafide rows get attack "-".
Attack IDs are separate namespaces per dataset ("A01" is a different system in each), so
`Sample.attack_key` prefixes the dataset whenever samples from both are reported together.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ASV19_SPLITS = {
    "train": ("ASVspoof2019.LA.cm.train.trn.txt", "ASVspoof2019_LA_train/flac"),
    "dev": ("ASVspoof2019.LA.cm.dev.trl.txt", "ASVspoof2019_LA_dev/flac"),
    "eval": ("ASVspoof2019.LA.cm.eval.trl.txt", "ASVspoof2019_LA_eval/flac"),
}
ASV5_SPLITS = {
    "train": ("ASVspoof5.train.tsv", "flac_T"),
    "dev": ("ASVspoof5.dev.track_1.tsv", "flac_D"),
    "eval": ("ASVspoof5.eval.track_1.tsv", "flac_E_eval"),
}
# Written by data/asv21.py from the parquet download. Test only: it is never part of a training pool.
ASV21_SPLITS = {"eval": ("ASVspoof2021.LA.eval.tsv", "flac_eval")}
SPLIT_TABLES = {"asv19": ASV19_SPLITS, "asv5": ASV5_SPLITS, "asv21": ASV21_SPLITS}


@dataclass(frozen=True)
class Sample:
    path: str
    label: int
    attack: str
    utt_id: str
    speaker: str = ""
    codec: str = "-"
    dataset: str = "asv19"

    @property
    def attack_key(self) -> str:
        return f"{self.dataset}:{self.attack}"


def _read_asv19(root: Path, split: str) -> list[Sample]:
    proto, audio_dir = ASV19_SPLITS[split]
    out = []
    with open(root / "ASVspoof2019_LA_cm_protocols" / proto) as f:
        for line in f:
            speaker, utt, _, attack, key = line.split()  # attack: column 4, "-" for bonafide
            out.append(Sample(str(root / audio_dir / f"{utt}.flac"), int(key == "spoof"), attack, utt,
                              speaker, "-", "asv19"))
    return out


def _read_asv5(root: Path, split: str) -> list[Sample]:
    """Columns: SPEAKER_ID FLAC_FILE_NAME GENDER CODEC CODEC_Q CODEC_SEED ATTACK_TAG ATTACK_LABEL KEY TMP.
    ATTACK_TAG (AC1-AC3) is the attacker adaptation condition, not a codec; CODEC is column 4."""
    proto, audio_dir = ASV5_SPLITS[split]
    out = []
    with open(root / proto) as f:
        for line in f:
            p = line.split()
            if len(p) != 10:
                raise ValueError(f"{proto}: expected 10 columns, got {len(p)}: {line!r}")
            speaker, utt, codec, attack, key = p[0], p[1], p[3], p[7], p[8]
            spoof = key == "spoof"
            out.append(Sample(str(root / audio_dir / f"{utt}.flac"), int(spoof),
                              attack if spoof else "-", utt, speaker, codec, "asv5"))
    return out


def _read_asv21(root: Path, split: str) -> list[Sample]:
    """Columns: utt_id speaker codec transmission attack key phase trim (see data/asv21.py). The condition used for
    per-channel reporting is the codec (none -> "-", alaw, ulaw, gsm, g722, opus, pstn); the transmission route, phase
    and trim stay in the protocol file for finer breakdowns."""
    proto, audio_dir = ASV21_SPLITS[split]
    out = []
    with open(root / proto) as f:
        for line in f:
            p = line.split()
            if len(p) != 8:
                raise ValueError(f"{proto}: expected 8 columns, got {len(p)}: {line!r}")
            utt, speaker, codec, _, attack, key = p[:6]
            out.append(Sample(str(root / audio_dir / f"{utt}.flac"), int(key == "spoof"),
                              attack if key == "spoof" else "-", utt, speaker, "-" if codec == "none" else codec,
                              "asv21"))
    return out


def read_protocol(data_root: str | Path, split: str, limit: int | None = None,
                  dataset: str = "asv19", available_only: bool = False) -> list[Sample]:
    """available_only keeps just the clips whose audio is on disk (a partially downloaded test split)."""
    readers = {"asv19": _read_asv19, "asv5": _read_asv5, "asv21": _read_asv21}
    if dataset not in readers:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {sorted(readers)}")
    samples = readers[dataset](Path(data_root), split)
    if available_only:
        audio_dir = Path(data_root) / SPLIT_TABLES[dataset][split][1]
        have = {f[:-5] for f in os.listdir(audio_dir) if f.endswith(".flac")}
        samples = [s for s in samples if s.utt_id in have]
    if limit and limit < len(samples):
        keep = np.sort(np.random.default_rng(0).choice(len(samples), limit, replace=False))
        samples = [samples[i] for i in keep]
    return samples
