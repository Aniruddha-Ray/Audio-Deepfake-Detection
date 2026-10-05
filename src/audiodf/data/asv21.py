"""ASVspoof 2021 LA eval as a telephony test set (real VoIP / PSTN transmission of the 2019 LA attacks).

Source: the Hugging Face packaging `SpeechAntiSpoofingBenchmarks/ASVspoof2021_LA` (24 parquet files, 181,566 trials, the
full official LA eval including the progress and hidden phases). Its audio is the original samples re-encoded as clean
16 kHz FLAC (the official FLAC files use an encoding libsndfile often cannot read). Each row carries the channel:
codec (none, alaw, ulaw, gsm, g722, opus, pstn), transmission route (local, Italy, Singapore, Madrid PSTN), trim
(hidden-phase clips are silence-trimmed) and phase. `extract` writes the FLAC files and a protocol file that
data/protocol.py reads; it is resumable and never decodes audio.

Test data only: nothing here is used for training or tuning.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

PROTOCOL_FILE, AUDIO_DIR = "ASVspoof2021.LA.eval.tsv", "flac_eval"
COLUMNS = ("utt_id", "speaker", "codec", "transmission", "attack", "key", "phase", "trim")


def extract(parquet_dir: str | Path, out_root: str | Path, log=print) -> int:
    """Write every clip of the parquet files to out_root/flac_eval/ and the protocol TSV to out_root.
    Returns the number of clips. Existing files are kept, so an interrupted extraction can be re-run."""
    import pyarrow.parquet as pq

    parquet_dir, out_root = Path(parquet_dir), Path(out_root)
    files = sorted(parquet_dir.glob("test-*-of-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no test-*.parquet files in {parquet_dir}")
    audio_dir = out_root / AUDIO_DIR
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows, written, kept = [], 0, 0
    for k, path in enumerate(files, 1):
        pf = pq.ParquetFile(path)
        for g in range(pf.num_row_groups):  # one row group at a time keeps memory small
            table = pf.read_row_group(g, columns=["path", "audio", "label", "notes"])
            for audio, label, notes in zip(table.column("audio").to_pylist(), table.column("label").to_pylist(),
                                           table.column("notes").to_pylist()):
                meta = json.loads(notes)
                utt = meta["utterance_id"]
                target = audio_dir / f"{utt}.flac"
                if target.exists() and target.stat().st_size:
                    kept += 1
                else:
                    tmp = target.with_suffix(".part")
                    tmp.write_bytes(audio["bytes"])
                    os.replace(tmp, target)
                    written += 1
                key = "spoof" if label == 1 else "bonafide"
                if (key == "spoof") != (meta["attack_id"] != "bonafide"):
                    raise ValueError(f"{utt}: label {key!r} disagrees with attack {meta['attack_id']!r}")
                rows.append((utt, meta["speaker_id"], meta["codec"], meta["transmission"], meta["attack_id"], key,
                             meta["phase"], meta["trim"]))
        log(f"  {k}/{len(files)} {path.name}: {written} written, {kept} already there")
    if len({r[0] for r in rows}) != len(rows):
        raise ValueError("duplicate utterance ids in the parquet files")
    tmp = out_root / (PROTOCOL_FILE + ".part")
    tmp.write_text("\n".join(" ".join(r) for r in rows) + "\n")
    os.replace(tmp, out_root / PROTOCOL_FILE)
    log(f"wrote {len(rows)} trials to {out_root / PROTOCOL_FILE}")
    return len(rows)
