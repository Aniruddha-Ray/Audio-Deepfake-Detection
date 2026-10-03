"""Dataset integrity audit: run before training on any data.

Hard errors (training must not start): protocol rows without audio, duplicate utterance IDs,
speakers shared between splits, bonafide/attack label inconsistencies, unexpected audio format.
Findings (reported, judged by a human): class balance, attack/codec mix, attack overlap between
splits, and shortcut signals, i.e. properties other than the voice itself that differ between
bonafide and spoof (duration, leading/trailing silence).
"""

from __future__ import annotations

import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

from audiodf.config import VadConfig
from audiodf.data.protocol import ASV5_SPLITS, ASV19_SPLITS, Sample, read_protocol
from audiodf.data.vad import frame_db, trim_silence

SPLIT_TABLES = {"asv19": ASV19_SPLITS, "asv5": ASV5_SPLITS}


def _stats(x) -> dict:
    x = np.asarray(x, dtype=float)
    if not len(x):
        return {}
    return {"n": len(x), "mean": round(float(x.mean()), 3), "p5": round(float(np.percentile(x, 5)), 3),
            "median": round(float(np.median(x)), 3), "p95": round(float(np.percentile(x, 95)), 3)}


def _header(path: str):
    try:
        info = sf.info(path)
        return info.samplerate, info.channels, info.subtype, info.frames / info.samplerate
    except Exception as exc:  # unreadable file is a finding, not a crash
        return None, None, repr(exc), None


def _edges(wave: np.ndarray, sr: int, frame: int = 400, rel_db: float = -40.0) -> tuple[float, float]:
    """Seconds of leading/trailing audio quieter than rel_db below the clip's loudest frame.
    Deliberately a different rule from the VAD, so the post-trim check is not circular."""
    energy = frame_db(wave, frame)
    if not len(energy):
        return 0.0, 0.0
    loud = np.nonzero(energy > energy.max() + rel_db)[0]
    return loud[0] * frame / sr, (len(energy) - 1 - loud[-1]) * frame / sr


def _clip_signals(path: str, vad: VadConfig, frame: int = 400) -> tuple[float, ...]:
    """Leading/trailing silence of the raw clip, the same after VAD trimming, and the fraction of
    frames that are exact digital zeros (padded TTS output, VoIP DTX)."""
    wave, sr = sf.read(path, dtype="float32", always_2d=False)
    n = len(wave) // frame
    zero_frac = float((np.abs(wave[: n * frame].reshape(n, frame)).max(axis=1) == 0).mean()) if n else 0.0
    return (*_edges(wave, sr), *_edges(trim_silence(wave, sr, vad), sr), zero_frac)


def audit_split(samples: list[Sample], audio_dir: Path, sample_n: int | None, silence_n: int,
                expected_sr: int, vad: VadConfig, workers: int = 16, partial_ok: bool = False) -> dict:
    rng = np.random.default_rng(0)
    report: dict = {"rows": len(samples), "errors": []}
    if not audio_dir.is_dir():
        report["audio"] = "not downloaded"
        return report

    on_disk = {f[:-5] for f in os.listdir(audio_dir) if f.endswith(".flac")}
    ids = [s.utt_id for s in samples]
    missing = [u for u in ids if u not in on_disk]
    dupes = [u for u, c in Counter(ids).items() if c > 1]
    report["files_on_disk"] = len(on_disk)
    report["missing_audio"] = len(missing)
    report["audio_not_in_protocol"] = len(on_disk - set(ids))
    report["duplicate_utt_ids"] = len(dupes)
    if missing and partial_ok:  # a partly downloaded test split: score what is there
        report["partial"] = f"{len(ids) - len(missing)} of {len(ids)} clips on disk"
        samples = [s for s in samples if s.utt_id in on_disk]
    elif missing:
        report["errors"].append(f"{len(missing)} protocol rows have no audio, e.g. {missing[:3]}")
    if dupes:
        report["errors"].append(f"{len(dupes)} duplicate utterance IDs, e.g. {dupes[:3]}")

    bad = [s.utt_id for s in samples if (s.label == 0) != (s.attack == "-")]
    if bad:
        report["errors"].append(f"{len(bad)} rows where bonafide/attack disagree, e.g. {bad[:3]}")
    report["label_counts"] = {"bonafide": sum(s.label == 0 for s in samples), "spoof": sum(s.label for s in samples)}
    report["attacks"] = dict(sorted(Counter(s.attack for s in samples if s.label).items()))
    report["codec_by_label"] = {lab: dict(sorted(Counter(s.codec for s in samples if s.label == i).items()))
                                for i, lab in ((0, "bonafide"), (1, "spoof"))}
    report["speakers"] = len({s.speaker for s in samples})

    present = [s for s in samples if s.utt_id in on_disk]
    pick = present if not sample_n or sample_n >= len(present) else \
        [present[i] for i in rng.choice(len(present), sample_n, replace=False)]
    with ThreadPoolExecutor(workers) as pool:
        headers = list(pool.map(lambda s: _header(s.path), pick))
    formats = Counter((sr, ch, sub) for sr, ch, sub, _ in headers)
    report["format_checked"] = len(pick)
    report["formats"] = {f"{sr}Hz/{ch}ch/{sub}": n for (sr, ch, sub), n in formats.items()}
    wrong = sum(n for (sr, ch, sub), n in formats.items() if (sr, ch, sub) != (expected_sr, 1, "PCM_16"))
    if wrong:
        report["errors"].append(f"{wrong} of {len(pick)} checked files are not {expected_sr}Hz mono PCM_16")
    dur = {lab: [d for s, (_, _, _, d) in zip(pick, headers) if s.label == i and d is not None]
           for i, lab in ((0, "bonafide"), (1, "spoof"))}
    report["duration_s_by_label"] = {lab: _stats(v) for lab, v in dur.items()}

    sil_pick = [pick[i] for i in rng.choice(len(pick), min(silence_n, len(pick)), replace=False)]
    with ThreadPoolExecutor(workers) as pool:
        edges = list(pool.map(lambda s: _clip_signals(s.path, vad), sil_pick))
    for col, key in enumerate(("leading_silence_s_by_label", "trailing_silence_s_by_label",
                               "after_vad_leading_silence_s_by_label", "after_vad_trailing_silence_s_by_label",
                               "digital_silence_frac_by_label")):
        report[key] = {lab: _stats([e[col] for s, e in zip(sil_pick, edges) if s.label == i])
                       for i, lab in ((0, "bonafide"), (1, "spoof"))}
    return report


def _shortcut_flags(name: str, rep: dict) -> list[str]:
    """Flag properties that alone separate the classes (a model can learn them instead of the voice).
    Needs both a relative gap (>1.25x) and an absolute one, so near-zero medians (e.g. 13 ms vs 0 ms,
    under one 25 ms VAD frame) don't raise false alarms."""
    flags = []
    for key, label, min_gap in (("duration_s_by_label", "duration", 1.0),
                                ("leading_silence_s_by_label", "leading silence", 0.05),
                                ("trailing_silence_s_by_label", "trailing silence", 0.05),
                                ("after_vad_leading_silence_s_by_label", "leading silence AFTER VAD trim", 0.05),
                                ("after_vad_trailing_silence_s_by_label", "trailing silence AFTER VAD trim", 0.05),
                                ("digital_silence_frac_by_label", "digital-silence frame fraction (mean)", 0.01)):
        b, s = rep.get(key, {}).get("bonafide"), rep.get(key, {}).get("spoof")
        if not (b and s):
            continue
        stat = "mean" if key.startswith("digital") else "median"  # most clips have none, so medians hide it
        hi, lo = max(b[stat], s[stat]), min(b[stat], s[stat])
        if hi - lo > min_gap and hi / max(lo, 1e-3) > 1.25:
            flags.append(f"{name}: {label} differs between bonafide ({b[stat]}) and spoof ({s[stat]})")
    return flags


def audit_dataset(settings, dataset: str, sample_n: int | None = 3000, silence_n: int = 600) -> dict:
    root = Path(settings.dataset_root(dataset))
    table = SPLIT_TABLES[dataset]
    splits = {}
    for split, (proto, audio_dir) in table.items():
        proto_path = root / ("ASVspoof2019_LA_cm_protocols" if dataset == "asv19" else "") / proto
        if not proto_path.exists():
            splits[split] = {"protocol": "missing"}
            continue
        samples = read_protocol(root, split, dataset=dataset)
        splits[split] = audit_split(samples, root / audio_dir, sample_n, silence_n, settings.audio.sample_rate,
                                    settings.vad, partial_ok=split == "eval")
        splits[split]["_speakers"] = {s.speaker for s in samples}
        splits[split]["_attacks"] = {s.attack for s in samples if s.label}

    errors, findings = [], []
    names = [s for s in splits if "_speakers" in splits[s]]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared_spk = splits[a]["_speakers"] & splits[b]["_speakers"]
            if shared_spk:
                errors.append(f"{len(shared_spk)} speakers appear in both {a} and {b}")
            shared_att = splits[a]["_attacks"] & splits[b]["_attacks"]
            findings.append(f"attacks shared by {a} and {b}: {sorted(shared_att) or 'none'}")
    for split in names:
        rep = splits[split]
        del rep["_speakers"], rep["_attacks"]
        errors += [f"{split}: {e}" for e in rep.pop("errors", [])]
        findings += _shortcut_flags(split, rep)
    return {"dataset": dataset, "root": str(root), "ok": not errors, "errors": errors,
            "findings": findings, "splits": splits}


def write_report(report: dict, results_dir: str | Path) -> Path:
    out = Path(results_dir) / f"data_integrity_{report['dataset']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    return out
