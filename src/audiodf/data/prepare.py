"""Training-data preprocessing for the two branches.

Shared front end for every clip, identical for both classes and mirrored in serving:
    load -> fill digital silence -> VAD trim -> [train only: label-blind codec augmentation]
Then the branches split on purpose:
    SVM : growing-buffer prefixes (2, 4, 6, 8, 10 s) -> one 318-d vector each
    RCNN: fixed 2 s windows -> Log-Mel 64x200, read straight from FLAC (seek ~1.3 ms per window),
          so ASVspoof5's ~3.5M windows never need a ~90 GB on-disk Log-Mel cache.

Stage 1 (`build_index`) decodes each clip once to record its speech bounds; everything after
reads only what it needs.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from audiodf.config import Settings
from audiodf.data.audio import fill_digital_silence, load_audio
from audiodf.data.augment import CodecAugmenter
from audiodf.data.protocol import Sample, read_protocol
from audiodf.data.segmenter import pad_to_length, prefix_snapshots, stream_window_starts
from audiodf.data.vad import speech_bounds
from audiodf.features import rcnn_features, svm_features
from audiodf.features.rcnn_features import RcnnFeatureExtractor
from audiodf.features.svm_features import SvmFeatureExtractor

CODEC_MARGIN_S = 0.1  # extra audio around a window so codec warm-up doesn't land inside it


def _worker_init(_):
    torch.set_num_threads(1)


def _prep_dir(settings: Settings) -> Path:
    v = settings.vad
    tag = (f"prep_vad{v.threshold_db:g}_{v.frame_seconds:g}_{v.margin_seconds:g}"
           f"_svmv{svm_features.FEATURE_VERSION}_melv{rcnn_features.FEATURE_VERSION}")
    return Path(settings.paths.cache_dir) / tag


# ----------------------------------------------------------------------------- stage 1: index

class _IndexDataset(Dataset):
    def __init__(self, samples: list[Sample], settings: Settings):
        self.samples, self.settings = samples, settings

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        wave = load_audio(self.samples[i].path, self.settings.audio.sample_rate)
        bounds = speech_bounds(wave, self.settings.audio.sample_rate, self.settings.vad)
        start, end = bounds if bounds else (0, 0)
        return np.array([len(wave), start, end], dtype=np.int64)


class SplitIndex:
    """Per-utterance metadata for one split: labels, attacks, codec, and VAD speech bounds.

    Held as compact arrays, not Sample objects: DataLoader workers each receive a pickled copy, and
    182k Sample objects x 8 workers would not fit in memory next to the audio buffers."""

    FIELDS = ("label", "n_samples", "speech_start", "speech_end")
    # Set by with_renders(): clips whose audio is read from a realistic-codec copy (ffmpeg_codecs.py).
    rendered: np.ndarray | None = None
    render_codec: np.ndarray | None = None
    render_dir: str | None = None
    render_tag: str = ""

    def __init__(self, samples: list[Sample], arrays: dict, dataset: str, split: str):
        self.dataset, self.split = dataset, split
        for k in self.FIELDS:
            setattr(self, k, arrays[k])
        self.audio_dirs = [str(Path(samples[0].path).parent)]  # all clips of one split share a folder
        self.dir_of = np.zeros(len(samples), dtype=np.int16)
        self.utt_id = np.array([s.utt_id for s in samples])
        self.attack = np.array([s.attack for s in samples])
        self.codec = np.array([s.codec for s in samples])
        self.speaker = np.array([s.speaker for s in samples])
        self.source = np.full(len(samples), f"{dataset}:{split}")

    @classmethod
    def concat(cls, parts: list["SplitIndex"], name: str = "mixed") -> "SplitIndex":
        """Join splits (possibly from different datasets) into one training index. Attack IDs become
        'dataset:attack' because the same code names different systems in each dataset; bonafide stay '-'."""
        out = object.__new__(cls)
        out.dataset, out.split = name, "train"
        for k in cls.FIELDS + ("utt_id", "codec", "speaker", "source"):
            setattr(out, k, np.concatenate([getattr(p, k) for p in parts]))
        out.attack = np.concatenate([np.where(p.attack == "-", "-", np.char.add(f"{p.dataset}:", p.attack))
                                     for p in parts])
        out.audio_dirs, offsets = [], []
        for p in parts:
            offsets.append(len(out.audio_dirs))
            out.audio_dirs += p.audio_dirs
        out.dir_of = np.concatenate([p.dir_of + off for p, off in zip(parts, offsets)]).astype(np.int16)
        return out

    def subset(self, select: np.ndarray, split: str) -> "SplitIndex":
        """The clips picked by a boolean mask or index array, as a new index (audio folders shared)."""
        out = object.__new__(SplitIndex)
        out.dataset, out.split, out.audio_dirs = self.dataset, split, list(self.audio_dirs)
        for k in self.FIELDS + ("utt_id", "attack", "codec", "speaker", "source", "dir_of"):
            setattr(out, k, getattr(self, k)[select])
        if self.rendered is not None:
            out.rendered, out.render_codec = self.rendered[select], self.render_codec[select]
            out.render_dir, out.render_tag = self.render_dir, self.render_tag
        return out

    def with_renders(self, render_dir: str, rendered: np.ndarray, codec: np.ndarray, tag: str) -> "SplitIndex":
        """This index with `rendered` clips read from their codec copies in render_dir (same sample positions)."""
        out = self.subset(np.arange(len(self)), self.split)
        out.rendered, out.render_codec, out.render_dir, out.render_tag = rendered, codec, render_dir, tag
        return out

    def is_rendered(self, i: int) -> bool:
        return self.rendered is not None and bool(self.rendered[i])

    def __len__(self):
        return len(self.utt_id)

    def path(self, i: int) -> str:
        if self.is_rendered(i):
            return f"{self.render_dir}/{self.utt_id[i]}.flac"
        return f"{self.audio_dirs[self.dir_of[i]]}/{self.utt_id[i]}.flac"

    @property
    def has_speech(self) -> np.ndarray:
        return self.speech_end > self.speech_start


def build_index(settings: Settings, dataset: str, split: str, limit: int | None = None,
                workers: int = 8, log=print, available_only: bool = False) -> SplitIndex:
    samples = read_protocol(settings.dataset_root(dataset), split, limit, dataset, available_only)
    path = _prep_dir(settings) / f"index_{dataset}_{split}{f'_limit{limit}' if limit else ''}.npz"
    if path.exists():
        arrays = dict(np.load(path))
        if len(arrays["label"]) == len(samples):
            return SplitIndex(samples, arrays, dataset, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(_IndexDataset(samples, settings), batch_size=None, num_workers=workers,
                        worker_init_fn=_worker_init, prefetch_factor=16 if workers else None)
    rows, t0 = [], time.time()
    for i, row in enumerate(loader):
        rows.append(row.numpy())
        if (i + 1) % 20000 == 0:
            log(f"    index {dataset}/{split}: {i + 1}/{len(samples)} ({(i + 1) / (time.time() - t0):.0f} utt/s)")
    rows = np.stack(rows)
    arrays = {"label": np.array([s.label for s in samples], dtype=np.int8), "n_samples": rows[:, 0],
              "speech_start": rows[:, 1], "speech_end": rows[:, 2]}
    np.savez(path, **arrays)
    (path.with_suffix(".json")).write_text(json.dumps({"vad": asdict(settings.vad), "n": len(samples),
                                                       "no_speech": int((rows[:, 2] <= rows[:, 1]).sum())}))
    log(f"  indexed {dataset}/{split}: {len(samples)} utts in {time.time() - t0:.0f}s, "
        f"{int((rows[:, 2] <= rows[:, 1]).sum())} without speech")
    return SplitIndex(samples, arrays, dataset, split)


def _speech(idx: SplitIndex, i: int, max_samples: int | None = None) -> np.ndarray:
    """VAD-trimmed audio of utterance i (optionally only its first max_samples)."""
    start, end = int(idx.speech_start[i]), int(idx.speech_end[i])
    if max_samples:
        end = min(end, start + max_samples)
    with sf.SoundFile(idx.path(i)) as f:
        f.seek(start)
        return fill_digital_silence(f.read(end - start, dtype="float32", always_2d=False))


# ----------------------------------------------------------------------------- stage 2: SVM snapshots

def holdout_split(pool: SplitIndex, attacks, speaker_source: str, speaker_frac: float,
                  seed: int = 0) -> tuple[np.ndarray, np.ndarray, int]:
    """Masks (train, tune) for tuning on attacks the models never see.

    tune  = every clip of the held-out attacks + bonafide clips of held-out speakers, drawn as
            `speaker_frac` of the bonafide speakers in `speaker_source` (e.g. "asv5:dev");
    train = everything else, minus every clip of the held-out speakers (no voice shared with tune bonafide).
    Clips of held-out speakers under other attacks are in neither set."""
    missing = sorted(set(attacks) - set(pool.attack))
    if missing:
        raise ValueError(f"held-out attacks not in the training pool: {missing}")
    bona_spk = np.unique(pool.speaker[(pool.source == speaker_source) & (pool.label == 0)])
    if not len(bona_spk):
        raise ValueError(f"no bonafide speakers in {speaker_source!r} to hold out")
    held_spk = np.random.default_rng(seed).choice(bona_spk, max(1, round(speaker_frac * len(bona_spk))),
                                                  replace=False)
    held_attack = np.isin(pool.attack, list(attacks))
    held_voice = np.isin(pool.speaker, held_spk)
    tune = held_attack | (held_voice & (pool.label == 0))
    train = ~held_attack & ~held_voice
    return train, tune, len(held_spk)


def stratified_subset(idx: SplitIndex, n: int, seed: int = 0) -> np.ndarray:
    """Up to n utterances with speech: a third bonafide, the rest split evenly across attacks."""
    rng = np.random.default_rng(seed)
    ok = idx.has_speech
    groups = {a: np.nonzero(ok & (idx.attack == a))[0] for a in sorted(set(idx.attack))}
    bona = groups.pop("-", np.array([], dtype=int))
    take = [rng.choice(bona, min(len(bona), n // 3), replace=False)]
    per_attack = (n - len(take[0])) // max(len(groups), 1)
    take += [rng.choice(g, min(len(g), per_attack), replace=False) for g in groups.values()]
    return np.sort(np.concatenate(take))


class _SnapshotDataset(Dataset):
    def __init__(self, idx: SplitIndex, utts: np.ndarray, settings: Settings, augment_p: float, seed: int):
        self.idx, self.utts, self.settings, self.seed = idx, utts, settings, seed
        self.augment = CodecAugmenter(augment_p)
        self._fx = None

    def __len__(self):
        return len(self.utts)

    def __getitem__(self, j):
        s, i = self.settings, int(self.utts[j])
        if self._fx is None:
            self._fx = SvmFeatureExtractor(s.audio, s.svm_features)
        wave = _speech(self.idx, i, s.window_samples)
        if self.idx.is_rendered(i):  # already through a real codec: don't stack a simulated one on top
            codec = str(self.idx.render_codec[i])
        else:
            wave, codec = self.augment(wave, np.random.default_rng((self.seed, i)))  # one codec per call
        snaps = prefix_snapshots(wave, s.audio.sample_rate, s.data.svm_snapshot_seconds, s.segment_samples)
        feats = np.stack([self._fx.extract(x) for x in snaps]).astype(np.float32)
        secs = np.array([len(x) / s.audio.sample_rate for x in snaps], dtype=np.float32)
        return {"x": feats, "seconds": secs, "codec": codec}


def build_svm_snapshots(idx: SplitIndex, utts: np.ndarray, settings: Settings, augment_p: float,
                        tag: str, workers: int = 8, seed: int = 0, log=print) -> dict:
    """Feature vectors for every growing-buffer snapshot of the given utterances (cached)."""
    # Keyed by which clips (IDs), not their positions: subsets re-number clips, and a position-keyed cache
    # could hand back another clip set's features. Positions are re-mapped to this index on load.
    ids = idx.utt_id[utts]
    key = hashlib.md5("\n".join(ids).encode()).hexdigest()[:10]
    grid = "-".join(f"{x:g}" for x in settings.data.svm_snapshot_seconds)
    render = f"_{idx.render_tag}" if idx.rendered is not None else ""  # codec copies change the features
    path = _prep_dir(settings) / f"svm_{idx.dataset}_{tag}_{key}_g{grid}_aug{augment_p:g}_s{seed}{render}.npz"
    if path.exists():
        data = dict(np.load(path))
        pos = dict(zip(ids.tolist(), np.asarray(utts).tolist()))
        data["utt"] = np.array([pos[u] for u in data["utt_id"].tolist()], dtype=np.int64)
        data["label"] = idx.label[data["utt"]].astype(np.int64)
        return data
    loader = DataLoader(_SnapshotDataset(idx, utts, settings, augment_p, seed), batch_size=None,
                        num_workers=workers, worker_init_fn=_worker_init, prefetch_factor=8 if workers else None)
    xs, secs, utt_ids, codecs, t0 = [], [], [], [], time.time()
    for j, item in enumerate(loader):
        k = len(item["seconds"])
        xs.append(item["x"].numpy())
        secs.append(item["seconds"].numpy())
        utt_ids.append(np.full(k, utts[j], dtype=np.int64))
        codecs += [item["codec"]] * k
        if (j + 1) % 5000 == 0:
            log(f"    svm snapshots {idx.split}: {j + 1}/{len(utts)} utts ({(j + 1) / (time.time() - t0):.0f} utt/s)")
    data = {"x": np.concatenate(xs), "seconds": np.concatenate(secs), "utt": np.concatenate(utt_ids),
            "codec": np.array(codecs)}
    data["label"] = idx.label[data["utt"]].astype(np.int64)
    data["utt_id"] = idx.utt_id[data["utt"]]
    np.savez(path, **data)
    log(f"  svm snapshots {idx.dataset}/{idx.split}: {len(utts)} utts -> {len(data['x'])} vectors "
        f"in {time.time() - t0:.0f}s")
    return data


# ----------------------------------------------------------------------------- stage 3: RCNN windows

class RcnnWindowDataset(Dataset):
    """Item (utt, start) -> (Log-Mel window, label). Starts are absolute sample offsets in the file."""

    def __init__(self, idx: SplitIndex, settings: Settings, augment_p: float = 0.0, seed: int = 0):
        self.idx, self.settings, self.seed = idx, settings, seed
        self.augment = CodecAugmenter(augment_p)
        self.margin = int(CODEC_MARGIN_S * settings.audio.sample_rate) if augment_p > 0 else 0
        self._fx = None

    def __len__(self):
        return len(self.idx)

    def read_window(self, i: int, start: int) -> np.ndarray:
        s, seg = self.settings, self.settings.segment_samples
        lo = max(0, start - self.margin)
        with sf.SoundFile(self.idx.path(i)) as f:
            f.seek(lo)
            chunk = fill_digital_silence(f.read(start + seg + self.margin - lo, dtype="float32", always_2d=False))
        off = start - lo
        if len(chunk) < off + seg:  # speech shorter than a window: repeat-pad, as serving does
            chunk = np.concatenate([chunk[:off], pad_to_length(chunk[off:], seg)])
        if self.margin and not self.idx.is_rendered(i):  # real-codec copies get no simulated codec on top
            chunk, _ = self.augment(chunk, np.random.default_rng((self.seed, i, start)))
        return chunk[off:off + seg]

    def __getitem__(self, key):
        i, start = key
        s = self.settings
        if self._fx is None:
            self._fx = RcnnFeatureExtractor(s.audio, s.rcnn_features, s.segment_samples)
        mel = self._fx.extract(self.read_window(i, start)[None])[0]
        return torch.from_numpy(mel).unsqueeze(0), torch.tensor(float(self.idx.label[i]))


class WaveWindowDataset(RcnnWindowDataset):
    """The same windows (renders, codec augmentation, silence fill), as raw waveform for the WavLM branch."""

    def __getitem__(self, key):
        i, start = key
        wave = np.ascontiguousarray(self.read_window(i, start), dtype=np.float32)
        return torch.from_numpy(wave), torch.tensor(float(self.idx.label[i]))


class RandomWindowSampler(Sampler):
    """Each epoch: `per_utt` random 2 s windows per utterance, shuffled. Windows start within the first
    `horizon_samples` of speech (the 10 s a live call decides on), so training never sees material
    serving can't, and clip length stops carrying label information (ASV5 train bonafide clips
    run ~14 s vs ~11 s for spoof). A fixed count per utterance keeps long clips from dominating."""

    def __init__(self, idx: SplitIndex, seg_samples: int, per_utt: int, horizon_samples: int, seed: int = 0):
        self.idx, self.seg, self.per_utt, self.seed, self.epoch = idx, seg_samples, per_utt, seed, 0
        self.horizon = horizon_samples
        self.utts = np.nonzero(idx.has_speech)[0]

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return len(self.utts) * self.per_utt

    def __iter__(self):
        rng = np.random.default_rng((self.seed, self.epoch))
        utts = np.repeat(self.utts, self.per_utt)
        lo = self.idx.speech_start[utts]
        span = np.maximum(np.minimum(self.idx.speech_end[utts] - lo, self.horizon) - self.seg, 0)
        starts = lo + (rng.random(len(utts)) * (span + 1)).astype(np.int64)
        order = rng.permutation(len(utts))
        return iter(zip(utts[order].tolist(), starts[order].tolist()))


def stream_windows(idx: SplitIndex, utts: np.ndarray, settings: Settings) -> list[tuple[int, int]]:
    """The windows CallSession scores within the decision horizon (10 s from speech onset)."""
    s = settings
    out = []
    for i in utts:
        n = int(idx.speech_end[i] - idx.speech_start[i])
        starts = stream_window_starts(n, s.segment_samples, s.segment_hop_samples, s.window_samples) or [0]
        out += [(int(i), int(idx.speech_start[i]) + st) for st in starts]
    return out
