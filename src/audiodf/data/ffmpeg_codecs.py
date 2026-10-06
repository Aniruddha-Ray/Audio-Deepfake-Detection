"""Realistic codec augmentation with real encoders: ffmpeg (bundled by imageio-ffmpeg) and the neural codec EnCodec.

Run 2 showed scores shifting with the codec: pooled ASV5-eval EER was 5.4 points worse than the
per-codec EERs. The in-process augmenter (augment.py) only covers Opus/MP3/Vorbis/G.711, and its simulated
Opus/MP3 did not transfer to eval's real codecs. This module encodes and decodes with the real codec
families of ASV5 eval (Opus, AMR, Speex, AAC, MP3, a Bluetooth-like SBC channel) plus G.722 and GSM, and
(catalogue 2) the neural codec EnCodec, alone and after MP3 like eval conditions C04 and C07: run 4's
WavLM was at 15.8% / 19.3% EER there against 1.5-6.3% on every other condition.

A round trip costs ~0.1-0.3 s per 10 s clip, far too slow inside a data loader, so copies are rendered
once to disk. Selection is label-blind: whether a clip gets a copy, and which codec, depends only on its ID
and a seed, never on bonafide/spoof. Rendered copies keep the original sample positions (codec delay is
measured and removed), so VAD speech bounds in the index stay valid.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import os
import shutil
import subprocess
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import correlate, resample_poly

CATALOGUE_VERSION = 2  # 1: ffmpeg codecs only (runs 3-4); 2: + EnCodec
_SR = 16000
RENDER_MARGIN_S = 0.25  # rendered beyond the 10 s decision horizon so window reads with codec margin stay inside
# Share of copies given a neural codec: EnCodec conditions (C04, C07) are 18.7% of ASV5 eval's codec-processed clips.
NEURAL_SHARE = 0.18
ENCODEC_BATCH = 8  # 15.8 clips/s on the 4 GB laptop GPU; batch 16 is slower (9/s), 32 thrashes


@dataclass(frozen=True)
class FfCodec:
    name: str
    rate: int  # codec sample rate
    encoder: tuple[str, ...]
    variants: tuple[tuple[str, ...], ...]  # bitrate / quality settings, picked at random
    fmt: str  # muxer used when encoding
    dec_fmt: str  # demuxer used when decoding
    decoder: tuple[str, ...] = ()
    pre: tuple[str, ...] = ()  # filter applied before encoding


@dataclass(frozen=True)
class NeuralCodec:
    name: str
    variants: tuple[tuple, ...]  # (index of the MP3 variant applied first, or None; EnCodec kbps)


def _b(*rates):
    return tuple(("-b:a", r) for r in rates)


# Bitrates follow ASVspoof5.codec.config.csv where the codec matches an eval condition.
# The order of CLASSICAL must not change: catalogue-2 clips that keep a classical codec get exactly their
# catalogue-1 codec and variant, which lets their catalogue-1 copies be reused.
CLASSICAL = (
    FfCodec("opus_wb", 16000, ("-c:a", "libopus", "-application", "voip"), _b("6k", "12k", "18k", "24k", "30k"),
            "ogg", "ogg"),
    FfCodec("opus_nb", 8000, ("-c:a", "libopus", "-application", "voip"), _b("6k", "8k", "12k", "16k", "20k"),
            "ogg", "ogg"),
    FfCodec("amr_wb", 16000, ("-c:a", "libvo_amrwbenc",), _b("6.6k", "8.85k", "12.65k", "14.25k", "18.25k", "23.05k"),
            "amr", "amr"),
    FfCodec("amr_nb", 8000, ("-c:a", "libopencore_amrnb",), _b("4.75k", "5.9k", "6.7k", "7.95k", "10.2k", "12.2k"),
            "amr", "amr"),
    # ffmpeg's built-in Speex decoder mis-decodes wideband (double length); the libspeex decoder is correct.
    FfCodec("speex_wb", 16000, ("-c:a", "libspeex"), tuple(("-cbr_quality", q) for q in ("2", "4", "6", "8", "10")),
            "ogg", "ogg", decoder=("-c:a", "libspeex")),
    FfCodec("speex_nb", 8000, ("-c:a", "libspeex"), tuple(("-cbr_quality", q) for q in ("2", "4", "6", "8", "10")),
            "ogg", "ogg", decoder=("-c:a", "libspeex")),
    FfCodec("aac", 16000, ("-c:a", "aac"), _b("16k", "24k", "32k", "48k", "64k"), "adts", "aac"),
    FfCodec("mp3", 16000, ("-c:a", "libmp3lame"), _b("32k", "48k", "64k", "96k", "128k"), "mp3", "mp3"),
    FfCodec("bluetooth_sbc", 16000, ("-c:a", "sbc"), _b("64k", "96k", "128k"), "sbc", "sbc",
            pre=("-af", "highpass=f=150,lowpass=f=7000")),
    FfCodec("g722", 16000, ("-c:a", "g722"), ((),), "g722", "g722"),
    FfCodec("gsm", 8000, ("-c:a", "libgsm"), ((),), "gsm", "gsm"),
)
ENCODEC_KBPS = (1.5, 3.0, 6.0, 12.0, 24.0)  # eval C04 qualities 1-5
NEURAL = (
    NeuralCodec("encodec", tuple((None, k) for k in ENCODEC_KBPS)),
    # eval C07: MP3, then EnCodec; every pairing of our 5 MP3 bitrates with the 5 EnCodec rates
    NeuralCodec("mp3_encodec", tuple((m, k) for m in range(5) for k in ENCODEC_KBPS)),
)
# G.711 (landline / SIP). Raw PCM demuxers default to 44.1 kHz, so the sample rate is given for decoding. Not part of CODECS
# (catalogue 2 stays exactly as run 5 used it); catalogue 3, used with impairments (run 6), adds it to the classical codecs.
G711A = FfCodec("alaw", 8000, ("-c:a", "pcm_alaw"), ((),), "alaw", "alaw", decoder=("-ar", "8000", "-ac", "1"))
G711U = FfCodec("ulaw", 8000, ("-c:a", "pcm_mulaw"), ((),), "mulaw", "mulaw", decoder=("-ar", "8000", "-ac", "1"))
CLASSICAL3 = CLASSICAL + (G711A, G711U)
IMPAIR_VERSION = 3
CODECS = CLASSICAL + NEURAL
BY_NAME = {c.name: c for c in CODECS + (G711A, G711U)}


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError("realistic codec augmentation needs ffmpeg: pip install imageio-ffmpeg") from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


@functools.lru_cache(maxsize=None)
def _ffmpeg_codec_names(exe: str, kind: str) -> frozenset:
    """Names in `ffmpeg -encoders` / `-decoders` (lines like ' A....D libgsm   libgsm GSM')."""
    res = subprocess.run([exe, "-hide_banner", f"-{kind}"], capture_output=True, text=True)
    names = (line.split()[1] for line in res.stdout.splitlines() if len(line.split()) > 1)
    return frozenset(names)


def _encodec_installed() -> bool:
    return importlib.util.find_spec("encodec") is not None


def missing_codecs(names=None) -> list[str]:
    """Catalogue codecs this machine cannot render. The imageio-ffmpeg binaries differ by platform: the Windows
    build (7.1) has every ffmpeg codec, the Linux build (7.0.2) lacks libgsm. EnCodec needs `pip install encodec`."""
    exe = ffmpeg_exe()
    encoders, decoders = _ffmpeg_codec_names(exe, "encoders"), _ffmpeg_codec_names(exe, "decoders")
    out = []
    for codec in CODECS if names is None else [BY_NAME[n] for n in names]:
        if isinstance(codec, NeuralCodec):
            needs_mp3 = any(m is not None for m, _ in codec.variants)
            if not _encodec_installed() or (needs_mp3 and "libmp3lame" not in encoders):
                out.append(codec.name)
            continue
        enc = codec.encoder[codec.encoder.index("-c:a") + 1]
        dec = codec.decoder[codec.decoder.index("-c:a") + 1] if "-c:a" in codec.decoder else None
        if enc not in encoders or (dec and dec not in decoders):
            out.append(codec.name)
    return out


def _run(cmd: list[str], data: bytes) -> bytes:
    res = subprocess.run(cmd, input=data, capture_output=True)
    if res.returncode:
        raise RuntimeError(f"ffmpeg failed ({' '.join(cmd[5:])}): {res.stderr.decode(errors='ignore')[:200]}")
    return res.stdout


def align(out: np.ndarray, ref: np.ndarray, max_lag: int = 2048) -> np.ndarray:
    """Undo codec delay: shift `out` by the lag (within +-max_lag) that best matches `ref`, then crop/pad to
    len(ref). Codec delays measured on speech range from ~20 to ~1100 samples."""
    n = len(ref)
    seg = min(32000, n // 2)
    start = (n - seg) // 2
    if seg < 4000 or start < max_lag or len(out) < start + seg + max_lag:
        lag = 0
    else:
        corr = correlate(out[start - max_lag:start + seg + max_lag], ref[start:start + seg], mode="valid", method="fft")
        lag = int(np.argmax(corr)) - max_lag  # out[i + lag] ~ ref[i]
    shifted = out[lag:] if lag >= 0 else np.concatenate([np.zeros(-lag, dtype=out.dtype), out])
    return shifted[:n] if len(shifted) >= n else np.pad(shifted, (0, n - len(shifted)))


def _ffmpeg_roundtrip(wave: np.ndarray, codec: FfCodec, variant: int) -> np.ndarray:
    exe = ffmpeg_exe()
    pcm = (np.clip(np.asarray(wave, dtype=np.float32), -1.0, 1.0) * 32767).astype("<i2").tobytes()
    head = [exe, "-hide_banner", "-loglevel", "error"]
    encoded = _run(head + ["-f", "s16le", "-ar", str(_SR), "-ac", "1", "-i", "pipe:0", *codec.pre,
                           "-ar", str(codec.rate), "-ac", "1", *codec.encoder, *codec.variants[variant],
                           "-f", codec.fmt, "pipe:1"], pcm)
    decoded = _run(head + ["-f", codec.dec_fmt, *codec.decoder, "-i", "pipe:0",
                           "-f", "s16le", "-ar", str(_SR), "-ac", "1", "pipe:1"], encoded)
    out = np.frombuffer(decoded, dtype="<i2").astype(np.float32) / 32768.0
    return align(out, np.asarray(wave, dtype=np.float32))


_ENCODEC: dict = {}
_ENCODEC_LOCK = threading.Lock()  # one model shared by render threads; set_target_bandwidth mutates it


def encodec_roundtrip(waves: list[np.ndarray], kbps: float) -> list[np.ndarray]:
    """EnCodec 24 kHz round trip of 16 kHz waves, batched (shorter clips zero-padded at the end); same length and
    timing as each input. Output is deterministic for a given batch, but not sample-identical across batch
    compositions: batch size changes the GPU arithmetic slightly and the residual quantiser can pick other codes
    (same coding quality, measured SNR within 0.1 dB). A re-render after an interruption may therefore differ in
    detail; selection, codec and bitrate per clip never change."""
    import torch

    with _ENCODEC_LOCK:
        if "model" not in _ENCODEC:
            from encodec import EncodecModel

            device = "cuda" if torch.cuda.is_available() else "cpu"
            _ENCODEC["model"] = EncodecModel.encodec_model_24khz().to(device).eval()
        model = _ENCODEC["model"]
        device = next(model.parameters()).device
        x24 = [resample_poly(np.asarray(w, dtype=np.float32), 3, 2).astype(np.float32) for w in waves]
        batch = torch.zeros(len(x24), 1, max(len(x) for x in x24), device=device)
        for k, x in enumerate(x24):
            batch[k, 0, :len(x)] = torch.from_numpy(x)
        model.set_target_bandwidth(kbps)
        with torch.no_grad():
            y = model.decode(model.encode(batch))[:, 0].float().cpu().numpy()
    out = []
    for k, w in enumerate(waves):
        y16 = resample_poly(y[k, :len(x24[k])], 2, 3).astype(np.float32)
        out.append(np.clip(align(y16, np.asarray(w, dtype=np.float32)), -1.0, 1.0 - 2 ** -15))
    return out


def release_encodec() -> None:
    """Drop the EnCodec model and give its GPU memory back."""
    with _ENCODEC_LOCK:
        if _ENCODEC.pop("model", None) is not None:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def roundtrip(wave: np.ndarray, codec: FfCodec | NeuralCodec | str, variant: int = 0) -> np.ndarray:
    """Encode + decode 16 kHz mono float audio with a real codec; same length and timing as the input."""
    codec = BY_NAME[codec] if isinstance(codec, str) else codec
    if isinstance(codec, FfCodec):
        return _ffmpeg_roundtrip(wave, codec, variant)
    mp3, kbps = codec.variants[variant]
    pre = _ffmpeg_roundtrip(wave, BY_NAME["mp3"], mp3) if mp3 is not None else wave
    return encodec_roundtrip([pre], kbps)[0]


def choice(utt_id: str, seed: int, neural_share: float = NEURAL_SHARE,
           classical: tuple = CLASSICAL) -> tuple[float, FfCodec | NeuralCodec, int]:
    """(u, codec, variant) for one clip, from its ID and the seed only (label-blind, reproducible).
    u decides membership: a clip gets a copy when u < frac. A NEURAL_SHARE of clips gets a neural codec (decided
    by hash bytes catalogue 1 never used); every other clip gets exactly its catalogue-1 codec and variant."""
    h = hashlib.md5(f"{seed}:{utt_id}".encode()).digest()
    u = int.from_bytes(h[:4], "little") / 2 ** 32
    if int.from_bytes(h[8:10], "little") / 2 ** 16 < neural_share:
        codec = NEURAL[h[10] % len(NEURAL)]
        return u, codec, int.from_bytes(h[11:13], "little") % len(codec.variants)
    codec = classical[int.from_bytes(h[4:6], "little") % len(classical)]
    return u, codec, int.from_bytes(h[6:8], "little") % len(codec.variants)


def render_dir(cache_dir: str | Path, seed: int, version: int = CATALOGUE_VERSION, tag: str = "") -> Path:
    """Where copies are kept. With impairments the folder carries a hash of their configuration, so copies made with
    other noise settings are never reused."""
    return Path(cache_dir) / f"render_ff{version}{'_' + tag if tag else ''}" / f"s{seed}"


def _adopt_catalogue1(todo: list, picks: dict, idx, out_dir: Path, cache_dir, seed: int) -> list:
    """Hard-link catalogue-1 copies of clips whose codec and variant are unchanged (every classical pick);
    returns the clips still to render."""
    old_dir = render_dir(cache_dir, seed, 1)
    if not old_dir.is_dir():
        return todo
    left = []
    for i in todo:
        src = old_dir / f"{idx.utt_id[i]}.flac"
        if isinstance(picks[i][1], FfCodec) and src.exists():
            dst = out_dir / src.name
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        else:
            left.append(i)
    return left


def render_copies(idx, utts: np.ndarray, settings, frac: float, seed: int, workers: int = 12, log=print,
                  impair=None):
    """Give a label-blind `frac` of `utts` a realistic-codec copy on disk; returns idx with those clips
    pointing at their copies. Resumable: existing copies are reused.

    impair: an `impairments.ImpairKit` adds room echo and noise **before** the codec (as a microphone and room come before
    the codec in a call) and switches to catalogue 3 (G.711 added, the kit's neural share). Without it: catalogue 2."""
    version = IMPAIR_VERSION if impair else CATALOGUE_VERSION
    out_dir = render_dir(settings.paths.cache_dir, seed, version, impair.tag if impair else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    pick = (lambda u: choice(u, seed, impair.neural_share, CLASSICAL3)) if impair else (lambda u: choice(u, seed))
    picks = {int(i): pick(str(idx.utt_id[i])) for i in utts}
    chosen = [i for i, (u, _, _) in picks.items() if u < frac]
    todo = [i for i in chosen if not (out_dir / f"{idx.utt_id[i]}.flac").exists()]
    if not impair:  # catalogue-2 picks equal catalogue-1 picks for classical codecs: reuse those copies
        adopted = len(todo)
        todo = _adopt_catalogue1(todo, picks, idx, out_dir, settings.paths.cache_dir, seed)
        adopted -= len(todo)
        if adopted:
            log(f"  linked {adopted} unchanged copies from codec catalogue 1")
    horizon = settings.window_samples + int(RENDER_MARGIN_S * settings.audio.sample_rate)

    need_gb = len(todo) * 0.22e-3  # ~0.22 MB per copy (~10 s of 16-bit FLAC)
    free_gb = shutil.disk_usage(out_dir).free / 1e9
    if need_gb + 15 > free_gb:
        raise RuntimeError(f"rendering {len(todo)} copies needs ~{need_gb:.0f} GB plus 15 GB headroom; "
                           f"only {free_gb:.0f} GB free")
    # Codec choice is fixed by clip ID and seed, so a build missing a codec cannot simply skip it: the copies
    # would differ from every other machine's. Refuse instead (cached copies need no encoder and are fine).
    missing = missing_codecs(sorted({picks[i][1].name for i in todo})) if todo else []
    if missing:
        raise RuntimeError(f"this machine cannot render {', '.join(missing)}: install an ffmpeg build that has them "
                           f"(point IMAGEIO_FFMPEG_EXE at it) and, for EnCodec, `pip install encodec`")

    def read(i: int) -> np.ndarray:
        with sf.SoundFile(idx.path(i)) as f:
            wave = f.read(min(f.frames, int(idx.speech_start[i]) + horizon), dtype="float32", always_2d=False)
        if impair:  # room echo, then noise: both before the codec
            wave = impair.apply(wave, str(idx.utt_id[i]), seed, str(idx.speaker[i]))
        return wave

    def write(i: int, wave: np.ndarray) -> None:
        sf.write(out_dir / f"{idx.utt_id[i]}.flac", wave, _SR, subtype="PCM_16")

    def render_classical(i: int):
        _, codec, variant = picks[i]
        write(i, _ffmpeg_roundtrip(read(i), codec, variant))

    def before_encodec(i: int):
        """Read, and apply the MP3 stage for mp3_encodec (ffmpeg, so it runs in the thread pool)."""
        _, codec, variant = picks[i]
        mp3, kbps = codec.variants[variant]
        wave = read(i)
        return i, kbps, (_ffmpeg_roundtrip(wave, BY_NAME["mp3"], mp3) if mp3 is not None else wave)

    classical = [i for i in todo if isinstance(picks[i][1], FfCodec)]
    neural = [i for i in todo if isinstance(picks[i][1], NeuralCodec)]
    t0 = time.time()
    with ThreadPoolExecutor(workers) as pool:
        for k, _ in enumerate(pool.map(render_classical, classical), 1):
            if k % 10000 == 0:
                log(f"    rendered {k}/{len(classical)} ffmpeg copies ({k / (time.time() - t0):.0f}/s)")
        done, t1 = 0, time.time()
        for c in range(0, len(neural), 512):  # chunks bound the memory held between the MP3 and EnCodec stages
            by_kbps = defaultdict(list)
            for i, kbps, wave in pool.map(before_encodec, neural[c:c + 512]):
                by_kbps[kbps].append((i, wave))
            for kbps, items in by_kbps.items():
                for b in range(0, len(items), ENCODEC_BATCH):
                    batch = items[b:b + ENCODEC_BATCH]
                    for (i, _), wave in zip(batch, encodec_roundtrip([w for _, w in batch], kbps)):
                        write(i, wave)
            done += len(neural[c:c + 512])
            if done % 5120 < 512:
                log(f"    rendered {done}/{len(neural)} EnCodec copies ({done / (time.time() - t1):.0f}/s)")
    if neural:
        release_encodec()  # training needs the GPU memory (WavLM uses ~3.4 of 4.3 GB)
    if todo:
        log(f"  rendered {len(classical)} ffmpeg + {len(neural)} EnCodec copies in {time.time() - t0:.0f}s "
            f"({len(chosen) - len(todo)} reused)" + (f"; impairments {impair.tag}" if impair else ""))
    mask = np.zeros(len(idx), dtype=bool)
    mask[chosen] = True
    labels = np.full(len(idx), "-", dtype=object)
    for i in chosen:
        labels[i] = picks[i][1].name
    return idx.with_renders(str(out_dir), mask, labels.astype(str),
                            f"ff{version}{'_' + impair.tag if impair else ''}s{seed}f{frac:g}")
