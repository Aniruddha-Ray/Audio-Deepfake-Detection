"""Realistic codec augmentation with real encoders (ffmpeg, bundled by the imageio-ffmpeg package).

Run 2 showed scores shifting with the codec: pooled ASV5-eval EER was 5.4 points worse than the
per-codec EERs. The in-process augmenter (augment.py) only covers Opus/MP3/Vorbis/G.711, and its simulated
Opus/MP3 did not transfer to eval's real codecs. This module encodes and decodes with the real codec
families of ASV5 eval (Opus, AMR, Speex, AAC, MP3, a Bluetooth-like SBC channel) plus G.722 and GSM.

An ffmpeg round trip costs ~0.1-0.3 s per 10 s clip (two processes), far too slow inside a data loader,
so copies are rendered once to disk. Selection is label-blind: whether a clip gets a copy, and which codec,
depends only on its ID and a seed, never on bonafide/spoof. Rendered copies keep the original sample
positions (codec delay is measured and removed), so VAD speech bounds in the index stay valid.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import correlate

CATALOGUE_VERSION = 1
_SR = 16000
RENDER_MARGIN_S = 0.25  # rendered beyond the 10 s decision horizon so window reads with codec margin stay inside


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


def _b(*rates):
    return tuple(("-b:a", r) for r in rates)


# Bitrates follow ASVspoof5.codec.config.csv where the codec matches an eval condition.
CODECS = (
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
BY_NAME = {c.name: c for c in CODECS}


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError("realistic codec augmentation needs ffmpeg: pip install imageio-ffmpeg") from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


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


def roundtrip(wave: np.ndarray, codec: FfCodec | str, variant: int = 0) -> np.ndarray:
    """Encode + decode 16 kHz mono float audio with a real codec; same length and timing as the input."""
    codec = BY_NAME[codec] if isinstance(codec, str) else codec
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


def choice(utt_id: str, seed: int) -> tuple[float, FfCodec, int]:
    """(u, codec, variant) for one clip, from its ID and the seed only (label-blind, reproducible).
    u decides membership: a clip gets a copy when u < frac."""
    h = hashlib.md5(f"{seed}:{utt_id}".encode()).digest()
    u = int.from_bytes(h[:4], "little") / 2 ** 32
    codec = CODECS[int.from_bytes(h[4:6], "little") % len(CODECS)]
    return u, codec, int.from_bytes(h[6:8], "little") % len(codec.variants)


def render_dir(cache_dir: str | Path, seed: int) -> Path:
    return Path(cache_dir) / f"render_ff{CATALOGUE_VERSION}" / f"s{seed}"


def render_copies(idx, utts: np.ndarray, settings, frac: float, seed: int, workers: int = 12, log=print):
    """Give a label-blind `frac` of `utts` a realistic-codec copy on disk; returns idx with those clips
    pointing at their copies. Resumable: existing copies are reused."""
    out_dir = render_dir(settings.paths.cache_dir, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    picks = {int(i): choice(str(idx.utt_id[i]), seed) for i in utts}
    chosen = [i for i, (u, _, _) in picks.items() if u < frac]
    todo = [i for i in chosen if not (out_dir / f"{idx.utt_id[i]}.flac").exists()]
    horizon = settings.window_samples + int(RENDER_MARGIN_S * settings.audio.sample_rate)

    need_gb = len(todo) * 0.22e-3  # ~0.22 MB per copy (~10 s of 16-bit FLAC)
    free_gb = shutil.disk_usage(out_dir).free / 1e9
    if need_gb + 15 > free_gb:
        raise RuntimeError(f"rendering {len(todo)} copies needs ~{need_gb:.0f} GB plus 15 GB headroom; "
                           f"only {free_gb:.0f} GB free")

    def render_one(i: int):
        _, codec, variant = picks[i]
        with sf.SoundFile(idx.path(i)) as f:
            wave = f.read(min(f.frames, int(idx.speech_start[i]) + horizon), dtype="float32", always_2d=False)
        sf.write(out_dir / f"{idx.utt_id[i]}.flac", roundtrip(wave, codec, variant), _SR, subtype="PCM_16")

    t0 = time.time()
    with ThreadPoolExecutor(workers) as pool:
        for k, _ in enumerate(pool.map(render_one, todo), 1):
            if k % 10000 == 0:
                log(f"    rendered {k}/{len(todo)} codec copies ({k / (time.time() - t0):.0f}/s)")
    if todo:
        log(f"  rendered {len(todo)} codec copies in {time.time() - t0:.0f}s ({len(chosen) - len(todo)} reused)")
    mask = np.zeros(len(idx), dtype=bool)
    mask[chosen] = True
    labels = np.full(len(idx), "-", dtype=object)
    for i in chosen:
        labels[i] = picks[i][1].name
    return idx.with_renders(str(out_dir), mask, labels.astype(str), f"ff{CATALOGUE_VERSION}s{seed}f{frac:g}")
