"""Denoise a folder of simulated calls with ffmpeg's `afftdn` (FFT denoiser), to test whether a denoising front end helps.

The test applies the filter to every call (a live system cannot know which calls are noisy) and keeps the protocol
unchanged, so the denoised set can be scored with the same model and compared clip by clip with the original. It also
times the filter. Offline whole-file denoising is optimistic compared with streaming chunks (noise tracking needs warm-up).
"""

from __future__ import annotations

import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import soundfile as sf

from audiodf.data.ffmpeg_codecs import ffmpeg_exe
from audiodf.data.voip_sim import AUDIO_DIR, PROTOCOL_FILE


def afftdn_filter(nr: float = 12.0, nf: float = -45.0, track: bool = True) -> str:
    """ffmpeg filter string: `nr` dB of noise reduction, noise floor `nf` dB, noise tracking on or off."""
    return f"afftdn=nr={nr:g}:nf={nf:g}:tn={int(track)}"


def denoise_calls(src_root: str | Path, dst_root: str | Path, filter_graph: str, workers: int = 8, log=print) -> dict:
    """Denoise every call of the set in src_root into dst_root (same file names and protocol). Resumable."""
    src, dst = Path(src_root), Path(dst_root)
    (dst / AUDIO_DIR).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src / PROTOCOL_FILE, dst / PROTOCOL_FILE)
    files = sorted((src / AUDIO_DIR).glob("*.flac"))
    exe = ffmpeg_exe()

    def one(path: Path) -> tuple[float, float]:
        target = dst / AUDIO_DIR / path.name
        seconds = sf.info(path).duration
        if target.exists() and target.stat().st_size:
            return seconds, 0.0
        t0 = time.perf_counter()
        tmp = target.with_suffix(".part.flac")
        res = subprocess.run([exe, "-y", "-hide_banner", "-loglevel", "error", "-i", str(path), "-af", filter_graph,
                              "-ar", "16000", "-ac", "1", "-c:a", "flac", "-sample_fmt", "s16", str(tmp)],
                             capture_output=True)
        if res.returncode:
            raise RuntimeError(f"ffmpeg failed on {path.name}: {res.stderr.decode(errors='ignore')[:200]}")
        tmp.replace(target)
        return seconds, time.perf_counter() - t0

    t0, audio_s, filter_s, done = time.time(), 0.0, 0.0, 0
    with ThreadPoolExecutor(workers) as pool:
        for k, (sec, took) in enumerate(pool.map(one, files), 1):
            audio_s += sec
            if took:
                filter_s += took
                done += 1
            if k % 2000 == 0:
                log(f"    denoised {k}/{len(files)} calls ({k / (time.time() - t0):.0f}/s)")
    out = {"filter": filter_graph, "calls": len(files), "freshly_denoised": done, "audio_seconds": round(audio_s, 1),
           "wall_seconds": round(time.time() - t0, 1),
           # per-file ffmpeg time includes process start-up (~tens of ms); an in-process filter would be cheaper
           "ms_per_10s_audio_incl_process_startup": round(1000 * filter_s / max(done, 1) / (audio_s / len(files) / 10), 1)
           if done else None}
    log(f"denoised {len(files)} calls with {filter_graph} in {out['wall_seconds']}s -> {dst}")
    return out
