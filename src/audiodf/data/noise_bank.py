"""Real noise recordings and room impulse responses (RIRs) for noise- and echo-robust training and for held-out tests.

Corpora, all 16 kHz mono (`audiodf prepare-noise` unpacks and resamples the parquet ones):
  training noise   musan (OpenSLR 17, noise folder: free-sound and sound-bible), pointsource (RIRS_NOISES point-source
                   noises), demand (DEMAND subset: kitchen, street, cafeteria...)
  training echo    sim_rir (RIRS_NOISES simulated room impulse responses), mit_rir (MIT IR Survey: 270 measured everyday
                   spaces at 1.5 m, CC-BY 4.0, Traer and McDermott 2016; run 8)
  held-out tests   esc50 (environmental sounds), real_rir (RIRS_NOISES real, measured RIRs)
Keeping the test corpora out of training means a noise-robust model is judged on noise and rooms it never saw.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR = 16000
TRAIN_NOISE = ("musan", "pointsource", "demand")
TEST_NOISE = ("esc50",)
TRAIN_RIR, TEST_RIR = ("sim_rir", "mit_rir"), ("real_rir",)


def _wavs(folder: Path, pattern: str = "*.wav") -> list[str]:
    return sorted(str(p) for p in folder.rglob(pattern)) if folder.is_dir() else []


def prepare_parquet_noise(parquet_files, out_dir: str | Path, log=print) -> int:
    """Unpack a Hugging Face audio parquet (columns: audio{bytes}) into 16 kHz mono FLAC files. Resumable."""
    import pyarrow.parquet as pq

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for path in parquet_files:
        pf = pq.ParquetFile(path)
        col = next(c for c in pf.schema_arrow.names if c in ("audio", "wav", "waveform"))
        for g in range(pf.num_row_groups):
            for k, item in enumerate(pf.read_row_group(g, columns=[col]).column(col).to_pylist()):
                target = out_dir / f"{Path(path).stem}_{g:03d}_{k:04d}.flac"
                if not target.exists():
                    wave, sr = sf.read(io.BytesIO(item["bytes"]), dtype="float32", always_2d=True)
                    wave = wave.mean(axis=1)
                    if sr != SR:
                        wave = resample_poly(wave, SR, sr).astype(np.float32)
                    sf.write(target, wave, SR, subtype="PCM_16")
                n += 1
    log(f"unpacked {n} clips to {out_dir}")
    return n


class NoiseBank:
    """File lists per corpus; noise segments and RIRs are read from disk on demand (nothing big is held in memory)."""

    def __init__(self, files: dict[str, list[str]]):
        self.files = {c: list(v) for c, v in files.items() if v}

    @classmethod
    def scan(cls, root: str | Path) -> "NoiseBank":
        root = Path(root)
        rirs = root / "rirs" / "RIRS_NOISES"
        return cls({
            "musan": _wavs(root / "musan" / "musan" / "noise"),
            "pointsource": _wavs(rirs / "pointsource_noises"),
            "demand": _wavs(root / "demand_wav", "*.flac"),
            "esc50": _wavs(root / "esc50_wav", "*.flac"),
            "sim_rir": _wavs(rirs / "simulated_rirs"),
            "mit_rir": _wavs(root / "mit_rir", "*.flac"),
            "real_rir": [p for p in _wavs(rirs / "real_rirs_isotropic_noises") if "_rir_" in Path(p).name],
        })

    def corpora(self) -> dict[str, int]:
        return {c: len(v) for c, v in self.files.items()}

    def require(self, *corpora: str) -> None:
        missing = [c for c in corpora if c not in self.files]
        if missing:
            raise FileNotFoundError(f"noise corpora not found: {missing} (have {sorted(self.files)}); "
                                    f"download them and run `audiodf prepare-noise`")

    def noise(self, n: int, rng: np.random.Generator, corpora: tuple[str, ...]) -> np.ndarray:
        """Unit-RMS noise of n samples: a random segment of a random file from one random corpus of `corpora`
        (a short file is repeated). The corpus is picked first, so a small corpus is not swamped by a large one."""
        corpus = corpora[int(rng.integers(len(corpora)))]
        path = self.files[corpus][int(rng.integers(len(self.files[corpus])))]
        with sf.SoundFile(path) as f:
            if f.samplerate != SR:
                raise ValueError(f"{path}: expected {SR} Hz, got {f.samplerate}")
            if f.frames > n:
                f.seek(int(rng.integers(0, f.frames - n)))
            seg = f.read(min(n, f.frames), dtype="float32", always_2d=True).mean(axis=1)
        reps = -(-n // max(len(seg), 1))
        seg = np.tile(seg, reps)[:n] if len(seg) else np.zeros(n, dtype=np.float32)
        rms = float(np.sqrt((seg.astype(np.float64) ** 2).mean()))
        return seg / rms if rms > 1e-9 else seg  # a silent stretch stays silent

    def rir(self, rng: np.random.Generator, corpora: tuple[str, ...], max_seconds: float = 1.0, stretch_p: float = 0.0,
            stretch_range: tuple = (0.8, 2.0), stretch_corpora: tuple = ()) -> np.ndarray:
        """One room impulse response from a random file (the corpus is picked first, so a small measured corpus is not
        swamped by a large simulated one), peak scaled to 1 and cut at `max_seconds`. With probability `stretch_p` an RIR of
        a corpus in `stretch_corpora` is stretched in time by a log-uniform factor in `stretch_range`: the room gets that
        much longer (or shorter) reverberation, which widens a small measured set. The random stream is untouched
        when stretching is off, so earlier recipes render exactly as before."""
        corpus = corpora[int(rng.integers(len(corpora)))]
        path = self.files[corpus][int(rng.integers(len(self.files[corpus])))]
        h, sr = sf.read(path, dtype="float32", always_2d=True)
        h = h.mean(axis=1)
        if sr != SR:
            h = resample_poly(h, SR, sr).astype(np.float32)
        if stretch_p > 0 and corpus in stretch_corpora and rng.random() < stretch_p:
            factor = float(np.exp(rng.uniform(np.log(stretch_range[0]), np.log(stretch_range[1]))))
            h = resample_poly(h, int(round(factor * 100)), 100).astype(np.float32)
        h = h[: int(max_seconds * SR)]
        peak = float(np.abs(h).max())
        return h / peak if peak > 0 else h
