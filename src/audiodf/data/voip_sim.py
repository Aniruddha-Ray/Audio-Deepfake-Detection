"""Simulated VoIP calls built from clean speech clips, for a live-call test without recordings.

A real call is: speech at a microphone (+ room noise) -> codec -> network (packet loss) -> the detector. This module
builds that chain in software from ASV5 eval clips that carry no codec of their own (a call starts from clean speech;
eval's own codec copies would stack a second, unrealistic codec): background noise is added first (what a
microphone picks up), the result goes through a real VoIP codec (ffmpeg: Opus, AMR, G.722, G.711, GSM), and 20 ms
packets are then dropped with a crude concealment. Not simulated: echo, jitter buffers, real phones and networks.

Everything about a call is chosen from the clip ID and a seed only, never from bonafide/spoof, so genuine and fake
calls see the same conditions and no channel shortcut exists in the test. The profile mix and noise levels are my
assumptions about typical calls, written down here so they can be changed; the protocol file lists them per call.
"""

from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from audiodf.data.ffmpeg_codecs import BY_NAME, G711A, G711U, FfCodec, _ffmpeg_roundtrip
from audiodf.data.vad import frame_db

SR = 16000
PACKET = 320  # 20 ms, the usual VoIP packet
PROTOCOL_FILE, AUDIO_DIR = "ASV5.eval.calls.tsv", "flac"
CALL_COLUMNS = ("utt_id", "speaker", "codec", "noise", "snr_bin", "snr_db", "loss", "attack", "key")

@dataclass(frozen=True)
class Profile:
    name: str
    weight: float
    codec: FfCodec | None  # None: no codec (lossless VoIP leg)
    variants: tuple[int, ...]  # indices into codec.variants (bitrates typical for calls)


PROFILES = (
    Profile("opus_wb", 0.30, BY_NAME["opus_wb"], (2, 3, 4)),  # 18-30 kbps: WhatsApp / Zoom / Teams style
    Profile("opus_nb", 0.08, BY_NAME["opus_nb"], (2, 3, 4)),
    Profile("amr_wb", 0.17, BY_NAME["amr_wb"], (2, 3, 4, 5)),  # mobile HD voice
    Profile("amr_nb", 0.10, BY_NAME["amr_nb"], (1, 2, 3, 4, 5)),
    Profile("g722", 0.10, BY_NAME["g722"], (0,)),  # SIP wideband
    Profile("alaw", 0.08, G711A, (0,)),  # landline G.711
    Profile("ulaw", 0.07, G711U, (0,)),
    Profile("gsm", 0.05, BY_NAME["gsm"], (0,)),
    Profile("none", 0.05, None, (0,)),
)
assert abs(sum(p.weight for p in PROFILES) - 1) < 1e-9
NOISE_P, SNR_RANGE = 0.75, (15.0, 35.0)  # share of calls with added noise; SNR in dB against the speech level
NOISES = ("white", "pink", "brown")
LOSSES = (1, 3, 5)  # percent of packets lost, in the half of calls that have loss
SNR_BINS = ((5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 35))  # 15-35 labels as in the first call set


@dataclass(frozen=True)
class CallPlan:
    profile: Profile
    variant: int
    noise: str  # "none" or one of NOISES
    snr_db: float  # nan when there is no noise
    loss: int  # percent of 20 ms packets lost (0 = none)
    seed: int  # for the noise and loss draws

    @property
    def snr_bin(self) -> str:
        if self.noise == "none":
            return "none"
        lo, hi = next(b for b in SNR_BINS if self.snr_db < b[1] or b is SNR_BINS[-1])
        return f"{lo}-{hi}"


def plan(utt_id: str, seed: int = 0, noises: tuple = NOISES, snr_range: tuple = SNR_RANGE) -> CallPlan:
    """The call's conditions, from the clip ID and seed only (label-blind, reproducible). `noises` / `snr_range` choose
    the noise kinds and levels of the noisy calls (defaults: the first call set)."""
    h = hashlib.md5(f"call{seed}:{utt_id}".encode()).digest()
    u = lambda i: int.from_bytes(h[i:i + 2], "little") / 65536  # noqa: E731
    x, acc = u(0), 0.0
    profile = PROFILES[-1]
    for p in PROFILES:
        acc += p.weight
        if x < acc:
            profile = p
            break
    noisy = u(3) < NOISE_P
    return CallPlan(profile, profile.variants[h[2] % len(profile.variants)],
                    noises[h[5] % len(noises)] if noisy else "none",
                    snr_range[0] + u(6) * (snr_range[1] - snr_range[0]) if noisy else float("nan"),
                    0 if u(8) < 0.5 else LOSSES[h[10] % len(LOSSES)], int.from_bytes(h[11:16], "little"))


def _coloured(n: int, kind: str, rng: np.random.Generator) -> np.ndarray:
    """Unit-RMS noise: white, pink (power ~ 1/f) or brown (power ~ 1/f^2)."""
    w = rng.standard_normal(n)
    if kind != "white":
        spec = np.fft.rfft(w)
        f = np.arange(len(spec), dtype=float)
        f[0] = 1.0
        w = np.fft.irfft(spec / (np.sqrt(f) if kind == "pink" else f), n)
    return w / (np.sqrt((w ** 2).mean()) + 1e-12)


def speech_rms(x: np.ndarray, threshold_db: float = -45.0) -> float:
    """RMS over the frames above the VAD threshold (the speech), or of everything when no frame is that loud."""
    db = frame_db(x, 400)
    loud = db > threshold_db
    return float(np.sqrt((10 ** (db[loud] / 10)).mean())) if loud.any() else float(np.sqrt((x ** 2).mean()))


class BabblePool:
    """Background talkers: genuine clean clips of other speakers, mixed 4-6 at a time into a babble of any length.

    Built from genuine clips only (people talking in the background) that are not call sources, and never the caller's
    own speaker. A noise kind "babble" in a call plan draws from this pool."""

    def __init__(self, paths, speakers, starts, ends):
        self.paths, self.speakers = list(paths), np.array(speakers)
        self.starts, self.ends = list(starts), list(ends)

    @classmethod
    def from_index(cls, idx, exclude: np.ndarray | None = None) -> "BabblePool":
        ok = (idx.label == 0) & (idx.codec == "-") & idx.has_speech
        if exclude is not None:
            ok[np.asarray(exclude, dtype=int)] = False
        keep = np.nonzero(ok)[0]
        if len(set(idx.speaker[keep])) < 8:
            raise ValueError("babble needs genuine clips of at least 8 different speakers")
        return cls([idx.path(int(i)) for i in keep], idx.speaker[keep], idx.speech_start[keep], idx.speech_end[keep])

    def _segment(self, k: int, n: int, rng: np.random.Generator) -> np.ndarray:
        start, end = int(self.starts[k]), int(self.ends[k])
        with sf.SoundFile(self.paths[k]) as f:
            f.seek(start + int(rng.integers(0, max(1, end - start - n))) if end - start > n else start)
            seg = f.read(min(n, end - start), dtype="float32", always_2d=False)
        while len(seg) < n:  # a short clip is repeated
            seg = np.concatenate([seg, seg])[:n] if len(seg) else np.zeros(n, dtype=np.float32)
        return seg[:n]

    def sample(self, n: int, rng: np.random.Generator, avoid_speaker: str = "") -> np.ndarray:
        """Unit-RMS babble of n samples: 4-6 other speakers' speech, each at the same level, summed."""
        pick = rng.choice(np.nonzero(self.speakers != avoid_speaker)[0], int(rng.integers(4, 7)), replace=False)
        mix = np.zeros(n)
        for k in pick:
            seg = self._segment(int(k), n, rng).astype(np.float64)
            mix += seg / (speech_rms(seg.astype(np.float32)) + 1e-9)
        return mix / (np.sqrt((mix ** 2).mean()) + 1e-12)


def add_noise(x: np.ndarray, kind: str, snr_db: float, rng: np.random.Generator, babble: BabblePool | None = None,
              avoid_speaker: str = "") -> np.ndarray:
    """Background noise at `snr_db` below the speech level (before encoding, as a microphone would pick it up)."""
    if kind == "babble":
        if babble is None:
            raise ValueError("noise kind 'babble' needs a BabblePool")
        noise = babble.sample(len(x), rng, avoid_speaker)
    else:
        noise = _coloured(len(x), kind, rng)
    return x + (noise * speech_rms(x) / 10 ** (snr_db / 20)).astype(np.float32)


def drop_packets(x: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Lose each 20 ms packet with probability `rate`. Crude concealment: a lost packet repeats the previous output
    packet at 0.6 gain (so bursts fade out); at the very start it is silence."""
    y = x.copy()
    lost = rng.random(len(x) // PACKET) < rate
    for k in np.nonzero(lost)[0]:
        sl = slice(k * PACKET, (k + 1) * PACKET)
        y[sl] = 0.6 * y[sl.start - PACKET:sl.stop - PACKET] if k else 0.0
    return y


def make_call(wave: np.ndarray, call: CallPlan, babble: BabblePool | None = None, speaker: str = "") -> np.ndarray:
    """The received 16 kHz call audio for clean speech `wave`: noise -> codec -> packet loss."""
    rng = np.random.default_rng(call.seed)
    x = np.asarray(wave, dtype=np.float32)
    if call.noise != "none":
        x = add_noise(x, call.noise, call.snr_db, rng, babble, speaker)
    if call.profile.codec is not None:
        x = _ffmpeg_roundtrip(x, call.profile.codec, call.variant)
    if call.loss:
        x = drop_packets(x, call.loss / 100, rng)
    return np.clip(x, -1.0, 1.0 - 2 ** -15)


def speaker_half(speakers, salt: str = "cal") -> np.ndarray:
    """True for speakers in half A of a fixed split by a hash of the speaker name: the same in every call set and run, so
    thresholds set on half A can be reported on half B of any call set."""
    return np.array([hashlib.md5(f"{salt}:{s}".encode()).digest()[0] % 2 == 0 for s in speakers], dtype=bool)


def select_sources(idx, n_genuine: int, n_per_attack: int, seed: int = 0, allowed: np.ndarray | None = None) -> np.ndarray:
    """Clean source clips (no codec of their own, with speech): n_genuine bonafide (0 = all) and n_per_attack per attack,
    only among `allowed` clips if given. Drawn at random within those groups; scores play no part."""
    rng = np.random.default_rng(seed)
    clean = (idx.codec == "-") & idx.has_speech
    if allowed is not None:
        clean = clean & allowed
    groups = {"bonafide": np.nonzero(clean & (idx.label == 0))[0]}
    for a in sorted(set(idx.attack[clean & (idx.label == 1)])):
        groups[a] = np.nonzero(clean & (idx.attack == a))[0]
    want = {g: (n_genuine or len(v)) if g == "bonafide" else n_per_attack for g, v in groups.items()}
    short = {g: len(v) for g, v in groups.items() if len(v) < want[g]}
    if short:
        raise ValueError(f"not enough clean source clips for {short}; lower n_genuine / n_per_attack")
    return np.sort(np.concatenate([rng.choice(v, want[g], replace=False) for g, v in groups.items()]))


def render_calls(idx, sources: np.ndarray, settings, out_root: str | Path, seed: int = 0, workers: int = 8,
                 log=print, noises: tuple = NOISES, snr_range: tuple = SNR_RANGE,
                 babble: BabblePool | None = None) -> int:
    """Write the call audio of the `sources` clips of `idx` (a SplitIndex of ASV5 eval) to out_root/flac/ and the
    protocol to out_root/ASV5.eval.calls.tsv. Resumable (existing files are kept). Returns the number of calls.
    `noises` / `snr_range` choose the noise of the noisy calls; "babble" needs a `babble` pool."""
    from audiodf.data.prepare import _prep_dir

    out_root = Path(out_root)
    (out_root / AUDIO_DIR).mkdir(parents=True, exist_ok=True)
    horizon = settings.window_samples + int(0.5 * SR)  # past the 10 s decision horizon, as the training renders do

    def render_one(i: int) -> tuple:
        utt, call = str(idx.utt_id[i]), plan(str(idx.utt_id[i]), seed, noises, snr_range)
        target = out_root / AUDIO_DIR / f"{utt}.flac"
        if not (target.exists() and target.stat().st_size):
            with sf.SoundFile(idx.path(i)) as f:
                wave = f.read(min(f.frames, int(idx.speech_start[i]) + horizon), dtype="float32", always_2d=False)
            tmp = target.with_suffix(".part.flac")
            sf.write(tmp, make_call(wave, call, babble, str(idx.speaker[i])), SR, subtype="PCM_16")
            tmp.replace(target)
        snr = "" if call.noise == "none" else f"{call.snr_db:.1f}"
        return (utt, str(idx.speaker[i]), call.profile.name, call.noise, call.snr_bin, snr or "-", str(call.loss),
                str(idx.attack[i]), "spoof" if idx.label[i] else "bonafide")

    t0, rows = time.time(), []
    with ThreadPoolExecutor(workers) as pool:
        for k, row in enumerate(pool.map(render_one, [int(i) for i in sources]), 1):
            rows.append(row)
            if k % 2000 == 0:
                log(f"    rendered {k}/{len(sources)} calls ({k / (time.time() - t0):.0f}/s)")
    (out_root / (PROTOCOL_FILE + ".part")).write_text("\n".join(" ".join(r) for r in rows) + "\n")
    (out_root / (PROTOCOL_FILE + ".part")).replace(out_root / PROTOCOL_FILE)
    for stale in _prep_dir(settings).glob("index_calls_eval*"):  # an index of an older call set must not be reused
        stale.unlink()  # (the index file name also carries a hash of the call set's folder, see data/prepare.py)
    log(f"rendered {len(rows)} calls in {time.time() - t0:.0f}s -> {out_root / PROTOCOL_FILE}")
    return len(rows)
