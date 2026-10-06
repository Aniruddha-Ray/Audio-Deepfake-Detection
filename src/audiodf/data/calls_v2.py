"""Second simulated call set (v2): noise, echo and packet loss built differently from what run 6 trains on.

Run 6 trains on MUSAN / point-source / DEMAND noise, babble from ASV5 *train* speakers, simulated room impulse responses
and packet loss with four concealment styles. A model judged only on that same recipe would look better than it is, so this
set uses what training never sees:
  noise    ESC-50 environmental sounds (a different corpus), babble from ASV5 *eval* speakers, white / pink / brown
  echo     real, measured room impulse responses (RIRS_NOISES real_rir) instead of simulated ones
  loss     long bursts (mean 5-12 packets, 100-240 ms gaps) at 2-6% loss with `ola_repeat` concealment, a style training never uses
The codec profiles and the clean ASV5 eval source clips are the same as in the first call set. Every choice comes from the
clip ID only (label-blind). Protocol columns as in the first set; the echo is marked "+rir" in the noise column and the
loss reads "<rate>b<burst>" (e.g. 4b6) in the loss column.
"""

from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from audiodf.data.ffmpeg_codecs import _ffmpeg_roundtrip
from audiodf.data.impairments import TEST_STYLE, conceal, loss_mask, mix_noise, reverb
from audiodf.data.noise_bank import TEST_NOISE, TEST_RIR, NoiseBank
from audiodf.data.voip_sim import (AUDIO_DIR, PACKET, PROFILES, PROTOCOL_FILE, SNR_BINS, SR, BabblePool, Profile, _coloured)

NOISE_P, REVERB_P, LOSS_P = 0.75, 0.40, 0.50
SNR_RANGE = (5.0, 30.0)
NOISE_MIX = (("esc50", 0.40), ("babble", 0.35), ("white", 0.09), ("pink", 0.08), ("brown", 0.08))
LOSS_RATES, LOSS_BURSTS = (0.02, 0.04, 0.06), (5.0, 8.0, 12.0)  # bursts longer than any in training (max 4)


@dataclass(frozen=True)
class CallPlanV2:
    profile: Profile
    variant: int
    rir: bool
    noise: str  # "none", "esc50", "babble", or a colour
    snr_db: float
    loss_rate: float  # 0 = no loss
    loss_burst: float
    seed: int

    @property
    def snr_bin(self) -> str:
        if self.noise == "none":
            return "none"
        lo, hi = next(b for b in SNR_BINS if self.snr_db < b[1] or b is SNR_BINS[-1])
        return f"{lo}-{hi}"

    @property
    def noise_label(self) -> str:
        return f"{self.noise}+rir" if self.rir else self.noise

    @property
    def loss_label(self) -> str:
        return "0" if not self.loss_rate else f"{self.loss_rate * 100:g}b{self.loss_burst:g}"


def plan_v2(utt_id: str, seed: int = 0) -> CallPlanV2:
    """The call's conditions, from the clip ID and seed only (label-blind, reproducible)."""
    h = hashlib.md5(f"callv2{seed}:{utt_id}".encode()).digest()
    u = lambda i: int.from_bytes(h[i:i + 2], "little") / 65536  # noqa: E731
    x, acc, profile = u(0), 0.0, PROFILES[-1]
    for p in PROFILES:
        acc += p.weight
        if x < acc:
            profile = p
            break
    noise, snr = "none", float("nan")
    if u(3) < NOISE_P:
        y, acc = u(5), 0.0
        noise = NOISE_MIX[-1][0]
        for name, w in NOISE_MIX:
            acc += w
            if y < acc:
                noise = name
                break
        snr = SNR_RANGE[0] + u(7) * (SNR_RANGE[1] - SNR_RANGE[0])
    lossy = u(9) < LOSS_P
    return CallPlanV2(profile, profile.variants[h[2] % len(profile.variants)], u(11) < REVERB_P, noise, snr,
                      LOSS_RATES[h[13] % len(LOSS_RATES)] if lossy else 0.0,
                      LOSS_BURSTS[h[14] % len(LOSS_BURSTS)] if lossy else 1.0, int.from_bytes(hashlib.md5(h).digest()[:6], "little"))


def make_call_v2(wave: np.ndarray, call: CallPlanV2, bank: NoiseBank, babble: BabblePool | None,
                 speaker: str = "") -> np.ndarray:
    """Room echo (real RIR) -> noise -> codec -> bursty packet loss with ola_repeat concealment."""
    rng = np.random.default_rng(call.seed)
    x = np.asarray(wave, dtype=np.float32)
    if call.rir:
        x = reverb(x, bank.rir(rng, TEST_RIR))
    if call.noise != "none":
        if call.noise == "babble":
            noise = babble.sample(len(x), rng, speaker)
        elif call.noise == "esc50":
            noise = bank.noise(len(x), rng, TEST_NOISE)
        else:
            noise = _coloured(len(x), call.noise, rng)
        x = mix_noise(x, noise, call.snr_db)
    if call.profile.codec is not None:
        x = _ffmpeg_roundtrip(x, call.profile.codec, call.variant)
    if call.loss_rate:
        x = conceal(x, loss_mask(len(x) // PACKET, call.loss_rate, call.loss_burst, rng), TEST_STYLE, rng)
    return np.clip(x, -1.0, 1.0 - 2 ** -15)


def render_calls_v2(idx, sources: np.ndarray, settings, out_root: str | Path, bank: NoiseBank, babble: BabblePool,
                    seed: int = 0, workers: int = 8, log=print) -> int:
    """Write the v2 calls of the `sources` clips of `idx` (ASV5 eval) to out_root/flac/ and the protocol. Resumable."""
    from audiodf.data.prepare import _prep_dir

    bank.require(*TEST_NOISE, *TEST_RIR)
    out_root = Path(out_root)
    (out_root / AUDIO_DIR).mkdir(parents=True, exist_ok=True)
    horizon = settings.window_samples + int(0.5 * SR)

    def render_one(i: int) -> tuple:
        utt, call = str(idx.utt_id[i]), plan_v2(str(idx.utt_id[i]), seed)
        target = out_root / AUDIO_DIR / f"{utt}.flac"
        if not (target.exists() and target.stat().st_size):
            with sf.SoundFile(idx.path(i)) as f:
                wave = f.read(min(f.frames, int(idx.speech_start[i]) + horizon), dtype="float32", always_2d=False)
            tmp = target.with_suffix(".part.flac")
            sf.write(tmp, make_call_v2(wave, call, bank, babble, str(idx.speaker[i])), SR, subtype="PCM_16")
            tmp.replace(target)
        snr = "-" if call.noise == "none" else f"{call.snr_db:.1f}"
        return (utt, str(idx.speaker[i]), call.profile.name, call.noise_label, call.snr_bin, snr, call.loss_label,
                str(idx.attack[i]), "spoof" if idx.label[i] else "bonafide")

    t0, rows = time.time(), []
    with ThreadPoolExecutor(workers) as pool:
        for k, row in enumerate(pool.map(render_one, [int(i) for i in sources]), 1):
            rows.append(row)
            if k % 2000 == 0:
                log(f"    rendered {k}/{len(sources)} v2 calls ({k / (time.time() - t0):.0f}/s)")
    (out_root / (PROTOCOL_FILE + ".part")).write_text("\n".join(" ".join(r) for r in rows) + "\n")
    (out_root / (PROTOCOL_FILE + ".part")).replace(out_root / PROTOCOL_FILE)
    for stale in _prep_dir(settings).glob("index_calls_eval*"):
        stale.unlink()
    log(f"rendered {len(rows)} v2 calls in {time.time() - t0:.0f}s -> {out_root / PROTOCOL_FILE}")
    return len(rows)
