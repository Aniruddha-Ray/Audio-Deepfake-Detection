"""Channel impairments for noise- and loss-robust training (run 6): room echo, noise at 5-35 dB, bursty packet loss.

Two kinds, applied in the order a call goes through them:
  copy impairments   reverb -> noise -> (codec, rendered offline by data/ffmpeg_codecs.py): decided per clip from its ID
                     and a seed only (label-blind), baked into the rendered copy
  loss impairments   packet loss with concealment on the received audio: drawn fresh for each training window, so every
                     epoch sees different gaps (cheap numpy; the codec has already run)
Everything depends on the clip ID or an RNG seed, never on bonafide / spoof, so no channel shortcut can form.

Packet loss is a two-state (Gilbert-Elliott) chain: an average loss rate and a mean burst length (real loss comes in
bursts). A lost 20 ms packet is concealed in one of several styles, because real codecs conceal differently:
repeat_fade, zero, crossfade, comfort noise; and ola_repeat (waveform-substitution style) which training never uses, so
tests built with it measure generalisation to an unseen concealment.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.signal import fftconvolve

from audiodf.data.noise_bank import NoiseBank
from audiodf.data.voip_sim import PACKET, BabblePool, speech_rms

TRAIN_STYLES = ("repeat_fade", "zero", "crossfade", "comfort")
TEST_STYLE = "ola_repeat"
COLOURS = ("white", "pink", "brown")


# ---------------------------------------------------------------- room echo and noise

def reverb(x: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """x convolved with a room impulse response, same length, rescaled to the speech level of x (the echo changes
    how the voice sounds, not how loud the call is)."""
    y = fftconvolve(x, rir)[: len(x)].astype(np.float32)
    return y * (speech_rms(x) / (speech_rms(y) + 1e-9))


def mix_noise(x: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """x plus unit-RMS `noise` at snr_db below the speech level of x."""
    return x + (noise * speech_rms(x) / 10 ** (snr_db / 20)).astype(np.float32)


# ---------------------------------------------------------------- packet loss

def loss_mask(n_packets: int, rate: float, burst: float, rng: np.random.Generator) -> np.ndarray:
    """Lost-packet mask from a two-state chain: stationary loss `rate`, mean burst `burst` packets (1 = independent)."""
    if burst <= 1.0:
        return rng.random(n_packets) < rate
    p_leave_bad = 1.0 / burst
    p_enter_bad = rate * p_leave_bad / max(1.0 - rate, 1e-9)
    lost, bad = np.zeros(n_packets, dtype=bool), bool(rng.random() < rate)
    for k, u in enumerate(rng.random(n_packets)):
        bad = (u >= p_leave_bad) if bad else (u < p_enter_bad)
        lost[k] = bad
    return lost


def conceal(x: np.ndarray, lost: np.ndarray, style: str, rng: np.random.Generator) -> np.ndarray:
    """Replace the lost 20 ms packets of x by concealment. A lost packet at the very start is silence."""
    y = x.copy()
    ramp = np.linspace(1.0, 0.0, PACKET, dtype=np.float32)
    half = PACKET // 2
    hann = np.hanning(PACKET).astype(np.float32)
    for k in np.nonzero(lost)[0]:
        sl = slice(k * PACKET, (k + 1) * PACKET)
        prev = y[sl.start - PACKET:sl.start] if k else None
        if prev is None or style == "zero":
            y[sl] = 0.0
        elif style == "repeat_fade":
            y[sl] = 0.6 * prev
        elif style == "crossfade":  # repeat, fading out within the packet; the next good packet is faded in
            y[sl] = prev * ramp
            nxt = (k + 1) * PACKET
            if nxt + 40 <= len(y) and not (k + 1 < len(lost) and lost[k + 1]):
                y[nxt:nxt + 40] *= np.linspace(0.0, 1.0, 40, dtype=np.float32)
        elif style == "comfort":  # low-level noise at the level of the previous packet
            y[sl] = (rng.standard_normal(PACKET) * 0.3 * np.sqrt((prev.astype(np.float64) ** 2).mean())).astype(np.float32)
        elif style == "ola_repeat":  # last 10 ms repeated and overlap-added: smoother, waveform-substitution style
            tail = prev[-half:]
            fill = np.concatenate([tail, tail])
            y[sl] = (fill * hann + prev * (1.0 - hann)).astype(np.float32)
        else:
            raise ValueError(f"unknown concealment style {style!r}")
    return y


@dataclass(frozen=True)
class LossConfig:
    p: float = 0.35  # share of training windows with packet loss
    rates: tuple = (0.01, 0.02, 0.03, 0.05, 0.08)  # average loss rates, drawn per window
    bursts: tuple = (1.0, 2.0, 4.0)  # mean burst length in packets
    styles: tuple = TRAIN_STYLES


def apply_loss(x: np.ndarray, rng: np.random.Generator, cfg: LossConfig = LossConfig()) -> np.ndarray:
    """Packet loss on audio x with a random rate, burst length and concealment style from `cfg`."""
    n = len(x) // PACKET
    if n == 0:
        return x
    lost = loss_mask(n, float(rng.choice(cfg.rates)), float(rng.choice(cfg.bursts)), rng)
    return conceal(x, lost, str(rng.choice(cfg.styles)), rng) if lost.any() else x


class LossAugmenter:
    """Training-time packet loss: with probability cfg.p a window gets `apply_loss`. Label-blind (takes no label)."""

    def __init__(self, cfg: LossConfig | None = None):
        self.cfg = cfg or LossConfig()

    def __call__(self, wave: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.cfg.p <= 0 or rng.random() >= self.cfg.p:
            return wave
        return apply_loss(wave, rng, self.cfg)


# ---------------------------------------------------------------- per-clip copy impairments (reverb, noise)

@dataclass(frozen=True)
class ImpairConfig:
    reverb_p: float = 0.30
    noise_p: float = 0.65
    snr_range: tuple = (5.0, 35.0)  # dB against the speech level
    # relative weights of the noise sources; "colours" = white / pink / brown in equal parts
    # DEMAND is a 28-clip subset (the full set is 10 GB), so it gets a small weight rather than being repeated to death
    mix: dict = field(default_factory=lambda: {"musan": 0.35, "pointsource": 0.15, "demand": 0.05, "babble": 0.20,
                                               "colours": 0.25})
    rir_corpora: tuple = ("sim_rir",)
    noise_corpora: tuple = ("musan", "pointsource", "demand")


@dataclass(frozen=True)
class ImpairPlan:
    reverb: bool
    noise: str  # "none", a corpus name, "babble", or a colour
    snr_db: float  # nan when there is no noise
    seed: int


def impair_plan(utt_id: str, seed: int, cfg: ImpairConfig = ImpairConfig()) -> ImpairPlan:
    """The copy's reverb / noise, from the clip ID and seed only (label-blind, reproducible)."""
    h = hashlib.md5(f"imp{seed}:{utt_id}".encode()).digest()
    u = lambda i: int.from_bytes(h[i:i + 2], "little") / 65536  # noqa: E731
    noise, snr = "none", float("nan")
    if u(2) < cfg.noise_p:
        names, weights = list(cfg.mix), np.array(list(cfg.mix.values()), dtype=float)
        pick = names[int(np.searchsorted(np.cumsum(weights / weights.sum()), u(4), side="right").clip(0, len(names) - 1))]
        noise = COLOURS[h[6] % 3] if pick == "colours" else pick
        snr = cfg.snr_range[0] + u(7) * (cfg.snr_range[1] - cfg.snr_range[0])
    return ImpairPlan(u(0) < cfg.reverb_p, noise, snr, int.from_bytes(h[9:15], "little"))


def apply_impairment(x: np.ndarray, plan: ImpairPlan, bank: NoiseBank | None, babble: BabblePool | None,
                     cfg: ImpairConfig = ImpairConfig(), speaker: str = "") -> np.ndarray:
    """Reverb, then noise, on clean speech x (before the codec)."""
    rng = np.random.default_rng(plan.seed)
    y = np.asarray(x, dtype=np.float32)
    if plan.reverb:
        y = reverb(y, bank.rir(rng, cfg.rir_corpora))
    if plan.noise == "none":
        return y
    if plan.noise == "babble":
        noise = babble.sample(len(y), rng, speaker)
    elif plan.noise in COLOURS:
        from audiodf.data.voip_sim import _coloured

        noise = _coloured(len(y), plan.noise, rng)
    else:
        noise = bank.noise(len(y), rng, (plan.noise,))
    return mix_noise(y, noise, plan.snr_db)


@dataclass
class ImpairKit:
    """Everything needed to impair a clip before its codec: the noise settings, the noise bank, the babble pool, and the
    EnCodec share of catalogue 3. `tag` hashes the configuration, so rendered copies of other settings are never reused."""

    cfg: ImpairConfig
    bank: NoiseBank
    babble: BabblePool | None
    neural_share: float = 0.07

    @property
    def tag(self) -> str:
        blob = json.dumps({"cfg": asdict(self.cfg), "neural_share": self.neural_share, "bank": self.bank.corpora(),
                           "babble": len(self.babble.paths) if self.babble else 0}, sort_keys=True)
        return hashlib.md5(blob.encode()).hexdigest()[:8]

    def apply(self, wave: np.ndarray, utt_id: str, seed: int, speaker: str = "") -> np.ndarray:
        return apply_impairment(wave, impair_plan(utt_id, seed, self.cfg), self.bank, self.babble, self.cfg, speaker)


def build_kit(settings, pool) -> ImpairKit:
    """The impairment kit for training from the settings: noise bank from paths.noise_root, babble from the genuine clips
    of the training pool `pool` (other speakers' speech; never the same speaker as the clip)."""
    d = settings.data
    cfg = ImpairConfig(reverb_p=d.reverb_p, noise_p=d.noise_p, snr_range=tuple(d.snr_db))
    bank = NoiseBank.scan(settings.paths.noise_root)
    bank.require(*cfg.noise_corpora, *cfg.rir_corpora)
    return ImpairKit(cfg, bank, BabblePool.from_index(pool), d.neural_share)
