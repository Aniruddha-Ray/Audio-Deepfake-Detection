import inspect
import os

import numpy as np
import pytest
import soundfile as sf

from audiodf.data.impairments import (COLOURS, TEST_STYLE, TRAIN_STYLES, ImpairConfig, LossAugmenter, LossConfig,
                                      apply_impairment, apply_loss, conceal, impair_plan, loss_mask, mix_noise, reverb)
from audiodf.data.noise_bank import NoiseBank, prepare_parquet_noise
from audiodf.data.voip_sim import PACKET, speech_rms

from conftest import SR, tone
from test_prepare import ROWS  # noqa: F401  (fixture data shared with test_prepare)
from test_prepare import asv5  # noqa: F401


def _speechlike(seconds=3.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    return ((0.2 * np.sin(2 * np.pi * 220 * t) + 0.02 * rng.standard_normal(len(t))) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))
            ).astype(np.float32)


def _bank(tmp_path):
    """A tiny noise bank on disk: two noise corpora and two RIR corpora."""
    files = {}
    for corpus, n in (("musan", 3), ("esc50", 2)):
        files[corpus] = []
        for k in range(n):
            p = tmp_path / f"{corpus}_{k}.wav"
            sf.write(p, tone(2.0, 300 + 100 * k, 0.2, k), SR)
            files[corpus].append(str(p))
    for corpus in ("sim_rir", "real_rir"):
        p = tmp_path / f"{corpus}.wav"
        h = np.zeros(4000, dtype=np.float32)
        h[0], h[800], h[2400] = 0.4, 0.3, 0.2  # a direct path (not the largest tap) and two echoes
        sf.write(p, h, SR)
        files[corpus] = [str(p)]
    return NoiseBank(files)


def test_noise_bank_returns_unit_rms_segments_and_scaled_rirs(tmp_path):
    bank = _bank(tmp_path)
    assert bank.corpora() == {"musan": 3, "esc50": 2, "sim_rir": 1, "real_rir": 1}
    rng = np.random.default_rng(0)
    for n in (SR // 2, 5 * SR):  # shorter and longer than the 2 s files: a short file is repeated
        seg = bank.noise(n, rng, ("musan",))
        assert seg.shape == (n,) and abs(float(np.sqrt((seg ** 2).mean())) - 1.0) < 1e-3
    h = bank.rir(rng, ("sim_rir",))
    assert abs(float(np.abs(h).max()) - 1.0) < 1e-6 and len(h) <= SR  # peak 1, cut at 1 s
    with pytest.raises(FileNotFoundError, match="not found"):
        bank.require("musan", "demand")
    with pytest.raises(ValueError, match="16000"):
        sf.write(tmp_path / "bad.wav", tone(1.0), 8000)
        NoiseBank({"x": [str(tmp_path / "bad.wav")]}).noise(SR, rng, ("x",))


def test_parquet_noise_is_unpacked_to_16k_mono(tmp_path):
    import io

    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    rows = []
    for k, sr in enumerate((44100, 16000)):
        buf = io.BytesIO()
        sf.write(buf, tone(1.0, 400 + 100 * k, 0.1, k)[: sr] if sr == 16000 else np.tile(tone(1.0), 3)[:sr], sr, format="WAV")
        rows.append({"audio": {"bytes": buf.getvalue(), "path": None}})
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "noise-0.parquet")
    assert prepare_parquet_noise([tmp_path / "noise-0.parquet"], tmp_path / "wav", log=lambda *_: None) == 2
    for p in (tmp_path / "wav").glob("*.flac"):
        info = sf.info(p)
        assert info.samplerate == SR and info.channels == 1 and abs(info.frames - SR) <= 2


def test_reverb_keeps_the_speech_level_and_adds_echo():
    x = _speechlike(2.0)
    h = np.zeros(2000, dtype=np.float32)
    h[0], h[1000] = 1.0, 0.5
    y = reverb(x, h)
    assert y.shape == x.shape and abs(speech_rms(y) / speech_rms(x) - 1.0) < 1e-4
    assert not np.allclose(y, x, atol=1e-3)
    # an echo 1000 samples later: the correlation of y with x at that lag is clearly positive
    assert float(np.dot(y[1000:], x[:-1000])) > 0.2 * float(np.dot(x, x))


def test_noise_mixing_hits_the_snr():
    x = _speechlike(3.0)
    noise = np.random.default_rng(0).standard_normal(len(x))
    noise /= np.sqrt((noise ** 2).mean())
    for snr in (5.0, 20.0, 35.0):
        n = mix_noise(x, noise, snr) - x
        assert abs(20 * np.log10(speech_rms(x) / np.sqrt((n ** 2).mean())) - snr) < 0.05


def test_loss_mask_matches_the_requested_rate_and_burst_length():
    rng = np.random.default_rng(0)
    for rate, burst in ((0.03, 1.0), (0.05, 4.0), (0.08, 2.0)):
        m = loss_mask(200000, rate, burst, rng)
        assert abs(m.mean() - rate) < 0.006, (rate, burst, m.mean())
        runs = np.diff(np.flatnonzero(np.diff(np.r_[0, m.astype(int), 0])))[::2]  # lengths of the loss bursts
        assert abs(runs.mean() - max(burst, 1.0 / (1.0 - rate))) < 0.3 * burst, (rate, burst, runs.mean())


@pytest.mark.parametrize("style", TRAIN_STYLES + (TEST_STYLE,))
def test_every_concealment_style_changes_only_lost_packets(style):
    x = _speechlike(2.0)
    lost = np.zeros(len(x) // PACKET, dtype=bool)
    lost[[0, 10, 11, 40]] = True
    y = conceal(x, lost, style, np.random.default_rng(1))
    assert y.shape == x.shape and np.isfinite(y).all() and np.abs(y).max() <= 1.0
    packet = lambda a, k: a[k * PACKET:(k + 1) * PACKET]  # noqa: E731
    assert np.allclose(packet(y, 0), 0)  # nothing to repeat at the very start
    for k in (5, 20, 30):  # untouched packets stay identical (a crossfade only fades in the packet after a loss)
        assert np.array_equal(packet(y, k), packet(x, k))
    if style != "zero":
        assert float(np.abs(packet(y, 10)).max()) > 0  # a concealed packet is not silence (except for zero-fill)
    assert not np.array_equal(packet(y, 40), packet(x, 40))
    with pytest.raises(ValueError, match="unknown concealment"):
        conceal(x, lost, "nope", np.random.default_rng(0))


def test_training_never_uses_the_test_concealment_style():
    assert TEST_STYLE not in LossConfig().styles and set(LossConfig().styles) == set(TRAIN_STYLES)
    rng = np.random.default_rng(0)
    x = _speechlike(2.0)
    aug = LossAugmenter(LossConfig(p=1.0, rates=(0.5,)))  # half the packets lost: the audio certainly changes
    out = aug(x, rng)
    assert out.shape == x.shape and not np.array_equal(out, x)
    never = LossAugmenter(LossConfig(p=0.0))
    assert never(x, rng) is x
    changed = np.mean([not np.array_equal(LossAugmenter(LossConfig(p=0.35))(x, np.random.default_rng(s)), x) for s in range(400)])
    assert 0.2 < changed < 0.4  # about the configured share of windows gets loss
    assert apply_loss(x[:100], rng).shape == (100,)  # shorter than one packet: untouched


def test_impairment_plan_is_label_blind_reproducible_and_matches_the_config():
    params = list(inspect.signature(impair_plan).parameters)
    assert params == ["utt_id", "seed", "cfg"] and not any("label" in p or "key" in p for p in params)
    assert impair_plan("T_1", 1) == impair_plan("T_1", 1) and impair_plan("T_1", 1) != impair_plan("T_1", 2)
    cfg = ImpairConfig()
    plans = [impair_plan(f"T_{k:010d}", 1, cfg) for k in range(8000)]
    assert abs(np.mean([p.reverb for p in plans]) - cfg.reverb_p) < 0.02
    assert abs(np.mean([p.noise != "none" for p in plans]) - cfg.noise_p) < 0.02
    noisy = [p for p in plans if p.noise != "none"]
    snr = np.array([p.snr_db for p in noisy])
    assert 5 <= snr.min() and snr.max() <= 35 and abs(snr.mean() - 20) < 0.5
    share = lambda name: np.mean([p.noise == name for p in noisy])  # noqa: E731
    assert abs(share("musan") - 0.35) < 0.03 and abs(share("babble") - 0.20) < 0.03 and abs(share("demand") - 0.05) < 0.03
    assert abs(sum(share(c) for c in COLOURS) - 0.25) < 0.03 and set(p.noise for p in noisy) <= {
        "musan", "pointsource", "demand", "babble", *COLOURS}


def test_apply_impairment_runs_every_noise_kind(tmp_path):
    from audiodf.data.voip_sim import BabblePool

    bank = _bank(tmp_path)
    cfg = ImpairConfig(rir_corpora=("sim_rir",))
    paths, speakers = [], []
    for k in range(10):
        p = tmp_path / f"talker{k}.flac"
        sf.write(p, tone(5.0, 150 + 40 * k, 0.05, k), SR)
        paths.append(str(p))
        speakers.append(f"S{k}")
    babble = BabblePool(paths, speakers, [0] * 10, [5 * SR] * 10)
    x = _speechlike(3.0)
    from audiodf.data.impairments import ImpairPlan

    for noise in ("none", "musan", "babble", "white", "pink", "brown"):
        for rv in (False, True):
            plan = ImpairPlan(rv, noise, 10.0 if noise != "none" else float("nan"), 7)
            y = apply_impairment(x, plan, bank, babble, cfg, speaker="S0")
            assert y.shape == x.shape and np.isfinite(y).all()
            if noise == "none" and not rv:
                assert np.array_equal(y, x)
            else:
                assert not np.allclose(y, x, atol=1e-4)
    a = apply_impairment(x, ImpairPlan(True, "musan", 12.0, 3), bank, babble, cfg)
    assert np.array_equal(a, apply_impairment(x, ImpairPlan(True, "musan", 12.0, 3), bank, babble, cfg))  # reproducible


# ---------------------------------------------------------------- integration: rendering and training

def _kit(tmp_path, **cfg):
    from audiodf.data.impairments import ImpairKit
    from audiodf.data.voip_sim import BabblePool

    bank = _bank(tmp_path)
    paths, speakers = [], []
    for k in range(10):
        p = tmp_path / f"talker{k}.flac"
        sf.write(p, tone(5.0, 150 + 40 * k, 0.05, k), SR)
        paths.append(str(p))
        speakers.append(f"S{k}")
    mix = {"musan": 0.5, "babble": 0.25, "colours": 0.25}  # only the corpora this tiny bank has
    cfg = ImpairConfig(noise_corpora=("musan",), rir_corpora=("sim_rir",), mix=mix, **cfg)
    return ImpairKit(cfg, bank, BabblePool(paths, speakers, [0] * 10, [5 * SR] * 10), neural_share=0.0)


def _skip_if_catalogue3_unrenderable(idx, seed, kit):
    from audiodf.data.ffmpeg_codecs import CLASSICAL3, choice, missing_codecs

    names = {choice(str(u), seed, kit.neural_share, CLASSICAL3)[1].name for u in idx.utt_id}
    if missing_codecs(sorted(names)):
        pytest.skip("this ffmpeg build lacks a codec the fixture clips are assigned to (libgsm on Linux)")


def test_render_copies_with_impairments_goes_to_a_configuration_tagged_catalogue3(asv5, tmp_path):
    from audiodf.data.ffmpeg_codecs import render_copies
    from audiodf.data.prepare import build_index

    idx = build_index(asv5, "asv5", "train", workers=0)
    kit = _kit(tmp_path, reverb_p=1.0, noise_p=1.0, snr_range=(5.0, 10.0))
    _skip_if_catalogue3_unrenderable(idx, 7, kit)
    rendered = render_copies(idx, np.arange(len(idx)), asv5, frac=1.0, seed=7, workers=2, log=lambda *_: None, impair=kit)
    assert rendered.rendered.all() and rendered.render_tag.startswith(f"ff3_{kit.tag}s7")
    assert f"render_ff3_{kit.tag}" in str(rendered.path(0)).replace("\\", "/")
    plain = render_copies(idx, np.arange(len(idx)), asv5, frac=1.0, seed=7, workers=2, log=lambda *_: None)
    assert "render_ff2" in str(plain.path(0)).replace("\\", "/")  # no kit: catalogue 2, as runs 4-5
    for i in range(len(idx)):  # noise at 5-10 dB and echo: far from the codec-only copy of the same clip
        noisy, _ = sf.read(rendered.path(i), dtype="float32")
        codec_only, _ = sf.read(plain.path(i), dtype="float32")
        n = min(len(noisy), len(codec_only))
        assert float(np.sqrt(((noisy[:n] - codec_only[:n]) ** 2).mean())) > 0.02
    mtime = os.path.getmtime(rendered.path(0))
    again = render_copies(idx, np.arange(len(idx)), asv5, frac=1.0, seed=7, workers=2, log=lambda *_: None, impair=kit)
    assert os.path.getmtime(again.path(0)) == mtime  # resumable
    (tmp_path / "k2").mkdir()
    other = _kit(tmp_path / "k2", reverb_p=0.1, noise_p=0.2)
    assert other.tag != kit.tag  # other settings never reuse these copies


def test_run_training_with_noise_echo_and_packet_loss(tmp_path):
    from audiodf.data.ffmpeg_codecs import missing_codecs
    from audiodf.training.pipeline import run_training

    from test_training_pipeline import _settings, _write_split

    if missing_codecs():
        pytest.skip("rendering catalogue 3 needs every codec (libgsm is missing on Linux)")
    root = tmp_path / "noise"
    for folder, ext in (("musan/musan/noise/free-sound", "wav"), ("rirs/RIRS_NOISES/pointsource_noises", "wav"),
                        ("demand_wav", "flac"), ("rirs/RIRS_NOISES/simulated_rirs/smallroom", "wav")):
        (root / folder).mkdir(parents=True)
        for k in range(2):
            h = np.zeros(3000, dtype=np.float32)
            h[0], h[900] = 1.0, 0.4
            sf.write(root / folder / f"n{k}.{ext}", h if "rirs" in folder and "point" not in folder else tone(2.0, 300 + 80 * k, 0.2, k), SR)
    _write_split(tmp_path, "T", "flac_T", "ASVspoof5.train.tsv", ["A01", "A02"], n=48)
    _write_split(tmp_path, "D", "flac_D", "ASVspoof5.dev.track_1.tsv", ["A09", "A10"], n=48)
    s = _settings()
    s.paths.asv5_root, s.paths.data_root = str(tmp_path), str(tmp_path / "no_asv19")
    s.paths.noise_root = str(root)
    s.paths.cache_dir, s.paths.artifacts_dir = str(tmp_path / "cache"), str(tmp_path / "artifacts")
    s.paths.results_dir = str(tmp_path / "results")
    s.rcnn_train.epochs, s.rcnn_train.batch_size = 1, 4
    s.data.tune_utts, s.data.train_splits, s.data.holdout_attacks = 24, ("asv5:train",), ()
    s.ensemble.branches = ("rcnn",)
    s.data.impairments, s.data.render_frac, s.data.loss_p = True, 1.0, 0.5
    lines = []
    report = run_training(s, workers=0, log=lambda m: lines.append(str(m)))
    assert any("room echo 30%, noise 65% at 5-35 dB" in line and "packet loss on 50%" in line for line in lines)
    assert any("impairments " in line for line in lines) and len(report["histories"]["rcnn"]) == 1
    assert any(p.name.startswith("render_ff3_") for p in (tmp_path / "cache").iterdir())
    s.data.impairments = False  # the old recipe still runs, in catalogue 2
    lines.clear()
    run_training(s, workers=0, log=lambda m: lines.append(str(m)))
    assert any(p.name == "render_ff2" for p in (tmp_path / "cache").iterdir())


# ---------------------------------------------------------------- the second call set (held-out noise, echo and loss)

def test_v2_plan_is_label_blind_and_built_from_conditions_training_never_uses():
    from audiodf.data import calls_v2

    params = list(inspect.signature(calls_v2.plan_v2).parameters)
    assert params == ["utt_id", "seed"]  # nothing about the label
    assert calls_v2.plan_v2("E_1", 2) == calls_v2.plan_v2("E_1", 2) and calls_v2.plan_v2("E_1", 2) != calls_v2.plan_v2("E_1", 3)
    plans = [calls_v2.plan_v2(f"E_{k:010d}", 2) for k in range(8000)]
    assert abs(np.mean([p.noise != "none" for p in plans]) - 0.75) < 0.02
    assert abs(np.mean([p.rir for p in plans]) - 0.40) < 0.02 and abs(np.mean([p.loss_rate > 0 for p in plans]) - 0.5) < 0.02
    noisy = [p for p in plans if p.noise != "none"]
    assert {p.noise for p in noisy} == {"esc50", "babble", "white", "pink", "brown"}
    assert abs(np.mean([p.noise == "esc50" for p in noisy]) - 0.40) < 0.03
    assert 5 <= min(p.snr_db for p in noisy) and max(p.snr_db for p in noisy) <= 30
    lossy = [p for p in plans if p.loss_rate]
    assert {p.loss_rate for p in lossy} == {0.02, 0.04, 0.06} and {p.loss_burst for p in lossy} == {5.0, 8.0, 12.0}
    # what training uses is disjoint from what this set tests with
    assert calls_v2.TEST_NOISE == ("esc50",) and "esc50" not in ImpairConfig().noise_corpora
    assert calls_v2.TEST_RIR == ("real_rir",) and "real_rir" not in ImpairConfig().rir_corpora
    assert TEST_STYLE not in LossConfig().styles
    assert max(LossConfig().bursts) < min(calls_v2.LOSS_BURSTS)  # the test's bursts are longer than any in training
    labels = {p.noise_label for p in plans}
    assert "babble+rir" in labels and "none" in labels and {p.loss_label for p in lossy} >= {"2b5", "6b12"}


def test_v2_calls_render_through_the_standard_evaluation_protocol(asv5, tmp_path):
    from audiodf.data import calls_v2
    from audiodf.data.ffmpeg_codecs import missing_codecs
    from audiodf.data.prepare import build_index
    from audiodf.data.protocol import read_protocol
    from audiodf.data.voip_sim import BabblePool, select_sources

    idx = build_index(asv5, "asv5", "train", workers=0)
    sources = select_sources(idx, n_genuine=2, n_per_attack=1, seed=3)
    bank = _bank(tmp_path)  # musan, esc50, sim_rir, real_rir
    paths, speakers = [], []
    for k in range(10):
        p = tmp_path / f"talker{k}.flac"
        sf.write(p, tone(5.0, 150 + 40 * k, 0.05, k), SR)
        paths.append(str(p))
        speakers.append(f"S{k}")
    babble = BabblePool(paths, speakers, [0] * 10, [5 * SR] * 10)
    if missing_codecs(sorted({calls_v2.plan_v2(str(u), 3).profile.name for u in idx.utt_id[sources]} & {"gsm"})):
        pytest.skip("this ffmpeg build lacks libgsm")
    asv5.paths.calls_root = str(tmp_path / "v2")
    n = calls_v2.render_calls_v2(idx, sources, asv5, asv5.paths.calls_root, bank, babble, seed=3, workers=2,
                                 log=lambda *_: None)
    samples = read_protocol(asv5.dataset_root("calls"), "eval", dataset="calls")
    assert n == len(samples) == 4
    by = {s.utt_id: s for s in samples}
    for i in sources:
        s = by[str(idx.utt_id[i])]
        assert (s.label, s.attack) == (int(idx.label[i]), str(idx.attack[i])) and s.codec == calls_v2.plan_v2(s.utt_id, 3).profile.name
    row = (tmp_path / "v2" / "ASV5.eval.calls.tsv").read_text().splitlines()[0].split()
    assert len(row) == 9
    with pytest.raises(FileNotFoundError, match="not found"):  # no held-out corpora, no v2 set
        from audiodf.data.noise_bank import NoiseBank

        calls_v2.render_calls_v2(idx, sources, asv5, tmp_path / "x", NoiseBank({"musan": bank.files["musan"]}), babble)
