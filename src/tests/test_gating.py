import inspect

import numpy as np
import pytest
from scipy.signal import fftconvolve

from audiodf.gating.channel_quality import (FEATURE_NAMES, FEATURE_VERSION, ChannelQualityModel, channel_features, leak_auc)
from audiodf.gating.fusion import gate_weight, gated_fuse, tune_gate

SR = 16000


def _speech(rng, seconds=10.0):
    t = np.arange(int(seconds * SR)) / SR
    f0, syll = rng.uniform(100, 220), rng.uniform(3, 6)
    env = np.clip(np.sin(2 * np.pi * syll * t + rng.uniform(0, 6)) ** 2 * (np.sin(2 * np.pi * 0.4 * t) > -0.3), 0, 1)
    x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 12)) * env
    return (x / np.abs(x).max() * 0.3).astype(np.float32)


def _degrade(rng, x, echo, snr_db):
    if echo:
        rt = rng.uniform(0.3, 1.0)
        n = int(rt * SR)
        h = rng.standard_normal(n) * np.exp(-6.9 * np.arange(n) / SR / rt)
        h[0] = 3
        y = fftconvolve(x, h)[: len(x)]
        x = (y * np.sqrt((x ** 2).mean() / (y ** 2).mean())).astype(np.float32)
    if snr_db is not None:
        x = x + rng.standard_normal(len(x)).astype(np.float32) * np.sqrt((x ** 2).mean() / 10 ** (snr_db / 10))
    return x


def test_features_have_the_stated_names_and_are_finite_even_for_silence_and_very_short_clips():
    rng = np.random.default_rng(0)
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))
    for x in (_speech(rng), _speech(rng, 1.0), np.zeros(SR, dtype=np.float32), np.zeros(10, dtype=np.float32)):
        f = channel_features(x)
        assert f.shape == (len(FEATURE_NAMES),) and np.isfinite(f).all()
    long = np.concatenate([_speech(rng), _speech(rng)])
    assert np.array_equal(channel_features(long), channel_features(long[: 10 * SR]))  # only the first 10 s count


def test_the_estimator_takes_no_label_and_separates_echo_and_noise_it_was_not_shown():
    assert "label" not in " ".join(inspect.signature(channel_features).parameters) and "key" not in inspect.signature(
        ChannelQualityModel.fit).parameters
    rng = np.random.default_rng(1)
    clips = []
    for _ in range(360):
        echo, noisy = rng.random() < 0.5, rng.random() < 0.5
        clips.append((_degrade(rng, _speech(rng, 6.0), echo, rng.uniform(0, 18) if noisy else None), echo, np.nan))
        clips[-1] = (clips[-1][0], echo, noisy)
    x = np.stack([channel_features(c[0]) for c in clips])
    echo = np.array([c[1] for c in clips])
    noisy = np.array([c[2] for c in clips])
    snr = np.where(noisy, 10.0, np.nan)
    model = ChannelQualityModel().fit(x[:240], echo[:240], snr[:240])
    pe, pn, q = model.predict(x[240:])
    assert ((pe > .5) == echo[240:]).mean() > 0.9 and ((pn > .5) == noisy[240:]).mean() > 0.9
    assert q[~echo[240:] & ~noisy[240:]].mean() < 0.3 < 0.7 < q[echo[240:] | noisy[240:]].mean()


def test_model_save_load_and_stale_version(tmp_path):
    rng = np.random.default_rng(2)
    x = rng.standard_normal((60, len(FEATURE_NAMES)))
    echo = np.arange(60) % 2 == 0
    model = ChannelQualityModel().fit(x, echo, np.where(echo, 5.0, np.nan))
    model.save(tmp_path / "m.joblib")
    again = ChannelQualityModel.load(tmp_path / "m.joblib")
    assert np.allclose(model.predict(x)[2], again.predict(x)[2])
    model.version = FEATURE_VERSION + 1
    model.save(tmp_path / "stale.joblib")
    with pytest.raises(ValueError, match="feature version"):
        ChannelQualityModel.load(tmp_path / "stale.joblib")


def test_gate_weight_is_bounded_and_the_fusion_is_a_convex_mix():
    q = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
    w = gate_weight(q, 0.1, 0.8)
    assert np.allclose(w, [0.1, 0.1, 0.45, 0.8, 0.8]) and w.min() >= 0.1 and w.max() <= 0.8
    p1, p2 = np.array([0.2, 0.2, 0.2]), np.array([0.8, 0.8, 0.8])
    assert np.allclose(gated_fuse(p1, p2, np.array([0.0, 0.5, 1.0]), 0.0, 0.5), [0.2, 0.35, 0.5])
    two = gated_fuse(np.tile(p1[:, None], (1, 4)), np.tile(p2[:, None], (1, 4)), np.array([0.0, 1.0, 1.0]), 0.0, 0.5)
    assert two.shape == (3, 4) and np.allclose(two[0], 0.2) and np.allclose(two[1], 0.5)
    assert np.allclose(gated_fuse(p1, p2, np.ones(3), 0.0, 0.0), p1)  # w_max = 0: WavLM alone


def _scores(rng, n, good_wavlm, good_whisper):
    y = np.arange(n) % 2
    s = lambda good: np.clip(y * good + (1 - good) * rng.random(n) + 0.02 * rng.standard_normal(n), 0, 1)  # noqa: E731
    return y, np.tile(s(good_wavlm)[:, None], (1, 3)), np.tile(s(good_whisper)[:, None], (1, 3))


def test_the_gate_keeps_whisper_small_where_it_is_useless_and_large_where_it_helps():
    rng = np.random.default_rng(3)
    n = 800
    q = np.r_[np.zeros(n // 2), np.ones(n // 2)]
    y1, w1, h1 = _scores(rng, n // 2, 0.9, 0.3)  # clean calls: Whisper is junk
    y2, w2, h2 = _scores(rng, n // 2, 0.3, 0.9)  # degraded calls: WavLM is junk, Whisper is good
    y = np.r_[y1, y2]
    pw, ph = np.vstack([w1, w2]), np.vstack([h1, h2])
    (lo, hi), table = tune_gate(pw, ph, q, y)
    assert lo == 0.0 and hi >= 0.5 and len(table) == 9  # nothing on clean calls, a real share on degraded ones
    key = f"w_min={lo:g} w_max={hi:g}"
    assert table[key] <= min(table.values()) + 0.05  # within the tie of the best setting
    from audiodf.evaluation.metrics import compute_metrics

    eer = lambda f: np.mean([compute_metrics(y, f[:, k])["eer_pct"] for k in range(3)])  # noqa: E731
    assert eer(gated_fuse(pw, ph, q, 0.0, 0.0)) > eer(gated_fuse(pw, ph, q, lo, hi)) + 3  # better than WavLM alone
    (lo2, hi2), _ = tune_gate(w1, h1, np.zeros(len(y1)), y1)
    assert hi2 >= lo2 and lo2 == 0.0 and gate_weight(0.0, lo2, hi2) == 0.0  # clean-only: Whisper stays off
    # equal scores: ties resolve to the smaller Whisper weight
    (lo3, hi3), _ = tune_gate(w1, w1, np.zeros(len(y1)), y1)
    assert (lo3, hi3) == (0.0, 0.5)


def test_leak_auc_is_chance_for_an_unrelated_q_and_one_for_a_q_that_is_the_label():
    rng = np.random.default_rng(4)
    y = np.arange(4000) % 2
    assert abs(leak_auc(rng.random(4000), y) - 0.5) < 0.04 and leak_auc(y.astype(float), y) == 1.0
    assert np.isnan(leak_auc(rng.random(10), np.zeros(10, dtype=int)))
