import inspect
import json

import numpy as np
import pytest

pytest.importorskip("imageio_ffmpeg")

from audiodf.calibrate import (calibrate, make_wavlm_bundle, policy_thresholds, speaker_halves,  # noqa: E402
                               threshold_rates)
from audiodf.data.ffmpeg_codecs import missing_codecs  # noqa: E402
from audiodf.data.prepare import build_index  # noqa: E402
from audiodf.data.protocol import read_protocol  # noqa: E402
from audiodf.data.voip_sim import (CALL_COLUMNS, PROFILES, CallPlan, add_noise, drop_packets, make_call,  # noqa: E402
                                   plan, render_calls, select_sources, speech_rms)
from audiodf.evaluation.compare_runs import eer_pct, paired_bootstrap  # noqa: E402
from audiodf.evaluation.metrics import compute_metrics  # noqa: E402

from conftest import SR, tone  # noqa: E402
from test_prepare import ROWS  # noqa: E402,F401
from test_prepare import asv5  # noqa: E402,F401


def _speechlike(seconds=3.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
    return ((0.2 * np.sin(2 * np.pi * 220 * t) + 0.02 * rng.standard_normal(len(t))) * env).astype(np.float32)


# ---------------------------------------------------------------- the call chain

def test_call_conditions_are_label_blind_reproducible_and_match_the_stated_mix():
    assert list(inspect.signature(plan).parameters) == ["utt_id", "seed"]  # nothing about the label
    assert plan("E_0000000001", 0) == plan("E_0000000001", 0) and plan("E_0000000001", 0) != plan("E_0000000001", 1)
    plans = [plan(f"E_{k:010d}", 0) for k in range(8000)]
    for p in PROFILES:
        assert abs(np.mean([c.profile.name == p.name for c in plans]) - p.weight) < 0.02, p.name
    assert abs(np.mean([c.noise != "none" for c in plans]) - 0.75) < 0.02
    assert abs(np.mean([c.loss == 0 for c in plans]) - 0.5) < 0.02
    snr = np.array([c.snr_db for c in plans if c.noise != "none"])
    assert 15 <= snr.min() and snr.max() <= 35 and abs(snr.mean() - 25) < 0.5
    assert {c.snr_bin for c in plans} == {"none", "15-20", "20-25", "25-30", "30-35"}
    assert {c.noise for c in plans} == {"none", "white", "pink", "brown"} and {c.loss for c in plans} == {0, 1, 3, 5}


def test_noise_is_added_at_the_requested_level_and_colour():
    x = _speechlike(4.0)
    rng = np.random.default_rng(0)
    low = lambda n: float((np.abs(np.fft.rfft(n)) ** 2)[: len(n) // 16].sum() / (np.abs(np.fft.rfft(n)) ** 2).sum())  # noqa: E731
    shares = {}
    for kind in ("white", "pink", "brown"):
        noise = add_noise(x, kind, 20.0, rng) - x
        assert abs(20 * np.log10(speech_rms(x) / np.sqrt((noise ** 2).mean())) - 20) < 0.1, kind
        shares[kind] = low(noise)
    assert shares["white"] < shares["pink"] < shares["brown"]  # energy moves to low frequencies


def test_packet_loss_conceals_by_repeating_and_fading():
    x = _speechlike(2.0)
    y = drop_packets(x, 0.0, np.random.default_rng(0))
    assert np.array_equal(x, y)
    y = drop_packets(x, 0.1, np.random.default_rng(1))
    n_packets = len(x) // 320
    changed = (x[: n_packets * 320] != y[: n_packets * 320]).reshape(n_packets, 320).any(axis=1)
    assert 0.03 < changed.mean() < 0.2  # about the requested 10% of packets were replaced, no others
    k = int(np.nonzero(changed)[0][-1])  # a replaced packet repeats its predecessor at 0.6 gain
    assert np.allclose(y[k * 320:(k + 1) * 320], 0.6 * y[(k - 1) * 320:k * 320])
    assert np.array_equal(drop_packets(x, 0.1, np.random.default_rng(1)), y)  # reproducible
    lost_first = drop_packets(x, 1.0, np.random.default_rng(0))
    assert np.allclose(lost_first[:320], 0) and np.abs(lost_first).max() == 0  # everything lost: silence stays silence


@pytest.mark.parametrize("profile", [p.name for p in PROFILES])
def test_every_voip_profile_returns_audio_of_the_same_length(profile):
    if profile == "gsm" and missing_codecs(["gsm"]):
        pytest.skip("this ffmpeg build lacks libgsm")
    p = next(q for q in PROFILES if q.name == profile)
    x = _speechlike(3.0)
    out = make_call(x, CallPlan(p, p.variants[0], "none", float("nan"), 0, 0))
    assert out.shape == x.shape and np.isfinite(out).all() and np.abs(out).max() <= 1.0
    if p.codec is not None:  # it really went through a codec
        assert not np.allclose(out, x, atol=1e-3)
        mid = slice(SR, 2 * SR)
        best = max(range(-40, 41), key=lambda k: float(np.dot(out[mid.start + k:mid.stop + k], x[mid])))
        assert abs(best) <= 8  # codec delay removed
    else:
        assert np.allclose(out, x, atol=2e-5)


def test_render_calls_writes_a_protocol_the_evaluator_reads(asv5, tmp_path):
    idx = build_index(asv5, "asv5", "train", workers=0)
    sources = select_sources(idx, n_genuine=2, n_per_attack=1, seed=3)
    assert len(sources) == 4 and sorted(idx.attack[sources]) == ["-", "-", "A05", "A06"]
    with pytest.raises(ValueError, match="not enough"):
        select_sources(idx, n_genuine=3, n_per_attack=1)
    asv5.paths.calls_root = str(tmp_path / "calls")
    if missing_codecs(sorted({plan(str(u), 3).profile.name for u in idx.utt_id[sources]} & {"gsm"})):
        pytest.skip("this ffmpeg build lacks libgsm")
    lines = []
    assert render_calls(idx, sources, asv5, asv5.paths.calls_root, seed=3, workers=2, log=lines.append) == 4
    samples = read_protocol(asv5.dataset_root("calls"), "eval", dataset="calls", available_only=True)
    assert len(samples) == 4 and {s.dataset for s in samples} == {"calls"}
    by = {s.utt_id: s for s in samples}
    for i in sources:
        s = by[str(idx.utt_id[i])]
        assert (s.label, s.attack, s.speaker) == (int(idx.label[i]), str(idx.attack[i]), str(idx.speaker[i]))
        assert s.codec == plan(s.utt_id, 3).profile.name  # the per-condition label is the VoIP profile
    row = (tmp_path / "calls" / "ASV5.eval.calls.tsv").read_text().splitlines()[0].split()
    assert len(row) == len(CALL_COLUMNS) == 9
    stamp = (tmp_path / "calls" / "flac" / f"{samples[0].utt_id}.flac").stat().st_mtime_ns
    render_calls(idx, sources, asv5, asv5.paths.calls_root, seed=3, workers=2, log=lambda *_: None)
    assert (tmp_path / "calls" / "flac" / f"{samples[0].utt_id}.flac").stat().st_mtime_ns == stamp  # resumable


# ---------------------------------------------------------------- thresholds on speaker halves

def test_speaker_halves_never_share_a_speaker():
    spk = np.array([f"S{k % 10}" for k in range(200)])
    a, b = speaker_halves(spk, seed=1)
    assert not (a & b).any() and (a | b).all() and len(set(spk[a]) & set(spk[b])) == 0 and len(set(spk[a])) == 5
    assert np.array_equal(speaker_halves(spk, seed=1)[0], a) and not np.array_equal(speaker_halves(spk, seed=2)[0], a)


def test_policy_thresholds_and_rates_follow_the_genuine_quantiles():
    rng = np.random.default_rng(0)
    y = np.array([0] * 4000 + [1] * 4000)
    p = np.concatenate([rng.beta(1, 8, 4000), rng.beta(6, 2, 4000)])
    risk = policy_thresholds(p, y)
    assert risk["medium"] < risk["high"]
    r = threshold_rates(p, y, None, risk)["overall"]
    assert abs(r["block"]["bonafide_flagged"] - 0.01) < 0.002 and abs(r["verify"]["bonafide_flagged"] - 0.10) < 0.002
    assert r["verify"]["spoof_caught"] > r["block"]["spoof_caught"] > 0.5
    with pytest.raises(ValueError, match="too few"):
        policy_thresholds(p[:200], y[:200])


def test_calibrating_on_one_half_holds_on_the_other():
    rng = np.random.default_rng(1)
    n = 12000
    speakers = np.array([f"S{k}" for k in rng.integers(0, 40, n)])
    y = (rng.random(n) < 0.6).astype(int)
    p = np.where(y == 1, rng.beta(6, 2, n), rng.beta(1, 8, n))
    codec = np.where(np.arange(n) % 2 == 0, "gsm", "opus")
    rep = calibrate(p, y, speakers, codec, seed=0, stored={"high": 0.5, "medium": 0.3})
    assert rep["half_a"]["speakers"] == rep["half_b"]["speakers"] == 20
    for k, target in (("block", 0.01), ("verify", 0.10)):  # set on A, so A is on target; B is a fair check
        assert abs(rep["half_a"]["overall"][k]["bonafide_flagged"] - target) < 0.003
        assert abs(rep["half_b"]["overall"][k]["bonafide_flagged"] - target) < 0.03
    assert set(rep["half_b"]["per_group"]) == {"gsm", "opus"} and "half_b_at_stored_thresholds" in rep


def test_wavlm_only_bundle_keeps_the_source_and_loads(tmp_path):
    import torch

    from audiodf.artifacts import load_bundle
    from audiodf.config import Settings
    from audiodf.models.wavlm import WavLMDetector, save_wavlm

    src = tmp_path / "run4"
    src.mkdir()
    save_wavlm(WavLMDetector(finetune_top=1, pretrained=False), src / "wavlm.pt")
    manifest = {"version": "0.1.0", "branches": ["svm", "rcnn", "wavlm"], "feature_versions": {"svm": 2, "rcnn": 1, "wavlm": 1},
                "fusion_weights": {"svm": 0.1, "rcnn": 0.15, "wavlm": 0.75}, "risk": {"high": 0.2, "medium": 0.08},
                "svm_feature_dim": 318, "rcnn_input_shape": [64, 200], "wavlm_backbone": "wavlm_base_plus",
                "wavlm_finetune_top": 1, "segment_seconds": 2.0, "segment_hop_seconds": 1.0, "metrics": {"x": 1}}
    (src / "bundle.json").write_text(json.dumps(manifest))
    before = (src / "bundle.json").read_text()
    out = make_wavlm_bundle(src, tmp_path / "wavlm_only", {"high": 0.3, "medium": 0.05}, {"seed": 0})
    assert (src / "bundle.json").read_text() == before and not (src / "svm.joblib").exists()
    m = json.loads((out / "bundle.json").read_text())
    assert m["branches"] == ["wavlm"] and m["fusion_weights"] == {"wavlm": 1.0} and m["feature_versions"] == {"wavlm": 1}
    assert m["risk"] == {"high": 0.3, "medium": 0.05} and "svm_feature_dim" not in m and "rcnn_input_shape" not in m
    assert m["metrics"]["source_metrics"] == {"x": 1} and m["metrics"]["threshold_calibration"] == {"seed": 0}
    s = Settings()
    s.paths.artifacts_dir, s.wavlm.pretrained = str(out), False
    bundle = load_bundle(s, "cpu")
    assert set(bundle.models) == {"wavlm"} and bundle.weights == {"wavlm": 1.0}
    assert bundle.models["wavlm"](torch.randn(1, 32000) * 0.1).shape == (1,)
    (src / "bundle.json").write_text(json.dumps({**manifest, "branches": ["svm", "rcnn"]}))
    with pytest.raises(ValueError, match="no WavLM"):
        make_wavlm_bundle(src, tmp_path / "x", {"high": 1, "medium": 0.5}, {})


# ---------------------------------------------------------------- paired comparison

def test_eer_and_paired_bootstrap():
    rng = np.random.default_rng(0)
    y = np.array([0, 1] * 400)
    good = np.clip(0.2 + 0.6 * y + rng.normal(0, 0.15, 800), 0, 1)
    bad = np.clip(0.3 + 0.4 * y + rng.normal(0, 0.25, 800), 0, 1)
    assert abs(eer_pct(y, good) - compute_metrics(y, good)["eer_pct"]) < 1e-9  # same definition as everywhere else
    r = paired_bootstrap(y, good, bad, n_boot=300)
    assert r["diff_second_minus_first"] > 0 and r["ci95"][0] > 0 and r["share_first_better"] > 0.99
    same = paired_bootstrap(y, good, good, n_boot=100)
    assert same["diff_second_minus_first"] == 0 and same["ci95"] == [0.0, 0.0]
    close = paired_bootstrap(y, good, good + rng.normal(0, 0.01, 800), n_boot=300)
    assert close["ci95"][0] < 0 < close["ci95"][1]  # a tie is not called a win
