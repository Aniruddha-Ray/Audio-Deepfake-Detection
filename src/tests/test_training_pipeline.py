import json

import numpy as np
import pytest
import soundfile as sf

from audiodf.config import Settings
from audiodf.evaluation.stream_eval import StreamScores, rcnn_at_times, summarize, svm_at_times
from audiodf.inference.engine import DetectionEngine
from audiodf.models.svm import CalibratedSvm, build_svm
from audiodf.training.pipeline import (build_training_pool, operating_points, run_training, training_splits,
                                       tune_fusion)

from conftest import SR, tone

GRID = (2.0, 4.0, 6.0, 8.0, 10.0)


def test_svm_at_times_uses_longest_prefix_not_exceeding_t():
    utt = np.array([5, 5, 5, 5, 9, 9, 9, 9, 9])
    sec = np.array([2, 4, 6, 7.3, 2, 4, 6, 8, 10])
    p = np.array([.1, .2, .3, .4, .5, .6, .7, .8, .9])
    out = svm_at_times(utt, sec, p, np.array([9, 5]), GRID)  # caller order differs from sorted order
    assert out[1].tolist() == [.1, .2, .3, .4, .4]  # clip 5 has only 7.3 s: later times reuse all of it
    assert out[0].tolist() == [.5, .6, .7, .8, .9]
    first = svm_at_times(np.array([1, 1]), np.array([3.0, 5.0]), np.array([.2, .8]), np.array([1]), (2.0, 4.0))
    assert first[0].tolist() == [.2, .2]  # before any prefix exists, the shortest one stands in


def test_rcnn_at_times_averages_finished_windows():
    upos = np.array([0, 0, 0, 1])
    end_s = np.array([2.0, 3.0, 4.0, 2.0])
    p = np.array([.1, .3, .5, .9])
    out = rcnn_at_times(upos, end_s, p, 2, (2.0, 3.0, 4.0))
    assert np.allclose(out[0], [.1, .2, .3]) and np.allclose(out[1], [.9, .9, .9])
    early = rcnn_at_times(upos, end_s, p, 2, (1.0, 2.0))
    assert np.allclose(early[:, 0], [.1, .9])  # nothing finished yet: the first window


def _scores(y, **branches):
    n = len(y)
    return StreamScores(np.arange(n), y, np.where(y == 1, "A01", "-"), np.full(n, "-"), GRID,
                        {b: np.tile(v[:, None], (1, 5)) for b, v in branches.items()})


def test_tune_fusion_prefers_the_informative_branch():
    rng = np.random.default_rng(0)
    y = np.array([0, 1] * 100)
    good = np.clip(y * 0.8 + 0.1 + rng.normal(0, 0.05, 200), 0, 1)
    junk = rng.random(200)
    weights, curve = tune_fusion(_scores(y, svm=good, rcnn=junk))
    assert weights["svm"] > 0.5 and "svm=0.00 rcnn=1.00" in curve and "svm=1.00 rcnn=0.00" in curve
    assert curve[" ".join(f"{b}={w:.2f}" for b, w in weights.items())] <= curve["svm=0.00 rcnn=1.00"]
    weights, _ = tune_fusion(_scores(y, svm=junk, rcnn=good))
    assert weights["rcnn"] > 0.5
    weights, _ = tune_fusion(_scores(y, svm=good, rcnn=good))  # equally good branches: ties resolve to balanced
    assert weights == {"svm": 0.5, "rcnn": 0.5}


def test_tune_fusion_three_branches():
    rng = np.random.default_rng(3)
    y = np.array([0, 1] * 200)
    good = np.clip(y * 0.6 + 0.2 + rng.normal(0, 0.15, 400), 0, 1)
    junk = rng.random(400)
    weights, curve = tune_fusion(_scores(y, svm=junk, rcnn=junk[::-1].copy(), wavlm=good))
    assert set(weights) == {"svm", "rcnn", "wavlm"} and abs(sum(weights.values()) - 1) < 1e-9
    assert weights["wavlm"] > max(weights["svm"], weights["rcnn"])
    assert len([k for k in curve if "1.00" in k]) == 3  # each branch alone is always reported
    weights, _ = tune_fusion(_scores(y, svm=good, rcnn=good, wavlm=good))
    assert max(weights.values()) - min(weights.values()) <= 0.05  # all equal: as balanced as the grid allows


def test_operating_points_follow_bonafide_quantiles():
    rng = np.random.default_rng(1)
    y = np.array([0] * 1000 + [1] * 1000)
    fused = np.concatenate([rng.beta(1, 8, 1000), rng.beta(8, 1, 1000)])
    high, medium, table = operating_points(fused, y, Settings())
    assert medium < high
    assert abs((fused[y == 0] >= high).mean() - 0.01) < 0.005
    assert abs((fused[y == 0] >= medium).mean() - 0.10) < 0.01
    assert all(0.9 < row["spoof_caught"] <= 1.0 for row in table)


def test_summarize_reports_horizon_curve_and_attacks():
    rng = np.random.default_rng(2)
    y = np.array([0, 1] * 100)
    s = np.clip(y * 0.7 + 0.15 + rng.normal(0, 0.1, 200), 0, 1)
    rep = summarize(_scores(y, svm=s, rcnn=s), {"svm": 0.5, "rcnn": 0.5})
    assert rep["at_horizon"]["fused"]["eer_pct"] < 5 and set(rep["at_horizon"]) == {"svm", "rcnn", "fused"}
    assert set(rep["time_to_decision_eer_pct"]["fused"]) == {"2s", "4s", "6s", "8s", "10s"}
    assert "A01" in rep["per_attack_eer_pct"]["fused"] and "per_codec_eer_pct" not in rep


def test_calibrated_svm_refuses_uncalibrated_use(settings):
    X = np.random.default_rng(0).standard_normal((40, settings.svm_features.dim))
    svm = CalibratedSvm(build_svm(settings.svm_model).fit(X, np.arange(40) % 2))
    with pytest.raises(RuntimeError, match="calibrat"):
        svm.predict_proba(X)
    p = svm.fit_calibrator(X, np.arange(40) % 2).predict_proba(X)
    assert p.shape == (40, 2) and np.allclose(p.sum(axis=1), 1)


def _write_split(root, prefix, audio, proto, attacks, n=24, seconds=5.0):
    (root / audio).mkdir(parents=True, exist_ok=True)
    rows = []
    for k in range(n):
        spoof = k % 2 == 0
        utt = f"{prefix}_{k:010d}"
        speaker = f"{prefix}_{k // 3:04d}"
        rows.append(f"{speaker} {utt} F - - - AC1 {attacks[k // 2 % len(attacks)]} spoof -" if spoof else
                    f"{speaker} {utt} F - - - - bonafide bonafide -")
        wave = tone(seconds, freq=200 + 25 * k, noise=0.02 if spoof else 0.2, seed=k)
        sf.write(root / audio / f"{utt}.flac", np.concatenate([np.zeros(SR // 4, dtype=np.float32), wave]), SR)
    (root / proto).write_text("\n".join(rows) + "\n")


def _settings() -> Settings:
    """Real-codec rendering (on by default) only where this ffmpeg build has the whole codec catalogue: CI's
    Linux imageio-ffmpeg lacks GSM, and rendering refuses to run without it. test_ffmpeg_codecs covers rendering."""
    s = Settings()
    s.data.impairments = False  # run 6's noise corpora are not part of the test data; tested separately
    try:
        from audiodf.data.ffmpeg_codecs import missing_codecs

        s.data.ffmpeg_codecs = not missing_codecs()
    except RuntimeError:  # no ffmpeg at all
        s.data.ffmpeg_codecs = False
    return s


def test_run_training_end_to_end_on_synthetic_asv5(tmp_path):
    _write_split(tmp_path, "T", "flac_T", "ASVspoof5.train.tsv", ["A01", "A02"])
    _write_split(tmp_path, "D", "flac_D", "ASVspoof5.dev.track_1.tsv", ["A09", "A10"])
    s = _settings()
    s.paths.asv5_root = str(tmp_path)
    s.paths.data_root = str(tmp_path / "no_asv19")
    s.paths.cache_dir = str(tmp_path / "cache")
    s.paths.artifacts_dir = str(tmp_path / "artifacts")
    s.paths.results_dir = str(tmp_path / "results")
    s.rcnn_train.epochs, s.rcnn_train.batch_size = 1, 4
    s.data.svm_train_utts = s.data.tune_utts = 24
    s.data.train_splits, s.data.holdout_attacks = ("asv5:train",), ()  # run-1 setup: tune on ASV5 dev
    s.ensemble.branches = ("svm", "rcnn")

    report = run_training(s, workers=0, log=lambda *_: None)
    assert report["tuning_set"] == "ASV5 dev (all attacks)"
    assert (tmp_path / "artifacts" / "svm.joblib").exists() and report["test"] == {}  # no eval audio: skipped
    manifest = json.loads((tmp_path / "artifacts" / "bundle.json").read_text())
    assert manifest["fusion_weights"] == report["fusion_weights"] and manifest["branches"] == ["svm", "rcnn"]
    assert manifest["risk"]["medium"] <= manifest["risk"]["high"]

    # distinct from every default
    manifest.update(fusion_weights={"svm": 0.35, "rcnn": 0.65}, risk={"high": 0.4, "medium": 0.2})
    (tmp_path / "artifacts" / "bundle.json").write_text(json.dumps(manifest))
    s2 = Settings()
    s2.paths.artifacts_dir = str(tmp_path / "artifacts")
    engine = DetectionEngine.from_artifacts(s2, "cpu")
    assert (engine.settings.ensemble.weights, engine.settings.risk.high, engine.settings.risk.medium) == (
        {"svm": 0.35, "rcnn": 0.65}, 0.4, 0.2)  # the bundle's tuned operating point wins over the config...
    assert s2.ensemble.weights == {"wavlm": 1.0}  # ...without mutating the caller's settings (default: WavLM alone)
    verdict = engine.predict_waveform(tone(5, noise=0.2, seed=3))
    assert verdict is not None and 0 <= verdict.fake_probability <= 1
    assert abs(verdict.fake_probability - (0.35 * verdict.svm_probability + 0.65 * verdict.rcnn_probability)) < 1e-3
    assert verdict.wavlm_probability is None

    # a bundle from runs 1-3 (svm_weight, no branch list) still loads; a zero-weight branch is not run
    del manifest["fusion_weights"], manifest["branches"]
    manifest["svm_weight"] = 0.0
    (tmp_path / "artifacts" / "bundle.json").write_text(json.dumps(manifest))
    engine = DetectionEngine.from_artifacts(s2, "cpu")
    assert set(engine.models) == {"rcnn"} and not engine.uses_svm
    verdict = engine.predict_waveform(tone(5, noise=0.2, seed=3))
    assert verdict.svm_probability is None and verdict.fake_probability == verdict.rcnn_probability


def _pooled_settings(tmp_path) -> Settings:
    """Run-2 setup on synthetic data: ASV5 train+dev + ASV2019 train+dev, ASV5 dev attack A10 held out."""
    from test_prepare import _asv19

    _write_split(tmp_path, "T", "flac_T", "ASVspoof5.train.tsv", ["A01", "A02"])
    _write_split(tmp_path, "D", "flac_D", "ASVspoof5.dev.track_1.tsv", ["A09", "A10"])
    s = _settings()
    s.paths.asv5_root = str(tmp_path)
    s.paths.data_root = str(_asv19(tmp_path, n=12))
    s.paths.cache_dir, s.paths.artifacts_dir = str(tmp_path / "cache"), str(tmp_path / "artifacts")
    s.paths.results_dir = str(tmp_path / "results")
    s.rcnn_train.epochs, s.rcnn_train.batch_size = 1, 4
    s.data.svm_train_utts = s.data.tune_utts = 24
    s.data.train_splits = ("asv5:train", "asv5:dev", "asv19:train", "asv19:dev")
    s.data.holdout_attacks, s.data.holdout_speaker_frac = ("asv5:A10",), 0.5
    s.ensemble.branches = ("svm", "rcnn")
    return s


def test_holdout_pool_keeps_tuning_attacks_and_voices_out_of_training(tmp_path):
    s = _pooled_settings(tmp_path)
    train, tune, desc = build_training_pool(s, training_splits(s), None, 0, lambda *_: None)
    assert len(train) + len(tune) <= 72  # 24 + 24 ASV5 + 12 + 12 ASV2019; held-out voices' other clips dropped
    assert "asv5:A10" not in set(train.attack)  # the held-out attack is never trained on...
    assert set(tune.attack[tune.label == 1]) == {"asv5:A10"}  # ...and is the only spoof in the tuning set
    tune_voices = set(tune.speaker[tune.label == 0])
    assert tune_voices and not tune_voices & set(train.speaker)  # tuning bonafide voices are unseen
    assert set(tune.source[tune.label == 0]) == {"asv5:dev"}
    assert not set(zip(train.source, train.utt_id)) & set(zip(tune.source, tune.utt_id))
    assert {"asv5:A01", "asv5:A02", "asv5:A09", "asv19:A01"} <= set(train.attack)  # pool spans both datasets
    assert "held-out attacks asv5:A10" in desc


def test_training_splits_refuse_test_data_and_double_use_of_dev():
    s = Settings()
    s.data.train_splits = ("asv5:train", "asv5:eval")
    with pytest.raises(ValueError, match="eval"):
        training_splits(s)
    s.data.train_splits, s.data.holdout_attacks = ("asv5:train", "asv5:dev"), ()
    with pytest.raises(ValueError, match="tuning set"):
        training_splits(s)
    s.data.holdout_attacks = ("asv5:A12",)
    assert training_splits(s) == [("asv5", "train"), ("asv5", "dev")]


def test_run_training_on_pooled_data_with_held_out_attack(tmp_path):
    s = _pooled_settings(tmp_path)
    lines = []
    report = run_training(s, workers=0, log=lambda m: lines.append(str(m)))
    assert any("training pool: asv5:train, asv5:dev, asv19:train, asv19:dev" in line for line in lines)
    assert report["training_pool"] == list(s.data.train_splits) and "asv5:A10" in report["tuning_set"]
    assert set(report["tuning_subset"]["per_attack_eer_pct"]["fused"]) == {"asv5:A10"}
    assert report["test"] == {} and (tmp_path / "artifacts" / "svm.joblib").exists()
    manifest = json.loads((tmp_path / "artifacts" / "bundle.json").read_text())
    assert manifest["metrics"]["tuning_set"] == report["tuning_set"]

    # resume after an interruption: reuse the saved RCNN, redo every other stage
    saved = tmp_path / "rcnn_backup.pt"
    saved.write_bytes((tmp_path / "artifacts" / "rcnn.pt").read_bytes())
    lines.clear()
    again = run_training(s, workers=0, log=lambda m: lines.append(str(m)), checkpoints={"rcnn": saved})
    assert again["histories"]["rcnn"] == [{"reused_checkpoint": str(saved)}]
    assert any("training skipped" in line for line in lines) and not any("epoch 1/" in line for line in lines)
    assert (tmp_path / "artifacts" / "rcnn.pt").read_bytes() == saved.read_bytes()


def test_add_wavlm_branch_reusing_trained_svm_and_rcnn(tmp_path):
    """Run-4 setup: the SVM and RCNN of an earlier run are reused, only WavLM trains, all three fuse."""
    s = _pooled_settings(tmp_path)
    run_training(s, workers=0, log=lambda *_: None)
    prev = tmp_path / "prev"
    prev.mkdir()
    for f in ("svm.joblib", "rcnn.pt"):
        (prev / f).write_bytes((tmp_path / "artifacts" / f).read_bytes())

    s.ensemble.branches = ("svm", "rcnn", "wavlm")
    s.wavlm.pretrained, s.wavlm.finetune_top, s.wavlm.epochs, s.wavlm.batch_size = False, 1, 2, 4
    s.wavlm.keep_epochs = True
    lines = []
    report = run_training(s, workers=0, log=lambda m: lines.append(str(m)),
                          checkpoints={"svm": prev / "svm.joblib", "rcnn": prev / "rcnn.pt"})
    assert sum("training skipped" in line for line in lines) == 2
    assert len(report["histories"]["wavlm"]) == 2 and "dev_eer_pct" in report["histories"]["wavlm"][0]
    from audiodf.models.wavlm import load_wavlm

    for e in (1, 2):  # every pass kept, loadable, outside the bundle's own files
        assert load_wavlm(tmp_path / "artifacts" / "epochs" / f"wavlm_e{e}.pt", "cpu", False) is not None
    assert set(report["fusion_weights"]) == {"svm", "rcnn", "wavlm"}
    assert set(report["tuning_subset"]["at_horizon"]) == {"svm", "rcnn", "wavlm", "fused"}
    manifest = json.loads((tmp_path / "artifacts" / "bundle.json").read_text())
    assert manifest["branches"] == ["svm", "rcnn", "wavlm"] and manifest["feature_versions"]["wavlm"] == 1
    assert (tmp_path / "artifacts" / "wavlm.pt").exists()

    manifest["fusion_weights"] = {"svm": 0.2, "rcnn": 0.3, "wavlm": 0.5}
    (tmp_path / "artifacts" / "bundle.json").write_text(json.dumps(manifest))
    s2 = Settings()
    s2.paths.artifacts_dir, s2.wavlm.pretrained = str(tmp_path / "artifacts"), False
    engine = DetectionEngine.from_artifacts(s2, "cpu")
    v = engine.predict_waveform(tone(4, noise=0.2, seed=5))
    assert v.wavlm_probability is not None and v.segments_scored == 3
    assert abs(v.fake_probability - (0.2 * v.svm_probability + 0.3 * v.rcnn_probability
                                     + 0.5 * v.wavlm_probability)) < 1e-3


def test_run_training_refuses_unknown_branches(tmp_path):
    s = _pooled_settings(tmp_path)
    s.ensemble.branches = ("svm", "vit")
    with pytest.raises(ValueError, match="branches"):
        run_training(s, workers=0, log=lambda *_: None)
    s.ensemble.branches = ("svm",)
    with pytest.raises(ValueError, match="checkpoints"):
        run_training(s, workers=0, log=lambda *_: None, checkpoints={"rcnn": tmp_path / "x.pt"})
