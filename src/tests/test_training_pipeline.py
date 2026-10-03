import json

import numpy as np
import pytest
import soundfile as sf

from audiodf.config import Settings
from audiodf.evaluation.stream_eval import StreamScores, rcnn_at_times, summarize, svm_at_times
from audiodf.inference.engine import DetectionEngine
from audiodf.models.svm import CalibratedSvm, build_svm
from audiodf.training.pipeline import operating_points, run_training, tune_fusion

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


def _scores(svm, rcnn, y):
    n = len(y)
    return StreamScores(np.arange(n), y, np.where(y == 1, "A01", "-"), np.full(n, "-"), GRID,
                        np.tile(svm[:, None], (1, 5)), np.tile(rcnn[:, None], (1, 5)))


def test_tune_fusion_prefers_the_informative_branch():
    rng = np.random.default_rng(0)
    y = np.array([0, 1] * 100)
    good = np.clip(y * 0.8 + 0.1 + rng.normal(0, 0.05, 200), 0, 1)
    junk = rng.random(200)
    weight, curve = tune_fusion(_scores(good, junk, y))
    assert weight > 0.5 and len(curve) == 21
    assert curve[f"{weight:.2f}"] <= curve["0.00"]  # no worse than the junk branch alone
    weight, _ = tune_fusion(_scores(junk, good, y))
    assert weight < 0.5
    weight, _ = tune_fusion(_scores(good, good, y))  # equally good branches: ties resolve to balanced
    assert weight == 0.5


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
    rep = summarize(_scores(s, s, y), 0.5)
    assert rep["at_horizon"]["fused"]["eer_pct"] < 5
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


def test_run_training_end_to_end_on_synthetic_asv5(tmp_path):
    _write_split(tmp_path, "T", "flac_T", "ASVspoof5.train.tsv", ["A01", "A02"])
    _write_split(tmp_path, "D", "flac_D", "ASVspoof5.dev.track_1.tsv", ["A09", "A10"])
    s = Settings()
    s.paths.asv5_root = str(tmp_path)
    s.paths.data_root = str(tmp_path / "no_asv19")
    s.paths.cache_dir = str(tmp_path / "cache")
    s.paths.artifacts_dir = str(tmp_path / "artifacts")
    s.paths.results_dir = str(tmp_path / "results")
    s.rcnn_train.epochs, s.rcnn_train.batch_size = 1, 4
    s.data.svm_train_utts = s.data.tune_utts = 24

    report = run_training(s, workers=0, log=lambda *_: None)
    assert (tmp_path / "artifacts" / "svm.joblib").exists() and report["test"] == {}  # no eval audio: skipped
    manifest = json.loads((tmp_path / "artifacts" / "bundle.json").read_text())
    assert manifest["svm_weight"] == report["svm_weight"] and manifest["risk"]["medium"] <= manifest["risk"]["high"]

    manifest.update(svm_weight=0.35, risk={"high": 0.4, "medium": 0.2})  # distinct from every default
    (tmp_path / "artifacts" / "bundle.json").write_text(json.dumps(manifest))
    s2 = Settings()
    s2.paths.artifacts_dir = str(tmp_path / "artifacts")
    engine = DetectionEngine.from_artifacts(s2, "cpu")
    assert (engine.settings.ensemble.svm_weight, engine.settings.risk.high, engine.settings.risk.medium) == (
        0.35, 0.4, 0.2)  # the bundle's tuned operating point wins over the config...
    assert s2.ensemble.svm_weight == 0.7  # ...without mutating the caller's settings
    verdict = engine.predict_waveform(tone(5, noise=0.2, seed=3))
    assert verdict is not None and 0 <= verdict.fake_probability <= 1
    assert abs(verdict.fake_probability - (0.35 * verdict.svm_probability + 0.65 * verdict.rcnn_probability)) < 1e-3


def test_run_training_on_mixed_datasets(tmp_path):
    from test_prepare import _asv19

    _write_split(tmp_path, "T", "flac_T", "ASVspoof5.train.tsv", ["A01", "A02"])
    _write_split(tmp_path, "D", "flac_D", "ASVspoof5.dev.track_1.tsv", ["A09", "A10"])
    s = Settings()
    s.paths.asv5_root = str(tmp_path)
    s.paths.data_root = str(_asv19(tmp_path, n=12))
    s.paths.cache_dir, s.paths.artifacts_dir = str(tmp_path / "cache"), str(tmp_path / "artifacts")
    s.paths.results_dir = str(tmp_path / "results")
    s.rcnn_train.epochs, s.rcnn_train.batch_size = 1, 4
    s.data.svm_train_utts = s.data.tune_utts = 24
    s.data.train_datasets = ("asv5", "asv19")
    lines = []
    report = run_training(s, workers=0, log=lines.append)
    assert any("training clips: 48" in str(line) for line in lines)  # 24 ASV5 + 12 ASV2019 train + 12 dev
    assert any("training on: asv5 + asv19" in str(line) for line in lines)
    assert report["test"] == {} and (tmp_path / "artifacts" / "svm.joblib").exists()
