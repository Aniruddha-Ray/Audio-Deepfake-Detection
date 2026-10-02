import numpy as np
import pytest
import torch

from audiodf.config import RiskConfig, load_settings
from audiodf.evaluation.metrics import compute_metrics, per_attack_eer
from audiodf.models.ensemble import fuse
from audiodf.models.rcnn import RCNN, load_rcnn, predict_spoof_proba, save_rcnn
from audiodf.risk import RiskEngine


def test_fuse_weighted_average():
    assert fuse(1.0, 0.0, 0.7) == pytest.approx(0.7)
    with pytest.raises(ValueError):
        fuse(0.5, 0.5, 1.5)


@pytest.mark.parametrize("p,level,action", [(0.95, "high", "block"), (0.80, "high", "block"),
                                            (0.65, "medium", "verify"), (0.50, "medium", "verify"),
                                            (0.49, "low", "allow"), (0.0, "low", "allow")])
def test_risk_thresholds(p, level, action):
    d = RiskEngine().classify(p)
    assert (d.level, d.action) == (level, action)


def test_risk_rejects_inverted_thresholds():
    with pytest.raises(ValueError):
        RiskEngine(RiskConfig(high=0.4, medium=0.6))


def test_metrics_perfect_and_random():
    y = np.array([0] * 50 + [1] * 50)
    perfect = compute_metrics(y, y.astype(float))
    assert perfect["eer_pct"] == 0 and perfect["auc"] == 1
    rng = np.random.default_rng(0)
    assert 35 < compute_metrics(y, rng.random(100))["eer_pct"] < 65


def test_per_attack_eer_isolates_attacks():
    y = np.array([0] * 20 + [1] * 20 + [1] * 20)
    attacks = np.array(["-"] * 20 + ["A07"] * 20 + ["A17"] * 20)
    p = np.concatenate([np.full(20, 0.1), np.full(20, 0.9), np.full(20, 0.1)])
    out = per_attack_eer(y, p, attacks)
    assert out["A07"] == 0 and out["A17"] == pytest.approx(50, abs=1)


def test_rcnn_shapes_and_bidirectionality(settings):
    model = RCNN(64, 200, settings.rcnn_model).eval()
    assert model(torch.randn(3, 1, 64, 200)).shape == (3,)
    uni = RCNN(64, 200, type(settings.rcnn_model)(bidirectional=False))
    assert uni.head.in_features * 2 == model.head.in_features


def test_rcnn_checkpoint_roundtrip(tmp_path, settings):
    model = RCNN(64, 200, settings.rcnn_model).eval()
    save_rcnn(model, tmp_path / "m.pt")
    loaded = load_rcnn(tmp_path / "m.pt")
    x = np.random.default_rng(0).standard_normal((4, 64, 200)).astype(np.float32)
    assert np.allclose(predict_spoof_proba(model, x), predict_spoof_proba(loaded, x), atol=1e-6)


def test_yaml_override(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("ensemble:\n  svm_weight: 0.5\nrisk:\n  high: 0.9\n")
    s = load_settings(cfg)
    assert s.ensemble.svm_weight == 0.5 and s.risk.high == 0.9 and s.risk.medium == 0.5
    cfg.write_text("nonsense: 1\n")
    with pytest.raises(KeyError):
        load_settings(cfg)
