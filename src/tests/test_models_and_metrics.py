import numpy as np
import pytest
import torch

from audiodf.config import RiskConfig, load_settings
from audiodf.evaluation.metrics import compute_metrics, per_attack_eer
from audiodf.models.ensemble import fuse, simplex_grid, weights_from_manifest
from audiodf.models.rcnn import RCNN, load_rcnn, predict_spoof_proba, save_rcnn
from audiodf.risk import RiskEngine


def test_fuse_weighted_average():
    assert fuse({"svm": 1.0, "rcnn": 0.0}, {"svm": 0.7, "rcnn": 0.3}) == pytest.approx(0.7)
    p = {"svm": np.array([1.0, 0.0]), "rcnn": np.array([0.0, 0.0]), "wavlm": np.array([0.5, 1.0])}
    assert np.allclose(fuse(p, {"svm": 0.2, "rcnn": 0.3, "wavlm": 0.5}), [0.45, 0.5])
    assert fuse({"rcnn": 0.4}, {"svm": 0.0, "rcnn": 1.0}) == pytest.approx(0.4)  # zero-weight branch may be absent
    with pytest.raises(ValueError):
        fuse({"svm": 0.5, "rcnn": 0.5}, {"svm": 1.5, "rcnn": -0.5})
    with pytest.raises(ValueError):
        fuse({"svm": 0.5, "rcnn": 0.5}, {"svm": 0.5, "rcnn": 0.6})
    with pytest.raises(ValueError, match="no scores"):
        fuse({"svm": 0.5}, {"svm": 0.5, "wavlm": 0.5})


def test_simplex_grid_and_old_manifests():
    grid = simplex_grid(("svm", "rcnn", "wavlm"), 0.05)
    assert len(grid) == 231 and all(abs(sum(w.values()) - 1) < 1e-9 for w in grid)
    assert {"svm": 0.0, "rcnn": 0.0, "wavlm": 1.0} in grid and len(simplex_grid(("svm", "rcnn"), 0.05)) == 21
    assert weights_from_manifest({"svm_weight": 0.3}) == {"svm": 0.3, "rcnn": 0.7}  # runs 1-3 bundles
    assert weights_from_manifest({"fusion_weights": {"rcnn": 0.4, "wavlm": 0.6}}) == {"rcnn": 0.4, "wavlm": 0.6}


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


@pytest.fixture(scope="module")
def wavlm_model():
    from audiodf.models.wavlm import WavLMDetector

    torch.manual_seed(0)
    return WavLMDetector(finetune_top=2, pretrained=False).eval()  # architecture only: no weight download


def test_wavlm_freezes_all_but_top_layers_and_head(wavlm_model):
    layers = wavlm_model.backbone.encoder.transformer.layers
    assert len(layers) == 12 and wavlm_model(torch.randn(2, 32000) * 0.1).shape == (2,)
    assert all(p.requires_grad for layer in layers[10:] for p in layer.parameters())
    assert not any(p.requires_grad for layer in layers[:10] for p in layer.parameters())
    assert not any(p.requires_grad for p in wavlm_model.backbone.feature_extractor.parameters())
    backbone, own = wavlm_model.param_groups(1e-3, 2e-5)
    assert (backbone["lr"], own["lr"]) == (2e-5, 1e-3)
    trainable = {id(p) for p in wavlm_model.parameters() if p.requires_grad}
    assert {id(p) for p in backbone["params"] + own["params"]} == trainable  # every trained tensor has an LR
    wavlm_model.train()
    assert not layers[0].training and layers[11].training and wavlm_model.head.training  # frozen part stays eval
    wavlm_model.eval()


def test_wavlm_checkpoint_holds_only_trained_part(tmp_path, wavlm_model):
    from audiodf.models.wavlm import load_wavlm, predict_spoof_proba as wavlm_predict, save_wavlm

    save_wavlm(wavlm_model, tmp_path / "w.pt")
    state = torch.load(tmp_path / "w.pt")["state"]
    assert not any(k.startswith("backbone.feature_extractor") or ".layers.0." in k for k in state)
    # the frozen part comes from the backbone builder; with pretrained=False both models must share it
    torch.manual_seed(0)
    loaded = load_wavlm(tmp_path / "w.pt", pretrained=False)
    x = (np.random.default_rng(0).standard_normal((3, 32000)) * 0.1).astype(np.float32)
    assert np.allclose(wavlm_predict(wavlm_model, x), wavlm_predict(loaded, x), atol=1e-5)
    bad = torch.load(tmp_path / "w.pt")
    bad["state"] = {k: v for k, v in bad["state"].items() if not k.startswith("head.")}
    torch.save(bad, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="lacks"):
        load_wavlm(tmp_path / "bad.pt", pretrained=False)


def test_yaml_override(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("ensemble:\n  branches: [rcnn, wavlm]\n  weights: {rcnn: 0.5, wavlm: 0.5}\n"
                   "risk:\n  high: 0.9\n")
    s = load_settings(cfg)
    assert s.ensemble.branches == ("rcnn", "wavlm") and s.ensemble.weights == {"rcnn": 0.5, "wavlm": 0.5}
    assert s.risk.high == 0.9 and s.risk.medium == 0.5
    cfg.write_text("nonsense: 1\n")
    with pytest.raises(KeyError):
        load_settings(cfg)
