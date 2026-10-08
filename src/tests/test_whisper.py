import json

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")

from audiodf.config import Settings  # noqa: E402
from audiodf.inference.engine import DetectionEngine  # noqa: E402
from audiodf.models.whisper import (FEATURE_VERSION, WhisperDetector, _mel_filters, load_whisper, log_mel,  # noqa: E402
                                    predict_spoof_proba, save_whisper)
from audiodf.training.pipeline import run_training  # noqa: E402

from conftest import SR, tone  # noqa: E402

TINY = {"d_model": 32, "encoder_layers": 4, "encoder_attention_heads": 4, "encoder_ffn_dim": 64, "num_mel_bins": 80,
        "decoder_layers": 1, "decoder_attention_heads": 4, "decoder_ffn_dim": 64}


def _speechlike(seconds=2.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    return ((0.2 * np.sin(2 * np.pi * 220 * t) + 0.03 * rng.standard_normal(len(t))) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))
            ).astype(np.float32)


def _tiny(seed=0, top=2) -> WhisperDetector:
    torch.manual_seed(seed)
    return WhisperDetector(finetune_top=top, hidden=16, pretrained=False, encoder_config=TINY)


def test_log_mel_is_whispers_own_log_mel_of_the_window():
    from transformers import WhisperFeatureExtractor

    x = _speechlike(2.0)
    ours = log_mel(torch.from_numpy(x)[None], _mel_filters())[0].numpy()
    ref = WhisperFeatureExtractor(feature_size=80)(x, sampling_rate=SR, padding=False, return_tensors="np")["input_features"][0]
    assert ours.shape == ref.shape == (80, 200)
    assert np.abs(ours - ref).max() < 1e-4  # same filters, window, log, clamp and scaling as the feature extractor
    loud = log_mel(torch.from_numpy(x * 8)[None], _mel_filters())[0]
    assert float(loud.max()) > float(torch.from_numpy(ours).max())  # not a per-window normalisation of the level


def test_encoder_runs_on_two_second_windows_without_the_thirty_second_pad():
    model = _tiny().eval()
    outs = model.layer_outputs(torch.from_numpy(np.stack([_speechlike(2.0, s) for s in range(3)])))
    assert len(outs) == 4 and all(o.shape == (3, 100, 32) for o in outs)  # 200 mel frames -> 100 positions
    with torch.no_grad():
        logits = model(torch.from_numpy(np.stack([_speechlike(2.0, s) for s in range(3)])))
    assert logits.shape == (3,) and torch.isfinite(logits).all()
    with pytest.raises(ValueError, match="3000"):  # the library's own forward is the one that insists on 30 s
        model.encoder(torch.zeros(1, 80, 200))


def test_only_the_top_layers_final_norm_mix_pool_and_head_train():
    model = _tiny(top=2)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert any(n.startswith("encoder.layers.2.") for n in trainable) and any(n.startswith("encoder.layers.3.") for n in trainable)
    assert not any(n.startswith(("encoder.layers.0.", "encoder.layers.1.", "encoder.conv", "encoder.embed")) for n in trainable)
    assert "encoder.layer_norm.weight" in trainable and "layer_weights" in trainable
    groups = model.param_groups(1e-3, 2e-5)
    assert sum(len(g["params"]) for g in groups) == len(trainable) and groups[0]["lr"] < groups[1]["lr"]
    loss = model(torch.from_numpy(_speechlike(2.0)[None])).sum()
    loss.backward()
    top = model.encoder.layers[3].fc1.weight.grad
    assert top is not None and float(top.abs().sum()) > 0 and model.encoder.layers[0].fc1.weight.grad is None
    model.train()
    assert not model.encoder.layers[0].training and model.encoder.layers[3].training and model.head.training


def test_save_keeps_only_the_trained_part_and_load_reproduces_the_model(tmp_path):
    model = _tiny().eval()
    path = tmp_path / "whisper.pt"
    save_whisper(model, path)
    ckpt = torch.load(path)
    assert ckpt["feature_version"] == FEATURE_VERSION and set(ckpt["state"]) == {
        n for n, p in model.named_parameters() if p.requires_grad}
    windows = np.stack([_speechlike(2.0, s) for s in range(4)])
    torch.manual_seed(0)  # the frozen part of a tiny test encoder is re-created from the seed, as for the WavLM test model
    again = load_whisper(path, "cpu", pretrained=False)
    assert np.allclose(predict_spoof_proba(model, windows), predict_spoof_proba(again, windows), atol=1e-5)
    ckpt["feature_version"] = FEATURE_VERSION + 1
    torch.save(ckpt, tmp_path / "stale.pt")
    with pytest.raises(ValueError, match="feature version"):
        load_whisper(tmp_path / "stale.pt", "cpu", pretrained=False)


def _whisper_settings(tmp_path, branches):
    from test_training_pipeline import _pooled_settings

    s = _pooled_settings(tmp_path)
    s.ensemble.branches = branches
    s.whisper.pretrained, s.whisper.encoder_config = False, dict(TINY)
    s.whisper.finetune_top, s.whisper.hidden, s.whisper.epochs, s.whisper.batch_size = 1, 16, 2, 4
    s.whisper.keep_epochs = True
    s.wavlm.pretrained, s.wavlm.finetune_top, s.wavlm.epochs, s.wavlm.batch_size = False, 1, 1, 4
    return s


def test_whisper_branch_trains_saves_every_pass_fuses_with_wavlm_and_serves(tmp_path):
    s = _whisper_settings(tmp_path, ("wavlm", "whisper"))
    lines = []
    report = run_training(s, workers=0, log=lambda m: lines.append(str(m)))
    assert len(report["histories"]["whisper"]) == 2 and set(report["fusion_weights"]) == {"wavlm", "whisper"}
    assert any("Whisper encoder" in line for line in lines)
    arts = tmp_path / "artifacts"
    assert (arts / "whisper.pt").exists() and {p.name for p in (arts / "epochs").glob("whisper_e*.pt")} == {
        "whisper_e1.pt", "whisper_e2.pt"}
    manifest = json.loads((arts / "bundle.json").read_text())
    assert manifest["branches"] == ["wavlm", "whisper"] and manifest["feature_versions"]["whisper"] == FEATURE_VERSION
    assert set(report["tuning_subset"]["at_horizon"]) == {"wavlm", "whisper", "fused"}
    engine = DetectionEngine.from_artifacts(s, "cpu")
    v = engine.predict_waveform(tone(4, noise=0.2, seed=5))
    w = engine.weights
    assert v.whisper_probability is not None and v.wavlm_probability is not None
    assert abs(v.fake_probability - (w["wavlm"] * v.wavlm_probability + w["whisper"] * v.whisper_probability)) < 1e-3
    assert "whisper_probability" in v.to_dict()


def test_a_trained_whisper_can_be_reused_next_to_a_new_wavlm(tmp_path):
    s = _whisper_settings(tmp_path, ("whisper",))
    run_training(s, workers=0, log=lambda *_: None)
    prev = tmp_path / "prev_whisper.pt"
    prev.write_bytes((tmp_path / "artifacts" / "whisper.pt").read_bytes())
    s.ensemble.branches = ("wavlm", "whisper")
    lines = []
    report = run_training(s, workers=0, log=lambda m: lines.append(str(m)), checkpoints={"whisper": prev})
    assert sum("training skipped" in line for line in lines) == 1 and "reused_checkpoint" in report["histories"]["whisper"][0]
    assert len(report["histories"]["wavlm"]) == 1
