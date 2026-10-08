"""Whisper-encoder branch: the speech encoder of OpenAI Whisper + the same small head as the WavLM branch, on 2 s windows.

Why: Whisper was trained on ~680,000 hours of noisy, multilingual web audio (WavLM-Base+: 94,000 hours, mostly clean read and
podcast speech), so its features may keep the traces of synthesis that the WavLM branch loses on real phone lines, and two models
trained on different data tend to miss different fakes. It is an experiment (`new_plan.md` 7.3r): it is judged alone and fused with
WavLM against the served model before it goes anywhere.

How: the library's encoder insists on 30 s of padded audio (1,500 positions). A 2 s window is 200 mel frames -> 100 positions, so
this module runs the encoder's own convolutions, position table and layers over just those 100 positions (the table is the fixed
sinusoid of Whisper, so a slice of it is the position of the first 2 s). The log-mel is computed on the GPU exactly as Whisper's
feature extractor does (80 mel bins, 25 ms / 10 ms, log10, clamp 8 dB below the maximum, (x + 4) / 4), so the same raw 16 kHz window
as for WavLM goes in. As in the WavLM branch, the top `finetune_top` layers (and the final layer norm) adapt slowly, the rest stays
frozen, the head mixes all layer outputs with learned weights, pools with attentive statistics and outputs a spoof logit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from audiodf.models.wavlm import AttentiveStatsPool

BACKBONE = "openai/whisper-small"  # encoder: 12 layers, d=768, ~88M (whisper-base: 6 layers, d=512, ~20M)
N_FFT, HOP, N_MELS = 400, 160, 80
FEATURE_VERSION = 1  # the log-mel as above; a change here must bump this (stored in the bundle)


def _mel_filters(n_mels: int = N_MELS) -> torch.Tensor:
    """Whisper's mel filter bank (n_mels, 201), built locally by the feature extractor (no download)."""
    from transformers import WhisperFeatureExtractor

    return torch.from_numpy(np.asarray(WhisperFeatureExtractor(feature_size=n_mels).mel_filters, dtype=np.float32).T.copy())


def log_mel(wave: torch.Tensor, filters: torch.Tensor) -> torch.Tensor:
    """wave: (B, samples) 16 kHz -> (B, 80, samples // 160) Whisper log-mel, in float32 whatever the autocast mode."""
    with torch.autocast(wave.device.type, enabled=False):
        w = wave.float()
        stft = torch.stft(w, N_FFT, HOP, window=torch.hann_window(N_FFT, device=w.device), return_complex=True)
        mel = filters.to(w.device) @ (stft[..., :-1].abs() ** 2)
        spec = torch.clamp(mel, min=1e-10).log10()
        spec = torch.maximum(spec, spec.amax(dim=(1, 2), keepdim=True) - 8.0)
        return (spec + 4.0) / 4.0


class WhisperDetector(nn.Module):
    def __init__(self, backbone: str = BACKBONE, finetune_top: int = 4, hidden: int = 128, dropout: float = 0.2,
                 pretrained: bool = True, encoder_config: dict | None = None):
        """pretrained=False builds the architecture only: `encoder_config` (WhisperConfig fields) sets a tiny one for tests."""
        super().__init__()
        from transformers import WhisperConfig
        from transformers.models.whisper.modeling_whisper import WhisperEncoder

        if pretrained:
            from transformers import WhisperModel

            self.encoder = WhisperModel.from_pretrained(backbone).encoder  # the decoder is dropped, never used
        else:
            self.encoder = WhisperEncoder(WhisperConfig(**(encoder_config or {})))
        self.backbone, self.finetune_top, self.hidden = backbone, finetune_top, hidden
        layers = self.encoder.layers
        dim = self.encoder.config.d_model
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        for module in [*layers[len(layers) - finetune_top:], self.encoder.layer_norm]:
            for p in module.parameters():
                p.requires_grad_(True)
        self.register_buffer("mel_filters", _mel_filters(self.encoder.config.num_mel_bins), persistent=False)
        self.layer_weights = nn.Parameter(torch.zeros(len(layers)))
        self.pool = AttentiveStatsPool(dim, hidden)
        self.head = nn.Sequential(nn.Linear(2 * dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
                                  nn.Linear(hidden, 1))

    def train(self, mode: bool = True):
        super().train(mode)
        layers = self.encoder.layers
        self.encoder.eval()  # no layer-drop or dropout noise in the encoder at all; only the head trains in train mode
        for layer in layers[len(layers) - self.finetune_top:]:
            layer.train(mode)
        return self

    def layer_outputs(self, wave: torch.Tensor) -> list[torch.Tensor]:
        """Output of every encoder layer for 2 s windows, each (B, positions, d); the last one after the final layer norm."""
        enc = self.encoder
        x = torch.nn.functional.gelu(enc.conv1(log_mel(wave, self.mel_filters).to(enc.conv1.weight.dtype)))
        x = torch.nn.functional.gelu(enc.conv2(x)).permute(0, 2, 1)
        x = x + enc.embed_positions(torch.arange(x.shape[1], device=x.device))  # first x.shape[1] positions only
        outs = []
        for layer in enc.layers:
            out = layer(x, None)
            x = out[0] if isinstance(out, tuple) else out  # transformers 5 returns the tensor, 4.x a tuple
            outs.append(x)
        outs[-1] = enc.layer_norm(outs[-1])
        return outs

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        """wave: (B, samples) raw 16 kHz audio -> spoof logits (B,)."""
        feats = self.layer_outputs(wave)
        mix = torch.softmax(self.layer_weights, dim=0)
        x = sum(w * f for w, f in zip(mix, feats))
        return self.head(self.pool(x)).squeeze(1)

    def param_groups(self, head_lr: float, backbone_lr: float) -> list[dict]:
        backbone = [p for p in self.encoder.parameters() if p.requires_grad]
        own = [self.layer_weights, *self.pool.parameters(), *self.head.parameters()]
        return [{"params": backbone, "lr": backbone_lr}, {"params": own, "lr": head_lr}]

    def trainable_state(self) -> dict:
        names = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v for k, v in self.state_dict().items() if k in names}


def save_whisper(model: WhisperDetector, path: str | Path) -> None:
    """Only the trained part (top layers, final norm, layer mix, pooling, head); the frozen rest comes from the pretrained
    backbone when loading. A test-sized encoder (pretrained=False) also stores its config."""
    torch.save({"backbone": model.backbone, "finetune_top": model.finetune_top, "hidden": model.hidden,
                "feature_version": FEATURE_VERSION,
                "encoder_config": model.encoder.config.to_dict(), "state": model.trainable_state()}, path)


def load_whisper(path: str | Path, device: str | torch.device = "cpu", pretrained: bool = True) -> WhisperDetector:
    ckpt = torch.load(path, map_location="cpu")
    if ckpt["feature_version"] != FEATURE_VERSION:
        raise ValueError(f"checkpoint uses Whisper feature version {ckpt['feature_version']}, this code is {FEATURE_VERSION}")
    model = WhisperDetector(ckpt["backbone"], ckpt["finetune_top"], ckpt["hidden"], pretrained=pretrained,
                            encoder_config=None if pretrained else ckpt["encoder_config"])
    missing = {n for n, p in model.named_parameters() if p.requires_grad} - set(ckpt["state"])
    if missing:
        raise ValueError(f"checkpoint lacks trained tensors: {sorted(missing)[:5]}")
    model.load_state_dict(ckpt["state"], strict=False)
    return model.to(device).eval()


@torch.no_grad()
def predict_spoof_proba(model: WhisperDetector, windows, device: str | torch.device = "cpu", batch_size: int = 32):
    """windows: (n, samples) float waveforms -> P(spoof) per window, shape (n,)."""
    device = torch.device(device)
    model.eval()
    out = np.empty(len(windows), dtype=np.float32)
    for i in range(0, len(windows), batch_size):
        x = torch.from_numpy(np.asarray(windows[i:i + batch_size], dtype=np.float32)).to(device)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            out[i:i + batch_size] = torch.sigmoid(model(x).float()).cpu().numpy()
    return out
