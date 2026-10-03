"""Third branch: pretrained WavLM-Base+ front end + small head, on the same 2 s windows as the RCNN.

The top `finetune_top` transformer layers adapt during training; the CNN feature extractor and the lower layers stay
frozen (4.3 GB VRAM, and frozen low layers keep the general speech knowledge the branch is here for). The head mixes
all 12 layer outputs with learned weights (lower layers carry the acoustic detail spoofing tends to leave), pools over
time with attentive statistics, and outputs a spoof logit. Input: raw 16 kHz waveform windows (B, samples); the
torchaudio bundle expects no waveform normalisation.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torchaudio

BACKBONE = "wavlm_base_plus"


class AttentiveStatsPool(nn.Module):
    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B, 2D): attention-weighted mean and standard deviation over time."""
        w = torch.softmax(self.attn(x).float(), dim=1).to(x.dtype)
        mean = (w * x).sum(dim=1)
        var = (w * (x - mean.unsqueeze(1)) ** 2).sum(dim=1)
        return torch.cat([mean, var.clamp_min(1e-6).sqrt()], dim=1)


class WavLMDetector(nn.Module):
    def __init__(self, finetune_top: int = 4, hidden: int = 128, dropout: float = 0.2, pretrained: bool = True):
        super().__init__()
        self.backbone = (torchaudio.pipelines.WAVLM_BASE_PLUS.get_model() if pretrained
                         else torchaudio.models.wavlm_base())  # architecture only (tests)
        self.finetune_top = finetune_top
        layers = self.backbone.encoder.transformer.layers
        dim = self.backbone.encoder.feature_projection.projection.out_features
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        for layer in layers[len(layers) - finetune_top:]:
            for p in layer.parameters():
                p.requires_grad_(True)
        self.layer_weights = nn.Parameter(torch.zeros(len(layers)))
        self.pool = AttentiveStatsPool(dim, hidden)
        self.head = nn.Sequential(nn.Linear(2 * dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
                                  nn.Linear(hidden, 1))

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen layers stay in eval mode (no dropout noise in features that cannot adapt).
        layers = self.backbone.encoder.transformer.layers
        self.backbone.feature_extractor.eval()
        self.backbone.encoder.feature_projection.eval()
        for layer in layers[:len(layers) - self.finetune_top]:
            layer.eval()
        return self

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        """wave: (B, samples) raw 16 kHz audio -> spoof logits (B,)."""
        feats, _ = self.backbone.extract_features(wave)
        mix = torch.softmax(self.layer_weights, dim=0)
        x = sum(w * f for w, f in zip(mix, feats))
        return self.head(self.pool(x)).squeeze(1)

    def param_groups(self, head_lr: float, backbone_lr: float) -> list[dict]:
        """Fine-tuned backbone layers learn slowly; the new layer mix, pooling and head learn fast."""
        backbone = [p for p in self.backbone.parameters() if p.requires_grad]
        own = [self.layer_weights, *self.pool.parameters(), *self.head.parameters()]
        return [{"params": backbone, "lr": backbone_lr}, {"params": own, "lr": head_lr}]

    def trainable_state(self) -> dict:
        return {k: v for k, v in self.state_dict().items() if k in _trainable_names(self)}


def _trainable_names(model: WavLMDetector) -> set[str]:
    return {n for n, p in model.named_parameters() if p.requires_grad} | {
        n for n, _ in model.named_buffers() if not n.startswith("backbone.")}


def save_wavlm(model: WavLMDetector, path: str | Path) -> None:
    """Only the trained part (top layers, layer mix, pooling, head, ~29M values); the frozen rest comes from the
    pretrained bundle when loading."""
    torch.save({"backbone": BACKBONE, "finetune_top": model.finetune_top, "state": model.trainable_state()}, path)


def load_wavlm(path: str | Path, device: str | torch.device = "cpu", pretrained: bool = True) -> WavLMDetector:
    ckpt = torch.load(path, map_location="cpu")
    if ckpt["backbone"] != BACKBONE:
        raise ValueError(f"checkpoint was trained on {ckpt['backbone']}, this code builds {BACKBONE}")
    model = WavLMDetector(finetune_top=ckpt["finetune_top"], pretrained=pretrained)
    missing = _trainable_names(model) - set(ckpt["state"])
    if missing:
        raise ValueError(f"checkpoint lacks trained tensors: {sorted(missing)[:5]}")
    model.load_state_dict(ckpt["state"], strict=False)
    return model.to(device).eval()


@torch.no_grad()
def predict_spoof_proba(model: WavLMDetector, windows, device: str | torch.device = "cpu", batch_size: int = 32):
    """windows: (n, samples) float waveforms -> P(spoof) per window, shape (n,)."""
    import numpy as np

    device = torch.device(device)
    model.eval()
    out = np.empty(len(windows), dtype=np.float32)
    for i in range(0, len(windows), batch_size):
        x = torch.from_numpy(np.asarray(windows[i:i + batch_size], dtype=np.float32)).to(device)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            out[i:i + batch_size] = torch.sigmoid(model(x).float()).cpu().numpy()
    return out
