"""Deep branch: CNN front-end feeding a (Bi)LSTM, trained end to end on 2 s Log-Mel windows.

The BiLSTM only looks across one fixed window, so the look-ahead is bounded by the 2 s the
streaming buffer already waits for; it never needs the future of an unbounded call.
Layer names (cnn / rnn / head) are kept stable so older checkpoints still load.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from audiodf.config import RcnnModelConfig


def _conv_block(c_in: int, c_out: int, pool: tuple[int, int]) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out),
                         nn.ReLU(inplace=True), nn.MaxPool2d(pool))


class RCNN(nn.Module):
    def __init__(self, n_mels: int = 64, n_frames: int = 200, cfg: RcnnModelConfig | None = None):
        super().__init__()
        cfg = cfg or RcnnModelConfig()
        self.cfg, self.input_shape = cfg, (n_mels, n_frames)
        self.cnn = nn.Sequential(_conv_block(1, 32, (2, 2)), _conv_block(32, 64, (2, 2)),
                                 _conv_block(64, 128, (2, 1)), nn.Dropout(cfg.dropout))
        with torch.no_grad():
            _, channels, freq, _ = self.cnn.eval()(torch.zeros(1, 1, n_mels, n_frames)).shape
        self.rnn = nn.LSTM(channels * freq, cfg.hidden, batch_first=True, bidirectional=cfg.bidirectional)
        self.head = nn.Linear(cfg.hidden * (2 if cfg.bidirectional else 1), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 1, n_mels, n_frames) -> spoof logits (batch,)."""
        z = self.cnn(x)
        b, c, f, t = z.shape
        out, _ = self.rnn(z.permute(0, 3, 1, 2).reshape(b, t, c * f))
        return self.head(out.mean(dim=1)).squeeze(1)


@torch.no_grad()
def predict_spoof_proba(model: RCNN, specs, device: str | torch.device = "cpu", batch_size: int = 512):
    """specs: (n, n_mels, n_frames) array or memmap -> P(spoof) per window, shape (n,)."""
    import numpy as np

    device = torch.device(device)
    model.eval()
    out = np.empty(len(specs), dtype=np.float32)
    for i in range(0, len(specs), batch_size):
        x = torch.from_numpy(np.asarray(specs[i:i + batch_size], dtype=np.float32)).unsqueeze(1).to(device)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            out[i:i + batch_size] = torch.sigmoid(model(x).float()).cpu().numpy()
    return out


def save_rcnn(model: RCNN, path: str | Path) -> None:
    torch.save({"state_dict": model.state_dict(), "n_mels": model.input_shape[0],
                "n_frames": model.input_shape[1], "model": vars(model.cfg)}, path)


def load_rcnn(path: str | Path, device: str | torch.device = "cpu", n_mels: int = 64,
              n_frames: int = 200, cfg: RcnnModelConfig | None = None) -> RCNN:
    """Loads a full checkpoint; a bare state_dict (older runs) needs matching n_mels/n_frames/cfg."""
    ckpt = torch.load(path, map_location=device)
    if "state_dict" in ckpt:
        n_mels, n_frames = ckpt["n_mels"], ckpt["n_frames"]
        cfg = RcnnModelConfig(**ckpt["model"])
        state = ckpt["state_dict"]
    else:
        state = ckpt
    model = RCNN(n_mels, n_frames, cfg)
    model.load_state_dict(state)
    return model.to(device).eval()
