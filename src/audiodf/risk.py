"""Risk engine: fused P(spoof) -> risk level and call action."""

from __future__ import annotations

from dataclasses import dataclass

from audiodf.config import RiskConfig

ACTIONS = {"high": "block", "medium": "verify", "low": "allow"}


@dataclass(frozen=True)
class RiskDecision:
    level: str
    action: str


class RiskEngine:
    def __init__(self, cfg: RiskConfig | None = None):
        cfg = cfg or RiskConfig()
        if not 0.0 <= cfg.medium <= cfg.high <= 1.0:
            raise ValueError("require 0 <= medium <= high <= 1")
        self.high, self.medium = cfg.high, cfg.medium

    def classify(self, fake_probability: float) -> RiskDecision:
        level = "high" if fake_probability >= self.high else "medium" if fake_probability >= self.medium else "low"
        return RiskDecision(level, ACTIONS[level])
