"""Prometheus metrics (scraped at /metrics on the API, or from the Kafka consumer's own port)."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from audiodf.inference.engine import Verdict

REGISTRY = CollectorRegistry()

VERDICTS = Counter("audiodf_verdicts_total", "Verdicts emitted", ["risk_level", "source"], registry=REGISTRY)
FAKE_PROBABILITY = Histogram("audiodf_fake_probability", "Fused P(fake) of emitted verdicts",
                             buckets=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
                             registry=REGISTRY)
SCORING_SECONDS = Histogram("audiodf_scoring_seconds", "Time to produce one verdict",
                            buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5], registry=REGISTRY)
ACTIVE_SESSIONS = Gauge("audiodf_active_sessions", "Calls currently buffered", registry=REGISTRY)
ERRORS = Counter("audiodf_errors_total", "Failures while handling audio", ["source"], registry=REGISTRY)


def record_verdict(verdict: Verdict, source: str) -> None:
    VERDICTS.labels(verdict.risk_level, source).inc()
    FAKE_PROBABILITY.observe(verdict.fake_probability)
    SCORING_SECONDS.observe(verdict.latency_ms / 1000)


def render() -> bytes:
    return generate_latest(REGISTRY)
