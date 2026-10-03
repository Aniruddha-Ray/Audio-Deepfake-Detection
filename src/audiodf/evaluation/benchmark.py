"""Per-stage latency of the serving path on synthetic audio (batch of one, as in live calls)."""

from __future__ import annotations

import time

import numpy as np

from audiodf.inference.engine import DetectionEngine
from audiodf.inference.session import CallSession


def _time_ms(fn, repeat: int = 30) -> float:
    fn()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    return round(1000 * (time.perf_counter() - t0) / repeat, 3)


def benchmark(engine: DetectionEngine) -> dict:
    s = engine.settings
    rng = np.random.default_rng(0)
    window = (rng.standard_normal(s.window_samples) * 0.05).astype(np.float32)
    seg = window[None, :s.segment_samples]
    stats = {"branches": dict(engine.weights)}  # only branches with a non-zero fusion weight run
    if engine.uses_svm:
        svm_x = engine.svm_features.extract(window)[None]
        stats["svm_features_10s_buffer_ms"] = _time_ms(lambda: engine.svm_features.extract(window))
        stats["svm_predict_ms"] = _time_ms(lambda: engine.models["svm"].predict_proba(svm_x))
    if "rcnn" in engine.models:
        stats["rcnn_features_2s_window_ms"] = _time_ms(lambda: engine.rcnn_features.extract(seg))
        stats["rcnn_2s_window_ms"] = _time_ms(lambda: engine.rcnn_probabilities(seg))
    if "wavlm" in engine.models:
        stats["wavlm_2s_window_ms"] = _time_ms(lambda: engine.wavlm_probabilities(seg))
    chunk = window[:s.segment_hop_samples]

    def push_one_hop():
        session = CallSession(engine)
        session.push(window[:s.segment_samples - s.segment_hop_samples])
        session.push(chunk)

    stats["session_first_verdict_ms"] = _time_ms(push_one_hop, repeat=10)
    stats["device"] = str(engine.device)
    stats["note"] = "synthetic noise audio; per-call cost, batch size 1"
    return stats
