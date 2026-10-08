import io

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from audiodf.data.audio import float_to_pcm16
from audiodf.inference.session import CallSession
from audiodf.serving.api import create_app
from audiodf.streaming.processor import ChunkProcessor

from conftest import SR, tone


def test_no_verdict_before_first_window(engine):
    s = CallSession(engine, "c")
    assert s.push(tone(1.5)) is None
    assert s.push(tone(0.6)) is not None


def test_verdict_cadence_follows_one_second_hop(engine):
    s = CallSession(engine, "c")
    audio = tone(6, noise=0.05)
    n = sum(s.push(audio[i:i + SR // 2]) is not None for i in range(0, len(audio), SR // 2))
    assert n == 5  # windows complete at 2,3,4,5,6 s


def test_rolling_buffer_and_window_are_bounded(engine, settings):
    s = CallSession(engine, "c")
    audio = tone(30, noise=0.05)
    last = None
    for i in range(0, len(audio), SR // 2):
        last = s.push(audio[i:i + SR // 2]) or last
    assert len(s._buf) <= settings.window_samples + settings.segment_samples
    assert last.segments_scored == 9 and last.final


def test_streaming_equals_offline_single_push(engine):
    audio = tone(8, noise=0.1, seed=3)
    s = CallSession(engine, "c")
    last = None
    for i in range(0, len(audio), SR // 2):
        last = s.push(audio[i:i + SR // 2]) or last
    offline = engine.predict_waveform(audio)
    assert abs(last.fake_probability - offline.fake_probability) < 1e-3


def test_short_clip_scored_on_finalize(engine):
    s = CallSession(engine, "c")
    assert s.push(tone(1.0)) is None
    v = s.finalize()
    assert v is not None and v.final and v.segments_scored == 1
    assert CallSession(engine).finalize() is None


def test_verdict_fields_are_consistent(engine):
    v = engine.predict_waveform(tone(5, noise=0.1))
    assert 0 <= v.fake_probability <= 1
    assert abs(v.fake_probability - (0.7 * v.svm_probability + 0.3 * v.rcnn_probability)) < 1e-3
    assert v.label == ("fake" if v.fake_probability >= 0.5 else "real")
    assert v.action == {"high": "escalate", "medium": "verify", "low": "allow"}[v.risk_level]


def test_chunk_processor_routes_calls_and_ends_on_eof(engine):
    proc = ChunkProcessor(engine)
    a, b = tone(4, 200), tone(4, 400, noise=0.2)
    out = {"a": [], "b": []}
    for i in range(0, len(a), SR // 2):
        out["a"] += proc.handle("a", float_to_pcm16(a[i:i + SR // 2]))
        out["b"] += proc.handle("b", float_to_pcm16(b[i:i + SR // 2]), eof=i + SR // 2 >= len(b))
    assert len(proc.sessions) == 1 and "a" in proc.sessions
    assert all(v.call_id == "b" for v in out["b"]) and out["b"][-1].final
    assert len(out["a"]) == 3  # windows at 2, 3, 4 s


def test_chunk_processor_evicts_idle_and_caps_sessions(engine):
    proc = ChunkProcessor(engine)
    proc.cfg = type(proc.cfg)(session_idle_seconds=-1, max_sessions=2)
    for cid in "abc":
        proc.handle(cid, float_to_pcm16(tone(0.5)))
    assert len(proc.sessions) == 2
    assert len(proc.evict_idle()) == 2 and not proc.sessions


def _wav_bytes(wave):
    buf = io.BytesIO()
    sf.write(buf, wave, SR, format="WAV")
    return buf.getvalue()


def test_api_predict_health_metrics(engine, settings):
    with TestClient(create_app(engine, settings)) as client:
        assert client.get("/health").json()["models_loaded"] is True
        r = client.post("/predict", content=_wav_bytes(tone(4, noise=0.1)))
        assert r.status_code == 200 and r.json()["risk_level"] in {"low", "medium", "high"}
        assert client.post("/predict", content=b"").status_code == 400
        assert client.post("/predict", content=b"not audio").status_code == 400
        assert b"audiodf_verdicts_total" in client.get("/metrics").content


def test_api_degraded_without_models(tmp_path, settings):
    settings = type(settings)()
    settings.paths.artifacts_dir = str(tmp_path)
    with TestClient(create_app(settings=settings)) as client:
        assert client.get("/health").json()["status"] == "degraded"
        assert client.post("/predict", content=b"x").status_code == 503


def test_api_websocket_stream(engine, settings):
    audio = tone(4, noise=0.1)
    with TestClient(create_app(engine, settings)) as client, client.websocket_connect("/stream/call-9") as ws:
        for i in range(0, len(audio), SR // 2):
            ws.send_bytes(float_to_pcm16(audio[i:i + SR // 2]))
        ws.send_text("end")
        verdicts = []
        try:
            while True:
                verdicts.append(ws.receive_json())
        except Exception:
            pass
    assert verdicts and verdicts[-1]["final"] and verdicts[-1]["call_id"] == "call-9"
    assert np.isfinite(verdicts[-1]["fake_probability"])
