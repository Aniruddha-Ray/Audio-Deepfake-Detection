"""FastAPI service.

POST /predict          raw audio file bytes (wav/flac/ogg) -> Verdict
WS   /stream/{call_id} binary frames of 16 kHz mono PCM16 -> JSON Verdict per completed window;
                       send the text frame "end" to get the final verdict
GET  /health, GET /metrics
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool

from audiodf.config import Settings, load_settings
from audiodf.data.audio import load_audio, pcm16_to_float
from audiodf.inference.engine import DetectionEngine
from audiodf.inference.session import CallSession
from audiodf.monitoring import metrics

MAX_UPLOAD_BYTES = 50 * 1024 * 1024


def create_app(engine: DetectionEngine | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = engine
        if app.state.engine is None:
            try:
                app.state.engine = DetectionEngine.from_artifacts(settings)
            except (FileNotFoundError, ValueError) as exc:
                app.state.load_error = str(exc)
        yield

    app = FastAPI(title="Real-time deepfake voice detection", lifespan=lifespan)
    app.state.load_error = None

    def get_engine() -> DetectionEngine:
        if app.state.engine is None:
            raise HTTPException(503, f"models not loaded: {app.state.load_error}")
        return app.state.engine

    @app.get("/health")
    def health():
        return {"status": "ok" if app.state.engine else "degraded", "models_loaded": app.state.engine is not None,
                "error": app.state.load_error}

    @app.get("/metrics")
    def prometheus_metrics():
        return Response(metrics.render(), media_type="text/plain; version=0.0.4")

    @app.post("/predict")
    async def predict(request: Request):
        eng = get_engine()
        body = await request.body()
        if not body:
            raise HTTPException(400, "empty body; send raw audio bytes")
        if len(body) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "audio too large")
        try:
            wave = await run_in_threadpool(load_audio, body, settings.audio.sample_rate)
        except Exception as exc:
            metrics.ERRORS.labels("http").inc()
            raise HTTPException(400, f"could not decode audio: {exc}") from exc
        if len(wave) < settings.audio.sample_rate // 2:
            raise HTTPException(400, "audio shorter than 0.5 s")
        verdict = await run_in_threadpool(eng.predict_waveform, wave)
        metrics.record_verdict(verdict, "http")
        return verdict.to_dict()

    @app.websocket("/stream/{call_id}")
    async def stream(ws: WebSocket, call_id: str):
        eng = app.state.engine
        if eng is None:
            await ws.close(code=1013)
            return
        await ws.accept()
        session = CallSession(eng, call_id)
        metrics.ACTIVE_SESSIONS.inc()
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    verdict = await run_in_threadpool(session.push, pcm16_to_float(msg["bytes"]))
                elif (msg.get("text") or "").strip().lower() == "end":
                    verdict = await run_in_threadpool(session.finalize)
                    if verdict:
                        metrics.record_verdict(verdict, "ws")
                        await ws.send_json(verdict.to_dict())
                    break
                else:
                    continue
                if verdict:
                    metrics.record_verdict(verdict, "ws")
                    await ws.send_json(verdict.to_dict())
        except WebSocketDisconnect:
            pass
        finally:
            metrics.ACTIVE_SESSIONS.dec()
            try:
                await ws.close()
            except RuntimeError:
                pass

    return app


def get_app() -> FastAPI:
    """uvicorn factory: `uvicorn audiodf.serving.api:get_app --factory`."""
    return create_app()
