"""Broker-agnostic chunk handling: routes PCM16 chunks to per-call sessions. Kafka plugs in on top."""

from __future__ import annotations

import time
from collections import OrderedDict

from audiodf.data.audio import pcm16_to_float
from audiodf.inference.engine import DetectionEngine, Verdict
from audiodf.inference.session import CallSession
from audiodf.monitoring import metrics


class ChunkProcessor:
    def __init__(self, engine: DetectionEngine):
        self.engine = engine
        self.cfg = engine.settings.stream
        self.sessions: OrderedDict[str, CallSession] = OrderedDict()

    def handle(self, call_id: str, pcm16: bytes, eof: bool = False) -> list[Verdict]:
        """One chunk for one call -> zero or more verdicts. Chunks of a call must arrive in order
        (Kafka guarantees this when messages are keyed by call_id)."""
        session = self.sessions.get(call_id)
        if session is None:
            self._make_room()
            session = self.sessions[call_id] = CallSession(self.engine, call_id)
        self.sessions.move_to_end(call_id)
        metrics.ACTIVE_SESSIONS.set(len(self.sessions))

        out = []
        if pcm16:
            verdict = session.push(pcm16_to_float(pcm16))
            if verdict:
                out.append(verdict)
        if eof:
            verdict = session.finalize()
            if verdict:
                out.append(verdict)
            self.sessions.pop(call_id, None)
            metrics.ACTIVE_SESSIONS.set(len(self.sessions))
        return out

    def evict_idle(self) -> list[str]:
        now = time.monotonic()
        stale = [cid for cid, s in self.sessions.items() if now - s.last_active > self.cfg.session_idle_seconds]
        for cid in stale:
            del self.sessions[cid]
        metrics.ACTIVE_SESSIONS.set(len(self.sessions))
        return stale

    def _make_room(self) -> None:
        while len(self.sessions) >= self.cfg.max_sessions:
            self.sessions.popitem(last=False)
