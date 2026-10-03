"""Per-call streaming state: VAD gate, rolling audio buffer, 2 s RCNN windows on a 1 s hop,
whole-buffer SVM."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import replace

import numpy as np

from audiodf.data.audio import fill_digital_silence
from audiodf.data.segmenter import pad_to_length
from audiodf.data.vad import frame_db
from audiodf.inference.engine import DetectionEngine, Verdict


class CallSession:
    """Feed audio with push(); a Verdict is returned each time at least one new window completes.

    Audio before speech starts is skipped (same VAD rule training uses to trim clips), so all
    timings below count from speech onset. RCNN: every completed 2 s window (hop 1 s) is scored
    once; the verdict averages the windows inside the last `window_seconds` (10 s). SVM: scores the
    most recent `window_seconds` of audio as one vector, recomputed per verdict.
    """

    def __init__(self, engine: DetectionEngine, call_id: str | None = None):
        s = engine.settings
        self.engine, self.call_id = engine, call_id
        self.sr = s.audio.sample_rate
        self.seg, self.hop, self.window = s.segment_samples, s.segment_hop_samples, s.window_samples
        self._vad = s.vad
        self._vad_frame = int(s.vad.frame_seconds * self.sr)
        self._vad_margin = int(s.vad.margin_seconds * self.sr)
        self._pre = np.zeros(0, dtype=np.float32)  # audio before speech onset
        self._speech = False
        self.skipped_seconds = 0.0
        self._buf = np.zeros(0, dtype=np.float32)
        self._buf_start = 0  # absolute index of _buf[0]
        self._total = 0
        self._next_start = 0  # absolute start of the next RCNN window
        self._scores: deque[float] = deque(maxlen=(self.window - self.seg) // self.hop + 1)
        self._last: Verdict | None = None
        self._last_total = -1
        self.last_active = time.monotonic()

    @property
    def audio_seconds(self) -> float:
        return self._total / self.sr

    def push(self, samples: np.ndarray) -> Verdict | None:
        t0 = time.perf_counter()
        samples = fill_digital_silence(np.asarray(samples, dtype=np.float32), seed=self._total)
        self.last_active = time.monotonic()
        if not self._speech:
            samples = self._gate(samples)
            if not len(samples):
                return None
        self._buf = np.concatenate([self._buf, samples])
        self._total += len(samples)

        windows = []
        while self._next_start + self.seg <= self._total:
            lo = self._next_start - self._buf_start
            windows.append(self._buf[lo:lo + self.seg])
            self._next_start += self.hop
        verdict = None
        if windows:
            self._scores.extend(self.engine.rcnn_probabilities(np.stack(windows)).tolist())
            verdict = self._verdict(final=self.audio_seconds >= self.engine.settings.stream.window_seconds,
                                    t0=t0)
        self._trim()
        return verdict

    def finalize(self) -> Verdict | None:
        """End of call/recording. Scores clips shorter than one window by repeat-padding, as in training."""
        if self._total == 0:
            return None
        t0 = time.perf_counter()
        if not self._scores:
            padded = pad_to_length(self._buf[-self.window:], self.seg)
            self._scores.extend(self.engine.rcnn_probabilities(padded[None, :self.seg]).tolist())
        elif self._last is not None and self._last_total == self._total:
            return replace(self._last, final=True)
        return self._verdict(final=True, t0=t0)

    def _verdict(self, final: bool, t0: float) -> Verdict:
        recent = self._buf[-self.window:]
        if len(recent) < self.seg:
            recent = pad_to_length(recent, self.seg)
        svm_p = self.engine.svm_probability(recent)
        verdict = self.engine.make_verdict(svm_p, list(self._scores), self.audio_seconds, final,
                                           (time.perf_counter() - t0) * 1000, self.call_id)
        self._last, self._last_total = verdict, self._total
        return verdict

    def _gate(self, samples: np.ndarray) -> np.ndarray:
        """Hold audio until the first loud frame; release it from (onset - margin) onwards."""
        self._pre = np.concatenate([self._pre, samples])
        loud = np.nonzero(frame_db(self._pre, self._vad_frame) > self._vad.threshold_db)[0]
        if len(loud):
            start = max(0, loud[0] * self._vad_frame - self._vad_margin)
            self._speech = True
            self.skipped_seconds += start / self.sr
            released, self._pre = self._pre[start:], np.zeros(0, dtype=np.float32)
            return released
        # still silence: keep only the tail the margin may need, dropping whole frames to stay aligned
        drop = (len(self._pre) - self._vad_margin - self._vad_frame) // self._vad_frame * self._vad_frame
        if drop > 0:
            self._pre = self._pre[drop:]
            self.skipped_seconds += drop / self.sr
        return self._pre[:0]

    def _trim(self) -> None:
        keep_from = max(self._buf_start, min(self._next_start, self._total - self.window))
        drop = keep_from - self._buf_start
        if drop > 0:
            self._buf = self._buf[drop:]
            self._buf_start = keep_from
