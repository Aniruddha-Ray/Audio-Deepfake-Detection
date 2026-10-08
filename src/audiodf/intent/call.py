"""One recording -> transcript (local), voice verdict, intent, combined action."""

from __future__ import annotations

import numpy as np

from audiodf.intent.llm_intent import analyse_with_llm
from audiodf.intent.policy import combine


def analyse_call(engine, wave: np.ndarray, provider: str = "template", language: str | None = None, transcribe=None) -> dict | None:
    """transcribe: injectable for tests; defaults to local Whisper (audiodf.intent.asr)."""
    verdict = engine.predict_waveform(wave)
    if verdict is None:
        return None
    if transcribe is None:
        from audiodf.intent.asr import transcribe
    segments = transcribe(wave, engine.settings.audio.sample_rate, language=language)
    intent = analyse_with_llm([t for _, t in segments], provider)
    # timestamps for the rules' evidence come from the segments (the LLM path works on text only)
    from audiodf.intent.rules import find

    first = {}
    for m in find(segments):
        first.setdefault(m.category, m.start_s)
    for e in intent["evidence"]:
        if e.get("start_s") is None:
            e["start_s"] = first.get(e["category"])
    return {"verdict": verdict.to_dict(), "transcript": [{"start_s": round(s, 1), "text": t} for s, t in segments],
            "intent": intent, "decision": combine(verdict.action, intent["level"])}


def text_intent(dialogue_or_turns, provider: str = "template") -> dict:
    """Intent of a written transcript ('caller: ... receiver: ...' or a list of caller turns), for demos without audio."""
    from audiodf.intent.rules import caller_turns

    turns = caller_turns(dialogue_or_turns) if isinstance(dialogue_or_turns, str) else list(dialogue_or_turns)
    return analyse_with_llm(turns, provider)
