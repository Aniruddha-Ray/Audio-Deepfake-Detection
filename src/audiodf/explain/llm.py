"""Plain-language explanation of a verdict from its facts record: a free-tier LLM through an OpenAI-compatible API, or a fixed template.

Rules (the explanation never decides anything):
- the LLM sees only the facts record (numbers and labels; no audio, no transcript, no personal data);
- every number in its answer must match a number in the facts, otherwise the answer is discarded and the template is used;
- any failure (no key, network, rate limit, bad JSON) falls back to the template, so a demo always gets an explanation.

Providers (free tiers; model names change, so each can be overridden with AUDIODF_LLM_MODEL): groq, gemini, openrouter. The key is read from
AUDIODF_LLM_API_KEY and is never written anywhere.
"""

from __future__ import annotations

import json
import os
import re

PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "llama-3.3-70b-versatile"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-2.0-flash"),
    "openrouter": ("https://openrouter.ai/api/v1", "meta-llama/llama-3.3-70b-instruct:free"),
}

SYSTEM = (
    "You explain the decision of a deepfake-voice detector to a bank fraud analyst. The decision is already made and final; never suggest "
    "changing it. Use ONLY the facts in the JSON you are given: do not add facts, numbers, causes or advice that are not in them. Region "
    "contributions are SHAP values: positive means that part of the audio pushed the score towards 'synthetic'. If the attribution check says "
    "the attribution is not faithful, say the regions are unreliable. Write plain English for a non-specialist. Answer with JSON only: "
    '{"summary": one sentence, "reasons": [2 to 4 short sentences], "caveats": [1 or 2 short sentences]}.')


def _numbers(obj) -> list[float]:
    out = []
    if isinstance(obj, dict):
        for v in obj.values():
            out += _numbers(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out += _numbers(v)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out.append(float(obj))
    elif isinstance(obj, str):
        out += [float(m) for m in re.findall(r"-?\d+(?:\.\d+)?", obj)]
    return out


def numbers_supported(text: str, facts: dict) -> tuple[bool, list[float]]:
    """Every number in `text` must equal a number of the facts, allowing rounding and the same value written as a percentage."""
    allowed = _numbers(facts)
    allowed += [100 * v for v in allowed if abs(v) <= 1.0]
    bad = []
    for x in (float(m) for m in re.findall(r"-?\d+(?:\.\d+)?", text)):
        if not any(abs(x - v) <= max(0.051, 0.006 * abs(v)) or (abs(v) >= 1 and abs(x - v) <= 0.5) for v in allowed):
            bad.append(x)
    return not bad, bad


def template(facts: dict) -> dict:
    """The fixed-text explanation: always available, built from the facts only."""
    v = facts["verdict"]
    level = ("at or above the escalation level" if v["action"] == "escalate" else "at or above the verify level" if v["action"] == "verify"
             else "below the verify level")
    p = v["fake_probability"]
    summary = (f"The call scored {p:.4f} on the synthetic-voice scale (0 = genuine, 1 = synthetic), {level} "
               f"(verify {v['verify_threshold']:.4f}, escalate {v['escalate_threshold']:.2f}), so the action is '{v['action']}'.")
    reasons = []
    top = [r for r in facts["regions_top"] if r["contribution"] > 0][:2]
    if top:
        reasons.append("The parts of the audio that pushed the score most towards synthetic: "
                       + "; ".join(f"{r['region']} ({r['contribution']:+.3f})" for r in top) + ".")
    w = max(facts["windows"], key=lambda x: x["p_fake"])
    reasons.append(f"The highest-scoring 2-second window was {w['start_s']:.0f}-{w['end_s']:.0f} s ({w['p_fake']:.4f}).")
    ch = facts.get("channel")
    if ch and "likely" in (ch["room_echo"], ch["background_noise"]):
        found = [n for n, k in (("room echo", "room_echo"), ("background noise", "background_noise")) if ch[k] == "likely"]
        reasons.append(f"The call likely has {' and '.join(found)} (estimated from the audio).")
    caveats = ["The regions show which parts of the audio drove the score, not why the voice is synthetic."]
    chk = facts["attribution_check"]
    if chk.get("checked") and not chk.get("faithful"):
        caveats.append("The attribution check failed for this call: treat the regions as unreliable.")
    return {"summary": summary, "reasons": reasons, "caveats": caveats, "source": "template"}


def _post(base: str, model: str, key: str, facts: dict, timeout: float) -> str:
    import requests

    r = requests.post(f"{base}/chat/completions", timeout=timeout, headers={"Authorization": f"Bearer {key}"},
                      json={"model": model, "temperature": 0.2,
                            "messages": [{"role": "system", "content": SYSTEM},
                                         {"role": "user", "content": json.dumps(facts, ensure_ascii=False)}]})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def explain(facts: dict, provider: str = "template", timeout: float = 30.0) -> dict:
    """Explanation dict {summary, reasons, caveats, source[, rejected]} for the facts record."""
    if provider == "template":
        return template(facts)
    base, model = PROVIDERS[provider]
    model = os.environ.get("AUDIODF_LLM_MODEL", model)
    key = os.environ.get("AUDIODF_LLM_API_KEY")
    if not key:
        return template(facts) | {"fallback": "no AUDIODF_LLM_API_KEY set"}
    try:
        raw = _post(base, model, key, facts, timeout)
        start, end = raw.find("{"), raw.rfind("}")
        out = json.loads(raw[start:end + 1])
        text = " ".join([out["summary"], *out["reasons"], *out.get("caveats", [])])
    except Exception as exc:  # network, rate limit, refusal, malformed JSON: the demo still gets an explanation
        return template(facts) | {"fallback": f"{provider} failed: {type(exc).__name__}"}
    ok, bad = numbers_supported(text, facts)
    if not ok:
        return template(facts) | {"fallback": f"{provider} answer used numbers not in the facts {bad[:5]}"}
    return {"summary": out["summary"], "reasons": list(out["reasons"]), "caveats": list(out.get("caveats", [])), "source": f"{provider}:{model}"}
