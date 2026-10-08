"""Transparent scam-intent patterns over the caller's words.

Categories describe what a fraudster tries to *do*, not the topic of the call (a rule that keys on "Microsoft" or "social security" would fit one dataset
and mean nothing to a bank): ask for credentials, redirect money, get remote access, get round verification, pressure or threaten, impersonate an
authority, demand secrecy, dangle a prize or refund. Each match keeps its phrase (and time, for a transcript with timestamps), so the explanation can quote
it. Score = 1 - prod(1 - weight) over the categories found (each counted once). English, plus a few common Hindi / Hinglish phrases.

The patterns were written while looking only at the training split of BothBosu/scam-dialogue (new_plan.md 8.0).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SENSITIVE = (r"(?:otp|one[- ]time (?:pass(?:word|code)|code)|verification code|security code|pin(?: number)?|cvv|password|passcode|"
              r"card (?:number|details)|account (?:number|details|credentials)|bank details|login(?: details)?|credentials|"
              r"social security number|ssn|date of birth|mother'?s maiden name|routing number|net ?banking|account info(?:rmation)?)")
_ASK = r"(?:confirm|give|read|tell|provide|share|verify|need|send|repeat|enter|type|say|have|get|what(?:'s| is))"

CATEGORIES = {
    # name: (weight, patterns)
    "credential_request": (0.6, [rf"\b{_ASK}\b(?:\W+\w+){{0,6}}?\W+{_SENSITIVE}\b", rf"\b{_SENSITIVE}\W+(?:\w+\W+){{0,3}}(?:please|now|quickly)\b",
                                 r"\botp\s+(?:batao|bata do|bataiye|share karo|bhejo|bolo)\b", r"\b(?:apna|aapka)\s+(?:otp|pin|password)\b"]),
    "payment_redirect": (0.5, [r"\b(?:transfer|move|wire|send)\b(?:\W+\w+){0,4}?\W+(?:money|funds|amount|balance|savings|payment)\b",
                               r"\b(?:new|different|safe|secure|another|holding)\s+(?:bank\s+)?account\b",
                               r"\bchange\s+(?:the\s+|my\s+|your\s+)?(?:payee|beneficiary|account details|bank details)\b",
                               r"\bgift ?cards?\b", r"\b(?:bitcoin|crypto(?:currency)?|western union|moneygram)\b",
                               r"\bpay\s+(?:a|the|an)?\s*(?:fee|fine|tax|processing fee|penalty|charge)\b", r"\bpaise\s+(?:transfer|bhejo)\b"]),
    "remote_access": (0.6, [r"\b(?:anydesk|teamviewer|team viewer|remote (?:access|desktop|control|support session)|screen ?shar\w*)\b",
                            r"\b(?:install|download)\s+(?:this|an|the|a)\s+(?:app|application|software|program|tool)\b",
                            r"\b(?:access|control)\s+(?:to\s+)?your\s+(?:computer|pc|laptop|phone|device)\b"]),
    "bypass_verification": (0.5, [r"\blost my (?:phone|mobile|sim)\b", r"\b(?:change|update|new)\s+(?:my\s+)?(?:phone|mobile)\s+number\b",
                                  r"\bsend\s+(?:the\s+|it\s+|that\s+)?(?:code|otp)?\s*to\s+(?:this|another|my new|a different)\b",
                                  r"\bskip\s+(?:the\s+)?(?:verification|security (?:check|questions))\b", r"\bdon'?t\s+call\s+(?:me\s+)?back\b",
                                  r"\bcan'?t\s+(?:receive|get)\s+(?:the\s+)?(?:code|otp|sms)\b"]),
    "urgency_threat": (0.25, [r"\b(?:immediately|right now|right away|urgent(?:ly)?|asap|as soon as possible|act now|last chance|before it'?s too late)\b",
                              r"\bwithin\s+(?:the\s+next\s+)?\d+\s+(?:minutes|hours)\b", r"\b(?:suspend\w*|frozen|freez\w*|blocked|deactivat\w*)\b",
                              r"\b(?:arrest\w*|warrant|legal action|lawsuit|prosecut\w*|penalt(?:y|ies)|you'?ll be sorry)\b",
                              r"\b(?:jaldi|abhi\s+ke\s+abhi|account\s+band)\b"]),
    "authority_impersonation": (0.3, [r"\b(?:calling|this is \w+)\s+from\s+(?:the\s+)?(?:\w+\s+)?(?:bank|fraud (?:department|team|prevention)|security (?:department|team)|"
                                      r"social security|irs|tax (?:department|office)|government|police|microsoft|apple|reserve bank|rbi)\b",
                                      r"\b(?:officer|agent|inspector|detective)\s+[A-Z]?\w+\b", r"\bfederal\b"]),
    "secrecy": (0.35, [r"\bdon'?t\s+(?:tell|inform|mention)\s+(?:this\s+to\s+)?(?:anyone|anybody|your (?:family|bank|wife|husband))\b",
                       r"\bkeep\s+(?:this|it)\s+(?:confidential|secret|between us|to yourself)\b", r"\bdon'?t\s+hang\s+up\b", r"\bkisi\s+ko\s+mat\s+batana\b"]),
    "lure": (0.3, [r"\byou(?:'ve| have)\s+(?:won|been selected)\b", r"\b(?:prize|lottery|jackpot|sweepstakes)\b",
                   r"\brefund\b(?:\W+\w+){0,4}?\W+(?:of|for)\b", r"\brefund you\b", r"\bissue (?:you )?a refund\b", r"\boverpa(?:id|yment)\b", r"\bclaim\s+(?:your|the)\b"]),
}
_COMPILED = {c: [re.compile(p, re.IGNORECASE) for p in ps] for c, (_, ps) in CATEGORIES.items()}
LEVELS = (("high", 0.6), ("caution", 0.3))


@dataclass(frozen=True)
class Match:
    category: str
    phrase: str
    start_s: float | None = None


def caller_turns(dialogue: str) -> list[str]:
    """The caller's turns of a 'caller: ... receiver: ...' transcript (the fraudster's role in both the datasets and a bank's inbound call)."""
    parts = re.split(r"\b(caller|receiver)\s*:", dialogue, flags=re.IGNORECASE)
    if len(parts) < 3:
        return [dialogue]
    return [parts[i + 1].strip() for i in range(1, len(parts) - 1, 2) if parts[i].lower() == "caller"]


_OWN = re.compile(r"\b(?:our|my)\s+" + _SENSITIVE, re.IGNORECASE)  # "verify our credentials": the caller offers their own, asks for nothing


def find(segments) -> list[Match]:
    """segments: list of str, or of (start_s, text). Every category's matches, in order of appearance."""
    out = []
    for seg in segments:
        start, text = (None, seg) if isinstance(seg, str) else seg
        for cat, pats in _COMPILED.items():
            for p in pats:
                for m in p.finditer(text):
                    if cat == "credential_request" and _OWN.search(m.group(0)):
                        continue
                    out.append(Match(cat, m.group(0), start))
    return out


def score(matches: list[Match]) -> float:
    found = {m.category for m in matches}
    keep = 1.0
    for c in found:
        keep *= 1 - CATEGORIES[c][0]
    return round(1 - keep, 4)


def level(s: float) -> str:
    return next((name for name, cut in LEVELS if s >= cut), "none")


def analyse(segments) -> dict:
    ms = find(segments)
    s = score(ms)
    first = {}
    for m in ms:
        first.setdefault(m.category, m)
    return {"score": s, "level": level(s), "categories": sorted(first),
            "evidence": [{"category": m.category, "phrase": m.phrase, "start_s": m.start_s} for m in first.values()]}
