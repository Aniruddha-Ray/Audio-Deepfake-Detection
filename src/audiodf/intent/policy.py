"""One action from the voice verdict and the caller's intent. Intent never lowers the voice's action and never decides alone; it raises the check:

| voice \\ intent | none     | caution  | high     |
|----------------|----------|----------|----------|
| allow          | allow    | allow    | verify   |
| verify         | verify   | verify   | escalate |
| escalate       | escalate | escalate | escalate |

"caution" is shown to the agent as context only: on an independent test set the rules flagged most legitimate calls about the same subjects
(audit.md phase 30), so mild patterns alone change nothing.
"""

from __future__ import annotations

ORDER = ("allow", "verify", "escalate")


def combine(voice_action: str, intent_level: str) -> dict:
    final = voice_action
    reason = "voice score"
    if intent_level == "high":
        final = "escalate" if voice_action in ("verify", "escalate") else "verify"
        reason = ("voice already escalated" if voice_action == "escalate"
                  else "risky request with a voice at or above the verify level" if final == "escalate" else "risky request in what the caller said")
    return {"action": final, "voice_action": voice_action, "intent_level": intent_level, "reason": reason}
