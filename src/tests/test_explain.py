import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from audiodf.config import Settings
from audiodf.explain import llm
from audiodf.explain.attribution import RegionMasker, faithfulness, make_scorer, shap_regions, window_scores
from audiodf.explain.explainer import explain_call

SR = 16000


def _band_ratio(seg: np.ndarray) -> float:
    spec = np.abs(np.fft.rfft(seg)) ** 2
    f = np.fft.rfftfreq(len(seg), 1 / SR)
    return float(spec[(f >= 1000) & (f < 3000)].sum() / (spec.sum() + 1e-12))


class FakeEngine:
    """Stands in for the served engine: a window's 'fake score' is the share of its energy in 1-3 kHz, so the answer is known."""

    def __init__(self):
        self.settings = Settings()
        self.weights, self.uses_svm = {"wavlm": 1.0}, False

    def window_probabilities(self, segs):
        return {"wavlm": np.array([_band_ratio(s) for s in segs])}

    def predict_waveform(self, wave):
        from audiodf.explain.explainer import explained_audio

        x = explained_audio(self, wave)
        p = float(make_scorer(self, self.settings.segment_samples, self.settings.segment_hop_samples)([x])[0])
        action = "escalate" if p >= self.settings.risk.high else "verify" if p >= self.settings.risk.medium else "allow"
        d = {"fake_probability": p, "action": action, "risk_level": {"escalate": "high", "verify": "medium"}.get(action, "low")}
        return SimpleNamespace(**d, to_dict=lambda: dict(d))


def _call(seconds=6.0, tone_from=2.0, tone_to=4.0, seed=0):
    """Low voice-like hum all along, plus a 2 kHz tone only between tone_from and tone_to."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    x = 0.2 * np.sin(2 * np.pi * 150 * t) + 0.1 * np.sin(2 * np.pi * 450 * t) + 0.002 * rng.standard_normal(len(t))
    x = x + 0.3 * np.sin(2 * np.pi * 2000 * t) * ((t >= tone_from) & (t < tone_to))
    return x.astype(np.float32)


def test_masker_keeps_the_audio_when_nothing_is_removed_and_removes_exactly_the_chosen_region():
    x = _call()
    m = RegionMasker(x, SR)
    assert m.n == 3 * 4 and [r.label for r in m.regions][:2] == ["0-2 s, below 300 Hz", "0-2 s, 300 Hz-1 kHz"]
    assert np.array_equal(m.apply(np.ones(m.n)), x)
    keep = np.ones(m.n)
    tone = next(r.index for r in m.regions if r.t0 == 2 and r.band == "1-3 kHz")
    keep[tone] = 0
    y = m.apply(keep)
    assert len(y) == len(x)
    seg = slice(int(2.3 * SR), int(3.7 * SR))
    assert _band_ratio(y[seg]) < 0.05 < 0.5 < _band_ratio(x[seg])  # the tone is gone from 2-4 s...
    out = slice(int(0.2 * SR), int(1.8 * SR))
    assert np.abs(y[out] - x[out]).max() < 0.02  # ...and the rest of the call is untouched


def test_shap_finds_the_region_that_drives_the_score_and_the_values_add_up():
    eng = FakeEngine()
    x = _call()
    m = RegionMasker(x, SR)
    score = make_scorer(eng, eng.settings.segment_samples, eng.settings.segment_hop_samples)
    att = shap_regions(m, score, nsamples=120)
    top = m.regions[int(np.argmax(att["values"]))]
    assert (top.t0, top.band) == (2.0, "1-3 kHz")
    assert att["values"].sum() == pytest.approx(att["full"] - att["base"], abs=1e-3)  # SHAP efficiency
    check = faithfulness(m, score, att["values"])
    assert check["checked"] and check["faithful"] and check["drop_top"] > check["drop_random"]
    ws = window_scores(score, x, eng.settings.segment_samples, eng.settings.segment_hop_samples, SR)
    assert [w["start_s"] for w in ws] == [0.0, 1.0, 2.0, 3.0, 4.0] and max(ws, key=lambda w: w["p_fake"])["start_s"] == 2.0


def test_explain_call_builds_facts_without_audio_or_caller_data_and_the_score_matches_the_verdict(tmp_path):
    eng = FakeEngine()
    out = explain_call(eng, _call(), "template", nsamples=80, channel_model=None, plot=tmp_path / "x.png")
    f = out["facts"]
    assert f["score_of_explained_audio"] == pytest.approx(out["verdict"]["fake_probability"], abs=1e-4)
    assert set(f) == {"verdict", "audio", "score_of_explained_audio", "score_with_every_region_removed", "regions_top", "windows",
                      "channel", "attribution_check", "model"}
    assert f["regions_top"][0]["region"] == "2-4 s, 1-3 kHz"
    assert (tmp_path / "x.png").stat().st_size > 10_000
    text = json.dumps(f)
    assert "call_id" not in text and len(text) < 4000  # numbers and labels only
    e = out["explanation"]
    assert e["source"] == "template" and "2-4 s, 1-3 kHz" in e["reasons"][0] and e["caveats"]


def _facts():
    eng = FakeEngine()
    return explain_call(eng, _call(), "template", nsamples=60, channel_model=None)["facts"]


def test_an_llm_answer_with_only_the_facts_numbers_is_used(monkeypatch):
    facts = _facts()
    p = facts["verdict"]["fake_probability"]
    good = {"summary": f"The call scored {p:.2f} and was sent to '{facts['verdict']['action']}'.",
            "reasons": ["Most of the evidence came from 2-4 s in the 1-3 kHz band."], "caveats": ["Regions are not causes."]}
    monkeypatch.setenv("AUDIODF_LLM_API_KEY", "test-key")
    monkeypatch.setattr(llm, "_post", lambda *a, **k: "Here you go: " + json.dumps(good))
    e = llm.explain(facts, "groq")
    assert e["source"].startswith("groq:") and e["summary"] == good["summary"]


def test_an_llm_answer_with_an_invented_number_or_any_failure_falls_back_to_the_template(monkeypatch):
    facts = _facts()
    monkeypatch.setenv("AUDIODF_LLM_API_KEY", "test-key")
    bad = {"summary": "The call scored 0.37 and the caller is 92% likely to be a fraudster.", "reasons": ["x"], "caveats": []}
    monkeypatch.setattr(llm, "_post", lambda *a, **k: json.dumps(bad))
    e = llm.explain(facts, "gemini")
    assert e["source"] == "template" and "not in the facts" in e["fallback"]
    monkeypatch.setattr(llm, "_post", lambda *a, **k: (_ for _ in ()).throw(ConnectionError("rate limited")))
    assert llm.explain(facts, "openrouter")["fallback"] == "openrouter failed: ConnectionError"
    monkeypatch.setattr(llm, "_post", lambda *a, **k: "no json at all")
    assert llm.explain(facts, "groq")["source"] == "template"
    monkeypatch.delenv("AUDIODF_LLM_API_KEY")
    assert "no AUDIODF_LLM_API_KEY" in llm.explain(facts, "groq")["fallback"]


def test_number_check_allows_rounding_and_percentages_but_not_new_values():
    facts = {"verdict": {"fake_probability": 0.7789, "verify_threshold": 0.006}, "regions_top": [{"region": "2-4 s, 1-3 kHz", "contribution": 0.192}]}
    assert llm.numbers_supported("Scored 0.78 (78%), region 2-4 s at 1-3 kHz, +0.19, verify 0.0060", facts)[0]
    ok, bad = llm.numbers_supported("Scored 0.78 with 99% certainty", facts)
    assert not ok and bad == [99.0]


def test_explain_endpoint_returns_the_verdict_facts_and_text():
    from fastapi.testclient import TestClient

    from audiodf.serving.api import create_app

    eng = FakeEngine()
    buf = io.BytesIO()
    sf.write(buf, _call(), SR, format="WAV")
    with TestClient(create_app(engine=eng, settings=eng.settings)) as client:
        r = client.post("/explain?nsamples=40", content=buf.getvalue())
        assert r.status_code == 200 and set(r.json()) == {"verdict", "facts", "explanation"}
        assert client.post("/explain?provider=nope", content=buf.getvalue()).status_code == 400


def test_an_unloadable_channel_model_leaves_out_the_channel_line_but_never_fails(tmp_path):
    bad = tmp_path / "estimator.joblib"
    bad.write_bytes(b"not a pickle from this scikit-learn")
    out = explain_call(FakeEngine(), _call(), "template", nsamples=40, channel_model=bad)
    assert out["facts"]["channel"] is None and out["explanation"]["summary"]
