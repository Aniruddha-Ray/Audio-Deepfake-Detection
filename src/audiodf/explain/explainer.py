"""One call -> verdict, SHAP regions, facts record, plain-language explanation (and optionally a picture).

Run only for calls that need it (verify / escalate, or a demo): it scores the call ~160 more times. The verdict comes from the served engine
unchanged; the explanation is built after it and cannot alter it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from audiodf.data.audio import fill_digital_silence
from audiodf.data.vad import trim_silence
from audiodf.explain.attribution import RegionMasker, faithfulness, make_scorer, shap_regions, window_scores
from audiodf.explain.llm import explain

HORIZON_S = 10.0
DEFAULT_CHANNEL_MODEL = Path(__file__).resolve().parents[3] / "results" / "gating" / "estimator.joblib"


def explained_audio(engine, wave: np.ndarray) -> np.ndarray:
    """The first 10 s of speech, prepared as the serving path prepares a recording."""
    s = engine.settings
    x = trim_silence(fill_digital_silence(np.asarray(wave, dtype=np.float32)), s.audio.sample_rate, s.vad)
    return x[: int(HORIZON_S * s.audio.sample_rate)]


LIKELY = 0.9  # measured on the call sets: "likely" is wrong on < 1% of calls without echo (< 0.3% without noise)


def channel_facts(x: np.ndarray, model_path: Path | None) -> dict | None:
    """Room echo / background noise from the channel-quality estimator, as "likely" or "not established" (descriptive only).

    Probabilities are not shown: on calls with no echo at all the estimator gives > 0.5 for 7.5% of them (up to 15% on G.711 / GSM),
    and a low value is no proof of absence (11-21% of echo calls score <= 0.05-0.2). Only the "likely" level is reliable (p >= 0.9)."""
    if not model_path or not Path(model_path).exists():
        return None
    from audiodf.gating.channel_quality import ChannelQualityModel, channel_features

    try:  # a pickled scikit-learn model only loads under the version that wrote it; the channel line is optional, the explanation is not
        model = ChannelQualityModel.load(model_path)
    except Exception:
        return None  # refit for the installed version: python -m audiodf.gating.run fit ...
    pe, pn, _ = model.predict(channel_features(x)[None])
    word = lambda p: "likely" if p >= LIKELY else "not established"  # noqa: E731
    return {"room_echo": word(float(pe[0])), "background_noise": word(float(pn[0])),
            "note": "estimated from the audio; 'likely' is wrong on about 1 in 100 calls without it"}


def build_facts(verdict, engine, x: np.ndarray, att: dict, masker: RegionMasker, check: dict, windows: list, channel: dict | None,
                top: int = 5) -> dict:
    """Numbers and labels only: no audio, no transcript, no caller details."""
    order = np.argsort(-np.abs(att["values"]))[:top]
    return {
        "verdict": {"action": verdict.action, "risk_level": verdict.risk_level, "fake_probability": round(float(verdict.fake_probability), 4),
                    "verify_threshold": round(float(engine.settings.risk.medium), 4), "escalate_threshold": round(float(engine.settings.risk.high), 4)},
        "audio": {"explained_seconds": round(len(x) / engine.settings.audio.sample_rate, 1)},
        "score_of_explained_audio": round(att["full"], 4),
        "score_with_every_region_removed": round(att["base"], 4),
        "regions_top": [{"region": masker.regions[i].label, "contribution": round(float(att["values"][i]), 4)} for i in order],
        "windows": windows,
        "channel": channel,
        "attribution_check": check,
        "model": "WavLM-Base+ (run 7, seed 0); explanation by SHAP over time x frequency regions",
    }


def explain_call(engine, wave: np.ndarray, provider: str = "template", nsamples: int = 160,
                 channel_model: Path | None = DEFAULT_CHANNEL_MODEL, plot: str | Path | None = None) -> dict | None:
    verdict = engine.predict_waveform(wave)
    if verdict is None:
        return None
    s = engine.settings
    x = explained_audio(engine, wave)
    score = make_scorer(engine, s.segment_samples, s.segment_hop_samples)
    masker = RegionMasker(x, s.audio.sample_rate)
    att = shap_regions(masker, score, nsamples=nsamples)
    check = faithfulness(masker, score, att["values"])
    windows = window_scores(score, x, s.segment_samples, s.segment_hop_samples, s.audio.sample_rate)
    facts = build_facts(verdict, engine, x, att, masker, check, windows, channel_facts(x, channel_model))
    out = {"verdict": verdict.to_dict(), "facts": facts, "explanation": explain(facts, provider)}
    if plot:
        from audiodf.explain.plot import draw

        draw(x, s.audio.sample_rate, masker, att["values"], windows, facts, plot)
        out["plot"] = str(plot)
    return out
