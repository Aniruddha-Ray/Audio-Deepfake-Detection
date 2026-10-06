import io
import json

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("pyarrow")
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from audiodf.config import Settings  # noqa: E402
from audiodf.data.asv21 import AUDIO_DIR, PROTOCOL_FILE, extract  # noqa: E402
from audiodf.data.protocol import read_protocol  # noqa: E402
from audiodf.evaluation.stream_eval import StreamScores, operating_point_report, write_scores  # noqa: E402
from audiodf.training.pipeline import evaluate_artifacts, run_training, training_splits  # noqa: E402

from conftest import SR, tone  # noqa: E402
from test_training_pipeline import _settings, _write_split  # noqa: E402

CODECS = ["none", "alaw", "ulaw", "gsm", "g722", "opus", "pstn"]


def _flac_bytes(wave) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, wave, SR, format="FLAC", subtype="PCM_16")
    return buf.getvalue()


def _fake_parquet(folder, n=28, parts=2, seconds=5.0):
    """The Hugging Face layout: path, audio{bytes,path}, label (1 = spoof), notes (JSON)."""
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for k in range(n):
        spoof = k % 2 == 0
        codec = CODECS[k % len(CODECS)]
        notes = {"utterance_id": f"LA_E_{k:07d}", "speaker_id": f"LA_{k % 5:04d}", "codec": codec,
                 "transmission": "-" if codec == "none" else ("mad_tx" if codec == "pstn" else "loc_tx"),
                 "attack_id": f"A{7 + k % 3:02d}" if spoof else "bonafide", "trim": "notrim", "phase": "eval"}
        wave = tone(seconds, freq=180 + 20 * k, noise=0.02 if spoof else 0.2, seed=k)
        wave = np.concatenate([np.zeros(SR // 4, dtype=np.float32), wave])
        rows.append({"path": f"{notes['utterance_id']}.flac", "audio": {"bytes": _flac_bytes(wave), "path": None},
                     "label": int(spoof), "notes": json.dumps(notes)})
    for p in range(parts):
        part = rows[p::parts]
        pq.write_table(pa.Table.from_pylist(part), folder / f"test-{p:05d}-of-{parts:05d}.parquet", row_group_size=5)
    return rows


def test_extract_writes_flacs_and_protocol_resumably(tmp_path):
    _fake_parquet(tmp_path / "data")
    lines = []
    n = extract(tmp_path / "data", tmp_path, log=lines.append)
    assert n == 28 and len(list((tmp_path / AUDIO_DIR).glob("*.flac"))) == 28
    s = Settings()
    s.paths.asv21_root = str(tmp_path)
    samples = read_protocol(s.dataset_root("asv21"), "eval", dataset="asv21")
    assert len(samples) == 28 and {x.dataset for x in samples} == {"asv21"}
    by = {x.utt_id: x for x in samples}
    first = by["LA_E_0000000"]  # spoof, codec "none"
    assert (first.label, first.attack, first.codec, first.attack_key) == (1, "A07", "-", "asv21:A07")
    assert (by["LA_E_0000001"].label, by["LA_E_0000001"].attack, by["LA_E_0000001"].codec) == (0, "-", "alaw")
    assert {x.codec for x in samples} == {"-", "alaw", "ulaw", "gsm", "g722", "opus", "pstn"}
    assert sf.info(first.path).samplerate == SR
    stamp = (tmp_path / AUDIO_DIR / "LA_E_0000003.flac").stat().st_mtime_ns
    lines.clear()
    extract(tmp_path / "data", tmp_path, log=lines.append)  # re-run keeps existing files
    assert (tmp_path / AUDIO_DIR / "LA_E_0000003.flac").stat().st_mtime_ns == stamp and "0 written, 28" in lines[-2]
    first_row = (tmp_path / PROTOCOL_FILE).read_text().splitlines()[0].split()
    assert len(first_row) == 8 and first_row[0].startswith("LA_E_")  # 8 columns, see data/asv21.COLUMNS


def test_extract_rejects_label_attack_mismatch_and_missing_files(tmp_path):
    rows = _fake_parquet(tmp_path / "data", n=4, parts=1)
    bad = json.loads(rows[0]["notes"])
    bad["attack_id"] = "bonafide"  # label says spoof
    rows[0]["notes"] = json.dumps(bad)
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "data" / "test-00000-of-00001.parquet")
    with pytest.raises(ValueError, match="disagrees"):
        extract(tmp_path / "data", tmp_path, log=lambda *_: None)
    with pytest.raises(FileNotFoundError):
        extract(tmp_path / "nothing_here", tmp_path, log=lambda *_: None)


def test_asv21_is_test_only():
    s = Settings()
    s.data.train_splits = ("asv5:train", "asv21:eval")
    with pytest.raises(ValueError, match="eval splits are tests"):
        training_splits(s)


def test_operating_point_report_and_scores_csv(tmp_path):
    rng = np.random.default_rng(0)
    y = np.array([0, 1] * 100)
    codec = np.where(np.arange(200) % 4 < 2, "alaw", "opus")
    p = np.clip(0.2 + 0.5 * y + rng.normal(0, 0.05, 200), 0, 1)
    sc = StreamScores(np.arange(200) + 10, y, np.where(y == 1, "A07", "-"), codec, (2.0, 10.0),
                      {"wavlm": np.tile(p[:, None], (1, 2))})
    rep = operating_point_report(sc, {"wavlm": 1.0}, {"high": 0.5, "medium": 0.3})
    assert rep["overall"]["verify"] == {"threshold": 0.3, "bonafide_flagged": 0.0, "spoof_caught": 1.0}
    assert set(rep["per_codec"]) == {"alaw", "opus"}
    rep = operating_point_report(sc, {"wavlm": 1.0}, {"high": 0.9, "medium": 0.01})  # all verified, none blocked
    assert rep["overall"]["verify"]["bonafide_flagged"] == 1.0 and rep["overall"]["block"]["spoof_caught"] == 0.0
    path = tmp_path / "scores.csv"
    write_scores(sc, {"wavlm": 1.0}, path)
    lines = path.read_text().splitlines()
    assert lines[0] == "clip_index,label,attack,codec,wavlm_2s,wavlm_10s,fused_2s,fused_10s" and len(lines) == 201
    assert lines[1].split(",")[:4] == ["10", "0", "-", "alaw"]


def test_evaluate_a_bundle_on_asv21_with_a_branch_subset(tmp_path):
    """Train a tiny svm+rcnn bundle, then score it on a fake ASV2021 set: all branches, then one branch only."""
    _write_split(tmp_path, "T", "flac_T", "ASVspoof5.train.tsv", ["A01", "A02"])
    _write_split(tmp_path, "D", "flac_D", "ASVspoof5.dev.track_1.tsv", ["A09", "A10"])
    s = _settings()
    s.paths.asv5_root, s.paths.data_root = str(tmp_path), str(tmp_path / "no_asv19")
    s.paths.asv21_root = str(tmp_path / "asv21")
    s.paths.cache_dir, s.paths.artifacts_dir = str(tmp_path / "cache"), str(tmp_path / "artifacts")
    s.paths.results_dir = str(tmp_path / "results")
    s.rcnn_train.epochs, s.rcnn_train.batch_size = 1, 4
    s.data.svm_train_utts = s.data.tune_utts = 24
    s.data.train_splits, s.data.holdout_attacks = ("asv5:train",), ()
    s.ensemble.branches = ("svm", "rcnn")
    run_training(s, workers=0, log=lambda *_: None)
    _fake_parquet(tmp_path / "pq")
    extract(tmp_path / "pq", tmp_path / "asv21", log=lambda *_: None)

    quiet = lambda *_: None  # noqa: E731
    both = evaluate_artifacts(s, "asv21", "eval", 0, 0, log=quiet, scores_path=tmp_path / "all.csv")
    assert set(both["at_horizon"]) == {"svm", "rcnn", "fused"} and both["n"] == 28
    point = both["operating_point"]
    assert point["overall"]["verify"]["threshold"] <= point["overall"]["block"]["threshold"]
    assert "per_codec" in point and (tmp_path / "all.csv").exists()
    assert both["n"] == 28 and "-" in point["per_codec"]  # codec "none" is reported as "-"
    one = evaluate_artifacts(s, "asv21", "eval", 0, 0, log=quiet, branches=("rcnn",))
    assert set(one["at_horizon"]) == {"rcnn", "fused"} and one["fusion_weights"] == {"rcnn": 1.0}
    assert "operating_point" not in one  # thresholds were tuned for the full fusion
    assert one["at_horizon"]["rcnn"]["eer_pct"] == both["at_horizon"]["rcnn"]["eer_pct"]  # same scores either way
    with pytest.raises(ValueError, match="not"):
        evaluate_artifacts(s, "asv21", "eval", 0, 0, log=quiet, branches=("wavlm",))

    # the live path (CallSession, 0.5 s chunks) agrees with the batched scores it is compared with
    from audiodf.evaluation.live_check import live_check

    live = live_check(s, "asv21", "eval", tmp_path / "all.csv", "fused_10s", n=12, log=quiet)
    assert live["clips"] == 12 and live["verdicts_per_call_median"] >= 1
    assert live["abs_diff"]["median"] < 0.05 and live["verdict_latency_ms"]["p50"] > 0
    assert 0 <= live["eer_live_pct"] <= 100
