import inspect

import numpy as np
import pytest
import soundfile as sf

from audiodf.config import Settings, VadConfig
from audiodf.data.augment import CODECS, CodecAugmenter, apply_codec
from audiodf.data.integrity import _shortcut_flags, audit_dataset
from audiodf.data.protocol import read_protocol
from audiodf.data.segmenter import prefix_snapshots, stream_window_starts
from audiodf.data.vad import speech_bounds, trim_silence
from audiodf.inference.session import CallSession

from conftest import SR, tone

ASV5_ROWS = {
    "train": ["T_0001 T_0000000000 F - - - AC3 A05 spoof -",
              "T_0002 T_0000000001 M - - - - bonafide bonafide -"],
    "dev": ["D_0001 D_0000000000 F - - - AC1 A11 spoof -",
            "D_0002 D_0000000001 M - - - - bonafide bonafide -"],
}


def _make_asv5(root, rows=ASV5_ROWS, seconds=3.0):
    names = {"train": ("ASVspoof5.train.tsv", "flac_T"), "dev": ("ASVspoof5.dev.track_1.tsv", "flac_D"),
             "eval": ("ASVspoof5.eval.track_1.tsv", "flac_E_eval")}
    for split, lines in rows.items():
        proto, audio = names[split]
        (root / audio).mkdir(parents=True, exist_ok=True)
        (root / proto).write_text("\n".join(lines) + "\n")
        for line in lines:
            sf.write(root / audio / f"{line.split()[1]}.flac", tone(seconds), SR)


def test_asv5_protocol_columns_and_namespaces(tmp_path):
    _make_asv5(tmp_path)
    spoof, bona = read_protocol(tmp_path, "train", dataset="asv5")
    assert (spoof.label, spoof.attack, spoof.codec, spoof.speaker) == (1, "A05", "-", "T_0001")
    assert (bona.label, bona.attack) == (0, "-")  # ATTACK_TAG (AC3) is not mistaken for the codec
    assert spoof.attack_key == "asv5:A05" != "asv19:A05"


def test_asv5_protocol_rejects_wrong_column_count(tmp_path):
    _make_asv5(tmp_path, {"train": ["T_0001 T_0000000000 F - - spoof -"]})
    with pytest.raises(ValueError, match="10 columns"):
        read_protocol(tmp_path, "train", dataset="asv5")


def test_audit_passes_clean_data_and_catches_shared_speakers(tmp_path):
    s = Settings()
    s.paths.asv5_root = str(tmp_path)
    _make_asv5(tmp_path)
    report = audit_dataset(s, "asv5", sample_n=None, silence_n=10)
    assert report["ok"], report["errors"]
    assert report["splits"]["eval"] == {"protocol": "missing"}

    leaky = dict(ASV5_ROWS, dev=["T_0001 D_0000000000 F - - - AC1 A11 spoof -"])
    _make_asv5(tmp_path / "leaky", leaky)
    s.paths.asv5_root = str(tmp_path / "leaky")
    report = audit_dataset(s, "asv5", sample_n=None, silence_n=10)
    assert not report["ok"] and any("speakers appear in both" in e for e in report["errors"])


def test_available_only_keeps_clips_on_disk(tmp_path):
    _make_asv5(tmp_path)
    (tmp_path / "flac_T" / "T_0000000000.flac").unlink()
    assert len(read_protocol(tmp_path, "train", dataset="asv5")) == 2
    kept = read_protocol(tmp_path, "train", dataset="asv5", available_only=True)
    assert [s.utt_id for s in kept] == ["T_0000000001"]


def test_partial_eval_is_allowed_but_partial_train_is_not(tmp_path):
    s = Settings()
    s.paths.asv5_root = str(tmp_path)
    eval_rows = ["E_0001 E_0000000000 F C01 1 E_0000000009 AC1 A17 spoof -",
                 "E_0002 E_0000000001 M - - - - bonafide bonafide -",
                 "E_0003 E_0000000002 F C02 2 E_0000000008 AC2 A18 spoof -"]
    _make_asv5(tmp_path, dict(ASV5_ROWS, eval=eval_rows))
    (tmp_path / "flac_E_eval" / "E_0000000002.flac").unlink()  # eval only partly downloaded
    report = audit_dataset(s, "asv5", sample_n=None, silence_n=10)
    assert report["ok"], report["errors"]
    assert report["splits"]["eval"]["partial"] == "2 of 3 clips on disk"
    assert report["splits"]["eval"]["rows"] == 3 and report["splits"]["eval"]["codec_by_label"]["spoof"] == {"C01": 1}
    (tmp_path / "flac_D" / "D_0000000000.flac").unlink()  # a missing dev clip is still an error
    assert not audit_dataset(s, "asv5", sample_n=None, silence_n=10)["ok"]


def test_audit_catches_missing_audio(tmp_path):
    s = Settings()
    s.paths.asv5_root = str(tmp_path)
    _make_asv5(tmp_path)
    (tmp_path / "flac_T" / "T_0000000000.flac").unlink()
    report = audit_dataset(s, "asv5", sample_n=None, silence_n=10)
    assert not report["ok"] and any("no audio" in e for e in report["errors"])


def test_shortcut_flag_needs_real_gap():
    stats = lambda b, s: {"bonafide": {"median": b}, "spoof": {"median": s}}
    assert _shortcut_flags("x", {"trailing_silence_s_by_label": stats(0.263, 0.025)})
    assert not _shortcut_flags("x", {"trailing_silence_s_by_label": stats(0.013, 0.0)})  # under one frame
    assert not _shortcut_flags("x", {"duration_s_by_label": stats(7.0, 7.06)})


@pytest.mark.parametrize("codec", CODECS)
def test_every_codec_keeps_length_and_stays_finite(codec):
    wave = tone(2, noise=0.05)
    out = apply_codec(wave, codec)
    assert out.shape == wave.shape and np.isfinite(out).all() and np.abs(out).max() <= 1.0


def test_narrowband_codecs_remove_high_frequencies():
    wave = np.random.default_rng(0).standard_normal(SR * 2).astype(np.float32) * 0.1
    power = lambda x: np.abs(np.fft.rfft(x)) ** 2
    above_4k = lambda x: power(x)[len(power(x)) // 2:].sum() / power(x).sum()
    for codec in ("opus_nb", "g711_ulaw", "narrowband"):
        assert above_4k(apply_codec(wave, codec)) < 0.1 * above_4k(wave)


def test_augmenter_is_label_blind_and_respects_p():
    assert list(inspect.signature(CodecAugmenter.__call__).parameters) == ["self", "wave", "rng"]
    wave = tone(1)
    rng = np.random.default_rng(0)
    assert all(CodecAugmenter(p=0.0)(wave, rng)[1] == "-" for _ in range(20))
    used = [CodecAugmenter(p=0.5)(wave, rng)[1] for _ in range(200)]
    assert 0.35 < np.mean([u != "-" for u in used]) < 0.65


def test_vad_trims_edges_and_keeps_margin():
    silence = np.zeros(SR, dtype=np.float32)
    clip = np.concatenate([silence, tone(2), silence])
    trimmed = trim_silence(clip, SR, VadConfig())
    assert abs(len(trimmed) / SR - (2 + 2 * VadConfig().margin_seconds)) < 0.03
    assert speech_bounds(np.zeros(SR * 2, dtype=np.float32), SR) is None
    assert len(trim_silence(np.zeros(SR, dtype=np.float32), SR)) == 0


def test_prefix_snapshots_match_growing_buffer():
    wave = np.arange(SR * 7, dtype=np.float32)
    snaps = prefix_snapshots(wave, SR, (2, 4, 6, 8, 10), min_samples=2 * SR)
    assert [len(s) for s in snaps] == [2 * SR, 4 * SR, 6 * SR, 7 * SR]  # capped by clip length
    assert all((s == wave[:len(s)]).all() for s in snaps)
    short = prefix_snapshots(np.ones(SR, dtype=np.float32), SR, (2, 4), min_samples=2 * SR)
    assert [len(s) for s in short] == [2 * SR]  # repeat-padded like serving


def test_stream_window_starts_follow_live_grid():
    assert stream_window_starts(SR * 30, 2 * SR, SR, 10 * SR) == [i * SR for i in range(9)]
    assert stream_window_starts(int(SR * 4.5), 2 * SR, SR, 10 * SR) == [0, SR, 2 * SR]


def test_session_skips_leading_silence(engine):
    s = CallSession(engine, "c")
    assert s.push(np.zeros(SR * 3, dtype=np.float32)) is None  # silence only: nothing buffered
    assert s.audio_seconds == 0 and s.finalize() is None
    v = s.push(tone(2.5))
    assert v is not None and v.segments_scored == 1
    assert abs(s.skipped_seconds - (3 - VadConfig().margin_seconds)) < 0.03


def test_offline_prediction_trims_silence_padding(engine):
    speech = tone(5, noise=0.1, seed=4)
    padded = np.concatenate([np.zeros(SR * 2, dtype=np.float32), speech, np.zeros(SR * 2, dtype=np.float32)])
    a, b = engine.predict_waveform(speech), engine.predict_waveform(padded)
    assert b.segments_scored == a.segments_scored
    assert b.audio_seconds - a.audio_seconds <= 2 * VadConfig().margin_seconds + 0.01  # 4 s of padding dropped
    assert engine.predict_waveform(np.zeros(SR * 3, dtype=np.float32)) is None


def test_svm_features_bounded_on_digital_silence(settings):
    from audiodf.features.svm_features import SvmFeatureExtractor

    fx = SvmFeatureExtractor(settings.audio, settings.svm_features)
    gappy = np.concatenate([tone(1), np.zeros(SR, dtype=np.float32), tone(1)])
    for wave in (np.zeros(SR * 2, dtype=np.float32), gappy):
        vec = fx.extract(wave)
        assert np.isfinite(vec).all() and np.abs(vec).max() < 1e3
