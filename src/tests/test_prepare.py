import numpy as np
import pytest
import soundfile as sf

from audiodf.config import Settings
from pathlib import Path

from audiodf.data.prepare import (RandomWindowSampler, RcnnWindowDataset, SplitIndex, build_index,
                                  build_svm_snapshots, stratified_subset, stream_windows)
from audiodf.inference.session import CallSession

from conftest import SR, tone

ROWS = ["T_0001 T_0000000000 F - - - AC3 A05 spoof -",
        "T_0002 T_0000000001 M - - - AC2 A06 spoof -",
        "T_0003 T_0000000002 F - - - - bonafide bonafide -",
        "T_0004 T_0000000003 M - - - - bonafide bonafide -"]


@pytest.fixture()
def asv5(tmp_path):
    (tmp_path / "flac_T").mkdir()
    (tmp_path / "ASVspoof5.train.tsv").write_text("\n".join(ROWS) + "\n")
    silence = np.zeros(SR // 2, dtype=np.float32)
    for k, line in enumerate(ROWS):
        speech = tone(4 + 4 * k, freq=200 + 50 * k, noise=0.05, seed=k)  # 4, 8, 12, 16 s of speech
        sf.write(tmp_path / "flac_T" / f"{line.split()[1]}.flac", np.concatenate([silence, speech, silence]), SR)
    s = Settings()
    s.paths.asv5_root = str(tmp_path)
    s.paths.cache_dir = str(tmp_path / "cache")
    return s


def test_index_records_speech_bounds(asv5):
    idx = build_index(asv5, "asv5", "train", workers=0)
    assert len(idx) == 4 and idx.has_speech.all()
    margin = int(asv5.vad.margin_seconds * SR)
    assert all(abs(int(st) - (SR // 2 - margin)) <= 400 for st in idx.speech_start)  # within one VAD frame
    assert list(idx.label) == [1, 1, 0, 0]
    assert build_index(asv5, "asv5", "train", workers=0).speech_end.tolist() == idx.speech_end.tolist()  # cached


def test_stratified_subset_balances_groups(asv5):
    idx = build_index(asv5, "asv5", "train", workers=0)
    sub = stratified_subset(idx, 3)
    assert set(idx.label[sub]) == {0, 1}


def test_svm_snapshots_are_growing_prefixes(asv5):
    idx = build_index(asv5, "asv5", "train", workers=0)
    data = build_svm_snapshots(idx, np.arange(4), asv5, augment_p=0.0, tag="t", workers=0)
    assert data["x"].shape[1] == asv5.svm_features.dim and np.isfinite(data["x"]).all()
    secs = {u: sorted(data["seconds"][data["utt"] == u].round(1)) for u in range(4)}
    assert secs[0][-1] <= 4.2 and secs[3] == [2.0, 4.0, 6.0, 8.0, 10.0]  # capped by speech, then by 10 s
    assert (data["label"] == idx.label[data["utt"]]).all()


def test_random_windows_stay_inside_speech(asv5):
    idx = build_index(asv5, "asv5", "train", workers=0)
    sampler = RandomWindowSampler(idx, asv5.segment_samples, 5, asv5.window_samples)
    keys = list(sampler)
    assert len(keys) == 20
    for i, start in keys:
        assert idx.speech_start[i] <= start <= idx.speech_end[i] - asv5.segment_samples
        # never later than the live decision horizon: a window must end within 10 s of speech onset
        assert start + asv5.segment_samples <= idx.speech_start[i] + asv5.window_samples
    long_clip = [start - idx.speech_start[i] for i, start in keys if i == 3]  # 16 s of speech
    assert max(long_clip) > 5 * SR and max(long_clip) <= 8 * SR  # uses the horizon, not just the clip start
    sampler.set_epoch(1)
    assert list(sampler) != keys  # new windows every epoch


def test_window_dataset_shapes_and_augmentation(asv5):
    idx = build_index(asv5, "asv5", "train", workers=0)
    plain, aug = RcnnWindowDataset(idx, asv5), RcnnWindowDataset(idx, asv5, augment_p=1.0)
    key = (2, int(idx.speech_start[2]) + SR)
    x, y = plain[key]
    assert tuple(x.shape) == (1, 64, 200) and float(y) == 0.0
    assert not np.allclose(aug.read_window(*key), plain.read_window(*key))
    assert np.array_equal(aug.read_window(*key), aug.read_window(*key))  # deterministic per window


def test_stream_windows_match_live_session(asv5, engine):
    idx = build_index(asv5, "asv5", "train", workers=0)
    for i in range(4):
        n_eval = len(stream_windows(idx, np.array([i]), asv5))
        wave, _ = sf.read(idx.path(i), dtype="float32")
        session = CallSession(engine)
        last = None
        for k in range(0, len(wave), SR // 2):
            last = session.push(wave[k:k + SR // 2]) or last
            if session.audio_seconds >= asv5.stream.window_seconds:
                break
        assert last.segments_scored == n_eval


def _asv19(tmp_path, n=6, seconds=4.0):
    """ASVspoof2019-style folder with a train and a dev split of n clips each (different speakers)."""
    root = tmp_path / "asv19"
    (root / "ASVspoof2019_LA_cm_protocols").mkdir(parents=True)
    for split, tag, proto, first_id, spk_base in (("train", "T", "train.trn", 1000, 0), ("dev", "D", "dev.trl", 2000, 50)):
        audio = root / f"ASVspoof2019_LA_{split}" / "flac"
        audio.mkdir(parents=True)
        rows = []
        for k in range(n):
            spoof = k % 2 == 0
            utt = f"LA_{tag}_{first_id + k}"
            rows.append(f"LA_{spk_base + k // 2:04d} {utt} - {'A01' if spoof else '-'} {'spoof' if spoof else 'bonafide'}")
            sf.write(audio / f"{utt}.flac", tone(seconds, freq=300 + 40 * k, noise=0.05, seed=k), SR)
        (root / "ASVspoof2019_LA_cm_protocols" / f"ASVspoof2019.LA.cm.{proto}.txt").write_text("\n".join(rows) + "\n")
    return root


def test_concat_index_resolves_paths_and_namespaces_attacks(asv5, tmp_path):
    asv5.paths.data_root = str(_asv19(tmp_path))
    a = build_index(asv5, "asv5", "train", workers=0)
    b = build_index(asv5, "asv19", "train", workers=0)
    both = SplitIndex.concat([a, b])
    assert len(both) == len(a) + len(b) == 10
    assert all(Path(both.path(i)).exists() for i in range(len(both)))
    assert Path(both.path(0)) == tmp_path / "flac_T" / "T_0000000000.flac"
    assert Path(both.path(5)) == tmp_path / "asv19" / "ASVspoof2019_LA_train" / "flac" / "LA_T_1001.flac"
    assert set(both.attack) == {"-", "asv5:A05", "asv5:A06", "asv19:A01"}  # same code, different systems
    assert both.label.tolist() == a.label.tolist() + b.label.tolist()
    assert (both.speech_end > both.speech_start).all()
    # downstream code works on the joined index unchanged
    picked = stratified_subset(both, 6)  # floor division per attack group: 5 or 6 clips
    assert 5 <= len(picked) <= 6 and set(both.label[picked]) == {0, 1} and len(set(both.attack[picked])) == 4
    data = build_svm_snapshots(both, np.arange(10), asv5, 0.0, "mix", workers=0)
    assert set(data["utt"]) == set(range(10)) and np.isfinite(data["x"]).all()
    ds = RcnnWindowDataset(both, asv5)
    x, y = ds[(7, int(both.speech_start[7]))]  # a clip from the second dataset
    assert tuple(x.shape) == (1, 64, 200) and float(y) == float(both.label[7])
    assert set(both.source) == {"asv5:train", "asv19:train"}


def test_snapshot_cache_follows_clips_not_positions(asv5, tmp_path):
    """A subset re-numbers clips; a cached snapshot file must still map features to the right clips."""
    asv5.paths.data_root = str(_asv19(tmp_path))
    both = SplitIndex.concat([build_index(asv5, "asv5", "train", workers=0),
                              build_index(asv5, "asv19", "train", workers=0)])
    full = build_svm_snapshots(both, np.arange(4, 10), asv5, 0.0, "t", workers=0)  # the ASV2019 clips
    sub = both.subset(both.source == "asv19:train", "tune")  # same clips, now at positions 0..5
    assert sub.utt_id.tolist() == both.utt_id[4:].tolist() and sub.path(0) == both.path(4)
    again = build_svm_snapshots(sub, np.arange(6), asv5, 0.0, "t", workers=0)  # served from the cache
    assert sorted(set(again["utt"])) == list(range(6))
    for j in range(6):
        assert np.array_equal(again["x"][again["utt"] == j], full["x"][full["utt"] == j + 4])
    assert (again["label"] == sub.label[again["utt"]]).all()
    other = build_svm_snapshots(both, np.arange(0, 4), asv5, 0.0, "t", workers=0)  # different clips: own cache
    assert sorted(set(other["utt"])) == [0, 1, 2, 3]
