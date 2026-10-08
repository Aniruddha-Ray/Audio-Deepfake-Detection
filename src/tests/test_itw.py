import numpy as np
import pytest
import soundfile as sf

from audiodf.config import Settings
from audiodf.data.protocol import read_protocol


def _itw(tmp_path, rows, write=None):
    (tmp_path / "meta.csv").write_text("file,speaker,label\n" + "".join(f"{f},{s},{l}\n" for f, s, l in rows))
    for f, _, _ in rows if write is None else write:
        sf.write(tmp_path / f, np.zeros(1600, dtype=np.float32), 16000)
    return tmp_path


def test_in_the_wild_protocol_reads_labels_speakers_and_only_clips_on_disk(tmp_path):
    rows = [("0.wav", "Barack Obama", "spoof"), ("1.wav", "Barack Obama", "bona-fide"), ("2.wav", "Alan Watts", "bona-fide")]
    root = _itw(tmp_path, rows, write=rows[:2])
    every = read_protocol(root, "eval", dataset="itw")
    assert [(s.utt_id, s.label, s.attack, s.speaker, s.dataset) for s in every] == [
        ("0", 1, "itw", "Barack Obama", "itw"), ("1", 0, "-", "Barack Obama", "itw"), ("2", 0, "-", "Alan Watts", "itw")]
    assert [s.utt_id for s in read_protocol(root, "eval", dataset="itw", available_only=True)] == ["0", "1"]  # wav files count too
    assert Settings().dataset_root("itw").replace("\\", "/").endswith("dataset_itw/release_in_the_wild")


def test_in_the_wild_refuses_an_unknown_label(tmp_path):
    root = _itw(tmp_path, [("0.wav", "X", "fake")])
    with pytest.raises(ValueError, match="unknown label"):
        read_protocol(root, "eval", dataset="itw")


def test_wav_clips_are_indexed_and_read_by_their_real_path(tmp_path):
    from audiodf.data.prepare import WaveWindowDataset, build_index, stream_windows

    t = np.arange(3 * 16000) / 16000
    (tmp_path / "meta.csv").write_text("file,speaker,label\n0.wav,A,spoof\n1.wav,B,bona-fide\n")
    for k in range(2):
        sf.write(tmp_path / f"{k}.wav", (0.3 * np.sin(2 * np.pi * (200 + 100 * k) * t)).astype(np.float32), 16000)
    s = Settings()
    s.paths.itw_root, s.paths.cache_dir = str(tmp_path), str(tmp_path / "cache")
    idx = build_index(s, "itw", "eval", None, 0, lambda *_: None, available_only=True)
    assert idx.path(0).endswith("0.wav") and idx.has_speech.all()
    ds = WaveWindowDataset(idx, s, 0.0, 0)
    for i, start in stream_windows(idx, np.arange(len(idx)), s):
        assert ds.read_window(i, start).shape == (s.segment_samples,)
