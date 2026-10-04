import inspect
import os

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("imageio_ffmpeg")

from audiodf.data.ffmpeg_codecs import CODECS, align, choice, missing_codecs, render_copies, roundtrip  # noqa: E402
from audiodf.data.prepare import RcnnWindowDataset, build_index, build_svm_snapshots  # noqa: E402

from conftest import SR, tone  # noqa: E402
from test_prepare import ROWS  # noqa: E402,F401  (fixture data shared with test_prepare)
from test_prepare import asv5  # noqa: E402,F401

MISSING = set(missing_codecs())  # the Linux imageio-ffmpeg build has no libgsm


def _skip_if_missing(codecs):
    absent = sorted(set(codecs) & MISSING)
    if absent:
        pytest.skip(f"this ffmpeg build lacks {', '.join(absent)}")


def _skip_unless_renderable(idx, seed, frac=1.0):
    _skip_if_missing({choice(str(u), seed)[1].name for u in idx.utt_id if choice(str(u), seed)[0] < frac})


def _speechlike(seconds=3.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)  # syllable-rate envelope so alignment has structure
    return ((0.2 * np.sin(2 * np.pi * 220 * t) + 0.05 * rng.standard_normal(len(t))) * env).astype(np.float32)


@pytest.mark.parametrize("codec", [c.name for c in CODECS])
def test_every_real_codec_keeps_length_and_timing(codec):
    _skip_if_missing({codec})
    wave = _speechlike()
    out = roundtrip(wave, codec, 0)
    assert out.shape == wave.shape and np.isfinite(out).all()
    assert not np.allclose(out, wave, atol=1e-3)  # it really went through a codec
    mid = slice(SR, 2 * SR)
    lags = range(-40, 41)
    best = max(lags, key=lambda k: float(np.dot(out[mid.start + k:mid.stop + k], wave[mid])))
    assert abs(best) <= 8  # codec delay removed (raw delays measured up to ~1100 samples)


def test_narrowband_codecs_cut_high_frequencies():
    wave = np.random.default_rng(1).standard_normal(SR * 2).astype(np.float32) * 0.1
    power = lambda x: np.abs(np.fft.rfft(x)) ** 2
    above_4k = lambda x: power(x)[len(power(x)) // 2:].sum() / power(x).sum()
    for codec in sorted({"amr_nb", "opus_nb", "speex_nb", "gsm"} - MISSING):
        assert above_4k(roundtrip(wave, codec, 0)) < 0.05 * above_4k(wave), codec


def test_align_undoes_a_known_delay():
    ref = _speechlike(2.0, seed=3)
    delayed = np.concatenate([np.zeros(700, dtype=np.float32), ref])
    assert np.allclose(align(delayed, ref), ref)
    early = ref[300:]
    assert np.allclose(align(early, ref)[300:2 * SR - 300], ref[300:2 * SR - 300])


def test_codec_choice_is_label_blind_and_reproducible():
    assert list(inspect.signature(choice).parameters) == ["utt_id", "seed"]  # nothing about the label
    assert choice("T_0000000001", 1) == choice("T_0000000001", 1)
    assert choice("T_0000000001", 1) != choice("T_0000000001", 2)
    us = [choice(f"T_{k:010d}", 1)[0] for k in range(2000)]
    assert 0.45 < np.mean(np.array(us) < 0.5) < 0.55
    names = {choice(f"T_{k:010d}", 1)[1].name for k in range(2000)}
    assert names == {c.name for c in CODECS}  # every codec family gets used


def test_catalogue2_keeps_every_classical_pick_of_catalogue1():
    """Clips that keep a classical codec get exactly their catalogue-1 codec and variant (so those copies are
    reused), and about NEURAL_SHARE of clips get EnCodec."""
    import hashlib

    from audiodf.data.ffmpeg_codecs import CLASSICAL, NEURAL_SHARE, NeuralCodec

    neural = 0
    for k in range(4000):
        utt = f"T_{k:010d}"
        u, codec, variant = choice(utt, 1)
        h = hashlib.md5(f"1:{utt}".encode()).digest()  # catalogue 1's formula, verbatim
        old = CLASSICAL[int.from_bytes(h[4:6], "little") % len(CLASSICAL)]
        assert u == int.from_bytes(h[:4], "little") / 2 ** 32  # membership never changes
        if isinstance(codec, NeuralCodec):
            neural += 1
        else:
            assert (codec, variant) == (old, int.from_bytes(h[6:8], "little") % len(old.variants))
    assert abs(neural / 4000 - NEURAL_SHARE) < 0.02


def test_encodec_batches_match_single_clips():
    _skip_if_missing({"encodec"})
    from audiodf.data.ffmpeg_codecs import encodec_roundtrip

    a, b = _speechlike(3.0, seed=4), _speechlike(2.2, seed=5)
    batched = encodec_roundtrip([a, b], 6.0)
    assert [len(x) for x in batched] == [len(a), len(b)]
    snr = lambda out, ref: 10 * np.log10((ref ** 2).sum() / ((out - ref) ** 2).sum())
    for wave, out in zip([a, b], batched):
        single = encodec_roundtrip([wave], 6.0)[0]
        # Not sample-identical: batch size changes the GPU arithmetic slightly and the quantiser can pick other
        # codes; the coding quality must be the same, and the copy must stay aligned with the original.
        assert abs(snr(out, wave) - snr(single, wave)) < 0.5
        assert np.corrcoef(out, single)[0, 1] > 0.95
    assert np.array_equal(encodec_roundtrip([a], 6.0)[0], encodec_roundtrip([a], 6.0)[0])  # deterministic per batch
    assert np.abs(batched[0]).max() < 1.0  # clipped for 16-bit FLAC


def test_missing_encodec_package_is_reported(monkeypatch):
    import audiodf.data.ffmpeg_codecs as fc

    monkeypatch.setattr(fc, "_encodec_installed", lambda: False)
    assert {"encodec", "mp3_encodec"} <= set(fc.missing_codecs())
    assert not {"encodec", "mp3_encodec"} & set(fc.missing_codecs(["opus_wb", "aac"]))


def test_render_links_catalogue1_copies_and_renders_only_new_picks(asv5):
    import audiodf.data.ffmpeg_codecs as fc

    idx = build_index(asv5, "asv5", "train", workers=0)
    kinds = lambda seed: {isinstance(choice(str(u), seed)[1], fc.NeuralCodec) for u in idx.utt_id}
    seed = next(s for s in range(100, 400) if kinds(s) == {True, False})  # both kinds among the fixture clips
    _skip_unless_renderable(idx, seed)
    old = fc.render_dir(asv5.paths.cache_dir, seed, 1)
    old.mkdir(parents=True)
    marker = np.full(SR, 0.25, dtype=np.float32)  # stands in for a catalogue-1 copy
    for u in idx.utt_id:
        sf.write(old / f"{u}.flac", marker, SR, subtype="PCM_16")
    lines = []
    rendered = fc.render_copies(idx, np.arange(len(idx)), asv5, frac=1.0, seed=seed, workers=2, log=lines.append)
    for i, u in enumerate(idx.utt_id):
        copy, _ = sf.read(rendered.path(i), dtype="float32")
        is_marker = len(copy) == SR and np.allclose(copy, 0.25, atol=1e-4)
        assert is_marker == (not isinstance(choice(str(u), seed)[1], fc.NeuralCodec))  # only classical reused
    assert any("from codec catalogue 1" in line for line in lines)


def test_render_copies_point_reads_at_aligned_codec_copies(asv5):
    idx = build_index(asv5, "asv5", "train", workers=0)
    _skip_unless_renderable(idx, 7)
    rendered = render_copies(idx, np.arange(len(idx)), asv5, frac=1.0, seed=7, workers=2, log=lambda *_: None)
    assert rendered.rendered.all() and set(rendered.render_codec) <= {c.name for c in CODECS}
    for i in range(len(idx)):
        assert rendered.path(i) != idx.path(i) and os.path.exists(rendered.path(i))
        copy, _ = sf.read(rendered.path(i), dtype="float32")
        orig, _ = sf.read(idx.path(i), dtype="float32")
        horizon = asv5.window_samples + int(0.25 * SR)
        assert len(copy) == min(len(orig), int(idx.speech_start[i]) + horizon)
        s = int(idx.speech_start[i]) + SR  # speech positions still line up after the codec
        assert np.corrcoef(copy[s:s + SR // 2], orig[s:s + SR // 2])[0, 1] > 0.5
    stamp = os.path.getmtime(rendered.path(0))
    again = render_copies(idx, np.arange(len(idx)), asv5, frac=1.0, seed=7, workers=2, log=lambda *_: None)
    assert os.path.getmtime(again.path(0)) == stamp  # reused, not re-rendered
    none = render_copies(idx, np.arange(len(idx)), asv5, frac=0.0, seed=7, workers=2, log=lambda *_: None)
    assert not none.rendered.any() and none.path(0) == idx.path(0)


def test_disk_guard_estimates_in_the_right_units(asv5, monkeypatch):
    import collections

    import audiodf.data.ffmpeg_codecs as fc

    idx = build_index(asv5, "asv5", "train", workers=0)
    _skip_unless_renderable(idx, 11)
    usage = collections.namedtuple("usage", "total used free")
    utts = np.arange(len(idx))
    monkeypatch.setattr(fc.shutil, "disk_usage", lambda _: usage(0, 0, 16e9))  # 16 GB: enough for 4 small copies
    fc.render_copies(idx, utts, asv5, frac=1.0, seed=11, workers=1, log=lambda *_: None)
    monkeypatch.setattr(fc.shutil, "disk_usage", lambda _: usage(0, 0, 10e9))  # below the 15 GB headroom
    with pytest.raises(RuntimeError, match="headroom"):
        fc.render_copies(idx, utts, asv5, frac=1.0, seed=12, workers=1, log=lambda *_: None)
    # 150k copies (run 3's scale) must come out in the tens of GB, not hundreds
    monkeypatch.setattr(fc.shutil, "disk_usage", lambda _: usage(0, 0, 40e9))
    big = np.arange(150_000)
    with pytest.raises(RuntimeError, match=r"~33 GB"):
        fc.render_copies(_FakeIdx(150_000), big, asv5, frac=1.0, seed=13, workers=1, log=lambda *_: None)


class _FakeIdx:
    """Minimal stand-in so the disk guard can be checked at full scale without any audio."""

    def __init__(self, n):
        self.utt_id = np.array([f"X_{k:010d}" for k in range(n)])

    def __len__(self):
        return len(self.utt_id)


def test_render_refuses_a_build_missing_a_chosen_codec(asv5, monkeypatch):
    """Skipping an unavailable codec would silently change which copies exist; it must stop instead."""
    import audiodf.data.ffmpeg_codecs as fc

    idx = build_index(asv5, "asv5", "train", workers=0)
    picked = {choice(str(u), 21)[1].name for u in idx.utt_id}
    monkeypatch.setattr(fc, "missing_codecs", lambda names=None: sorted(set(names or []) & picked)[:1])
    with pytest.raises(RuntimeError, match="IMAGEIO_FFMPEG_EXE"):
        fc.render_copies(idx, np.arange(len(idx)), asv5, frac=1.0, seed=21, workers=1, log=lambda *_: None)


def test_missing_codecs_reads_the_build():
    assert set(missing_codecs()) <= {c.name for c in CODECS}
    assert "aac" not in missing_codecs()  # ffmpeg's native encoder, present in every build


def test_rendered_clips_get_no_simulated_codec_on_top(asv5):
    idx = build_index(asv5, "asv5", "train", workers=0)
    _skip_unless_renderable(idx, 7)
    rendered = render_copies(idx, np.arange(len(idx)), asv5, frac=1.0, seed=7, workers=2, log=lambda *_: None)
    sub = rendered.subset(np.array([2, 3]), "x")  # render info survives subsetting
    assert sub.is_rendered(0) and sub.path(0) == rendered.path(2)
    key = (2, int(rendered.speech_start[2]) + SR)
    aug = RcnnWindowDataset(rendered, asv5, augment_p=1.0).read_window(*key)
    plain = RcnnWindowDataset(rendered, asv5, augment_p=0.0).read_window(*key)
    assert np.array_equal(aug, plain)  # augment_p=1 would otherwise always change the audio
    snaps = build_svm_snapshots(rendered, np.arange(4), asv5, 1.0, "r", workers=0)
    assert set(snaps["codec"]) == set(rendered.render_codec)  # labelled by the real codec, not a simulated one
    plain_snaps = build_svm_snapshots(idx, np.arange(4), asv5, 0.0, "r", workers=0)
    assert not np.allclose(snaps["x"], plain_snaps["x"])  # separate cache entry for rendered audio
