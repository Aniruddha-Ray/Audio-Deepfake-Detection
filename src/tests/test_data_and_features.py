import numpy as np

from audiodf.data.audio import float_to_pcm16, pcm16_to_float, to_mono_target_rate
from audiodf.data.segmenter import pad_to_length, segment
from audiodf.features.rcnn_features import RcnnFeatureExtractor
from audiodf.features.svm_features import SvmFeatureExtractor

from conftest import tone


def test_segment_two_second_windows_one_second_hop():
    segs = segment(np.arange(5 * 16000, dtype=np.float32), 32000, 16000)
    assert segs.shape == (4, 32000)
    assert segs[1][0] == 16000 and segs[3][0] == 48000


def test_segment_end_aligns_trailing_window():
    segs = segment(np.arange(int(4.5 * 16000), dtype=np.float32), 32000, 16000)
    assert segs.shape[0] == 4
    assert segs[-1][-1] == int(4.5 * 16000) - 1


def test_short_audio_is_repeat_padded():
    out = pad_to_length(np.arange(10, dtype=np.float32), 25)
    assert len(out) == 25 and list(out[:12]) == list(range(10)) + [0, 1]
    assert segment(np.ones(100, dtype=np.float32), 32000, 16000).shape == (1, 32000)


def test_pcm16_roundtrip():
    wave = tone(0.1)
    assert np.abs(pcm16_to_float(float_to_pcm16(wave)) - wave).max() < 1e-3


def test_resample_to_16k_mono():
    stereo = np.stack([tone(1), tone(1)], axis=1)
    assert len(to_mono_target_rate(stereo, 16000, 16000)) == 16000
    assert abs(len(to_mono_target_rate(tone(1), 16000, 8000)) - 8000) <= 1


def test_svm_features_are_one_318_vector_per_buffer(settings):
    fx = SvmFeatureExtractor(settings.audio, settings.svm_features)
    vec = fx.extract(tone(3))
    assert vec.shape == (318,) == (settings.svm_features.dim,)
    assert np.isfinite(vec).all()
    assert fx.extract(tone(0.2)).shape == (318,)  # tiny buffers are padded, not rejected


def test_rcnn_features_are_standardised_logmel_windows(settings):
    fx = RcnnFeatureExtractor(settings.audio, settings.rcnn_features, settings.segment_samples)
    segs = segment(tone(4, noise=0.1), settings.segment_samples, settings.segment_hop_samples)
    mel = fx.extract(segs)
    assert mel.shape == (len(segs), 64, 200)
    assert abs(mel.mean()) < 1e-3 and abs(mel.std() - 1) < 1e-2


def test_branches_are_processed_differently(settings):
    """SVM input length is free (whole buffer); RCNN input is a fixed 2 s window."""
    svm_fx = SvmFeatureExtractor(settings.audio, settings.svm_features)
    assert svm_fx.extract(tone(2)).shape == svm_fx.extract(tone(9)).shape
    rcnn_fx = RcnnFeatureExtractor(settings.audio, settings.rcnn_features, settings.segment_samples)
    assert rcnn_fx.shape == (64, 200)
