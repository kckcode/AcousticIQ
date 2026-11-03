"""
Unit tests for Feature Extraction Pipeline.
"""

import numpy as np
from ml.features.extractor import LogMelExtractor, InverseMelExtractor, DualFeatureExtractor, FeatureConfig


def test_log_mel_extraction():
    sr = 16000
    t = np.linspace(0, 10.0, int(sr * 10.0), endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 500 * t)).astype(np.float32)

    config = FeatureConfig(sample_rate=sr, n_mels=64, n_fft=1024, hop_length=512)
    extractor = LogMelExtractor(config)
    mel = extractor.extract(audio)

    assert isinstance(mel, np.ndarray)
    assert mel.shape[0] == 64
    assert mel.shape[1] > 0
    assert np.isfinite(mel).all()


def test_inverse_mel_extraction():
    sr = 16000
    t = np.linspace(0, 10.0, int(sr * 10.0), endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 500 * t)).astype(np.float32)

    config = FeatureConfig(sample_rate=sr, n_mels=64, n_fft=1024, hop_length=512)
    extractor = InverseMelExtractor(config)
    inv_mel = extractor.extract(audio)

    assert isinstance(inv_mel, np.ndarray)
    assert inv_mel.shape[0] == 64
    assert inv_mel.shape[1] > 0
    assert np.isfinite(inv_mel).all()


def test_dual_feature_extractor_alignment():
    sr = 16000
    t = np.linspace(0, 10.0, int(sr * 10.0), endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * 300 * t) + 0.2 * np.sin(2 * np.pi * 1200 * t)).astype(np.float32)

    dual = DualFeatureExtractor()
    mel, inv_mel, info = dual.extract(audio)

    assert mel.shape == inv_mel.shape
    assert info["n_mels"] == 64
    assert info["time_frames"] == mel.shape[1]
