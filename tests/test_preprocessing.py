"""
Unit tests for Audio Preprocessing Pipeline.
"""

import io
import pytest
import numpy as np
import soundfile as sf
from ml.preprocessing.audio import AudioPreprocessor, PreprocessingConfig, AudioValidationError


def test_valid_audio_preprocessing():
    # Generate 3-second 440 Hz test sine wave at 16 kHz
    sr = 16000
    t = np.linspace(0, 3.0, int(sr * 3.0), endpoint=False)
    sine = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    buf = io.BytesIO()
    sf.write(buf, sine, sr, format="WAV")
    buf.seek(0)

    preprocessor = AudioPreprocessor()
    processed_audio, meta = preprocessor.load_and_preprocess(buf)

    assert isinstance(processed_audio, np.ndarray)
    assert processed_audio.dtype == np.float32
    assert len(processed_audio) == int(16000 * 10.0) # padded to target 10s
    assert meta["processed_sample_rate"] == 16000
    assert np.max(np.abs(processed_audio)) <= 1.0


def test_stereo_downmix_to_mono():
    sr = 16000
    t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
    left = 0.4 * np.sin(2 * np.pi * 200 * t)
    right = 0.4 * np.sin(2 * np.pi * 400 * t)
    stereo = np.column_stack([left, right]).astype(np.float32)

    buf = io.BytesIO()
    sf.write(buf, stereo, sr, format="WAV")
    buf.seek(0)

    preprocessor = AudioPreprocessor()
    processed_audio, meta = preprocessor.load_and_preprocess(buf)

    assert processed_audio.ndim == 1
    assert any("downmixed to mono" in w for w in meta["warnings"])


def test_silence_detection_error():
    sr = 16000
    silence = np.zeros(int(sr * 2.0), dtype=np.float32)

    buf = io.BytesIO()
    sf.write(buf, silence, sr, format="WAV")
    buf.seek(0)

    preprocessor = AudioPreprocessor()
    with pytest.raises(AudioValidationError, match="silent"):
        preprocessor.load_and_preprocess(buf)


def test_short_duration_error():
    sr = 16000
    # 0.1s is shorter than min_duration (0.5s)
    too_short = (0.3 * np.sin(2 * np.pi * 100 * np.linspace(0, 0.1, int(sr * 0.1)))).astype(np.float32)

    buf = io.BytesIO()
    sf.write(buf, too_short, sr, format="WAV")
    buf.seek(0)

    preprocessor = AudioPreprocessor()
    with pytest.raises(AudioValidationError, match="shorter than minimum"):
        preprocessor.load_and_preprocess(buf)


def test_nan_inf_detection():
    sr = 16000
    bad_audio = np.ones(sr * 2, dtype=np.float32)
    bad_audio[100] = np.nan

    buf = io.BytesIO()
    sf.write(buf, bad_audio, sr, subtype="FLOAT", format="WAV")
    buf.seek(0)

    preprocessor = AudioPreprocessor()
    with pytest.raises(AudioValidationError, match="invalid values"):
        preprocessor.load_and_preprocess(buf)
