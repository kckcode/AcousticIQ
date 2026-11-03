"""
Audio Preprocessing Pipeline for AcousticIQ.
Provides robust WAV audio validation, resampling, channel mixing, segmentation, and normalization.
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import numpy as np
import soundfile as sf
import librosa


class AudioValidationError(Exception):
    """Custom exception raised when an input audio file fails quality or format checks."""
    pass


@dataclass
class PreprocessingConfig:
    target_sample_rate: int = 16000
    target_duration_seconds: float = 10.0
    normalize_amplitude: bool = True
    silence_threshold_db: float = -60.0
    mono: bool = True
    max_duration_seconds: float = 30.0
    min_duration_seconds: float = 0.5


class AudioPreprocessor:
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()

    def load_and_preprocess(self, file_path_or_buffer) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Loads an audio file, applies channel mixing, resampling, duration validation,
        silence checks, and amplitude normalization.

        Returns:
            audio (np.ndarray): 1D float32 array in range [-1.0, 1.0].
            metadata (Dict[str, Any]): Dictionary containing sample rate, original duration, warnings.
        """
        warnings = []
        try:
            # Load with soundfile first for exactness
            audio, orig_sr = sf.read(file_path_or_buffer, dtype="float32", always_2d=False)
        except Exception as sf_err:
            try:
                # Fallback to librosa if soundfile fails on non-standard WAV headers
                audio, orig_sr = librosa.load(file_path_or_buffer, sr=None, mono=False)
            except Exception as lib_err:
                raise AudioValidationError(f"Failed to decode audio file. Error: {sf_err}; Fallback: {lib_err}")

        # Check for empty audio
        if audio is None or audio.size == 0:
            raise AudioValidationError("Audio file is empty (zero samples).")

        # Check for NaN / Inf values
        if not np.isfinite(audio).all():
            raise AudioValidationError("Audio signal contains invalid values (NaN or Infinity).")

        # Handle channels (convert multi-channel to mono)
        if audio.ndim > 1:
            # Channels along axis 1 (or 0 if librosa)
            if audio.shape[0] < audio.shape[1] and audio.shape[0] <= 8:
                # shape is (channels, samples)
                audio = np.mean(audio, axis=0)
            else:
                # shape is (samples, channels)
                audio = np.mean(audio, axis=1)
            warnings.append("Multi-channel audio downmixed to mono by averaging channels.")

        original_duration = len(audio) / orig_sr

        # Duration bounds check
        if original_duration < self.config.min_duration_seconds:
            raise AudioValidationError(
                f"Audio recording duration ({original_duration:.2f}s) is shorter than minimum supported ({self.config.min_duration_seconds:.1f}s)."
            )

        if original_duration > self.config.max_duration_seconds:
            # Crop with warning
            audio = audio[: int(self.config.max_duration_seconds * orig_sr)]
            warnings.append(
                f"Audio recording exceeded {self.config.max_duration_seconds:.1f}s and was truncated."
            )

        # Resample to target sample rate if needed
        if orig_sr != self.config.target_sample_rate:
            audio = librosa.resample(
                audio,
                orig_sr=orig_sr,
                target_sr=self.config.target_sample_rate,
                res_type="soxr_hq"
            )

        # Check for digital silence (signal RMS below threshold)
        rms = np.sqrt(np.mean(audio**2) + 1e-12)
        rms_db = 20 * np.log10(rms + 1e-12)
        if rms_db < self.config.silence_threshold_db:
            raise AudioValidationError(
                f"Audio signal is effectively silent (RMS: {rms_db:.1f} dB < {self.config.silence_threshold_db:.1f} dB threshold)."
            )

        # Amplitude normalization (peak normalization with headroom guard)
        if self.config.normalize_amplitude:
            peak = np.max(np.abs(audio))
            if peak > 1e-6:
                audio = audio / (peak + 1e-7) * 0.95

        # Align to target duration (pad with reflection or zero if slightly short, crop if long)
        target_samples = int(self.config.target_sample_rate * self.config.target_duration_seconds)
        if len(audio) < target_samples:
            pad_width = target_samples - len(audio)
            audio = np.pad(audio, (0, pad_width), mode="constant", constant_values=0.0)
            warnings.append(f"Audio was padded to match standard window of {self.config.target_duration_seconds}s.")
        elif len(audio) > target_samples:
            audio = audio[:target_samples]

        metadata = {
            "original_sample_rate": orig_sr,
            "processed_sample_rate": self.config.target_sample_rate,
            "original_duration_seconds": round(original_duration, 3),
            "processed_duration_seconds": round(len(audio) / self.config.target_sample_rate, 3),
            "rms_db": round(float(rms_db), 2),
            "sample_count": len(audio),
            "warnings": warnings,
        }

        return audio.astype(np.float32), metadata
