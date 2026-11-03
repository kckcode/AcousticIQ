"""
Feature Extraction Pipeline for AcousticIQ.
Implements:
1. Standard Log-Mel Spectrogram representation.
2. Mathematically grounded Inverse-Mel-inspired linear frequency reconstruction.
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import numpy as np
import librosa
from scipy.linalg import pinv


@dataclass
class FeatureConfig:
    sample_rate: int = 16000
    n_fft: int = 1024
    hop_length: int = 512
    n_mels: int = 64
    f_min: float = 0.0
    f_max: float = 8000.0
    power: float = 2.0
    top_db: float = 80.0
    standardize: bool = True


class LogMelExtractor:
    """Extracts standardized Log-Mel Spectrogram features."""
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        # Precompute mel filterbank
        self.mel_basis = librosa.filters.mel(
            sr=self.config.sample_rate,
            n_fft=self.config.n_fft,
            n_mels=self.config.n_mels,
            fmin=self.config.f_min,
            fmax=self.config.f_max,
        )

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Extracts log-Mel spectrogram.
        Returns array of shape (n_mels, time_frames).
        """
        stft = librosa.stft(
            audio,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            window="hann",
            center=True,
        )
        power_spec = np.abs(stft) ** self.config.power
        mel_spec = np.dot(self.mel_basis, power_spec)
        mel_spec = np.maximum(mel_spec, 1e-10)
        log_mel = librosa.power_to_db(mel_spec, ref=np.max, top_db=self.config.top_db)

        if self.config.standardize:
            mean = np.mean(log_mel)
            std = np.std(log_mel) + 1e-7
            log_mel = (log_mel - mean) / std

        return log_mel.astype(np.float32)


class InverseMelExtractor:
    """
    Extracts an Inverse-Mel-inspired complementary acoustic representation.
    
    Mathematical Formulation:
    Let M be the Mel filterbank basis matrix of shape (n_mels, 1 + n_fft//2).
    Standard Mel projection maps linear STFT bins to non-linearly grouped Mel bands:
        S_mel = M @ S_linear
    The Moore-Penrose pseudo-inverse M_dagger = pinv(M) produces an approximate reconstruction
    of the linear-frequency spectral power distribution:
        S_hat_lin = M_dagger @ S_mel
    Since M has a non-trivial nullspace, M_dagger @ S_mel projects into the subspace spanned
    by the Mel filters. We downsample or pool this reconstructed linear distribution back to
    n_mels bins using an orthogonal inverted linear spacing, capturing fine harmonic structures
    in the upper acoustic spectrum that standard log-Mel integration compresses.
    """
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self.mel_basis = librosa.filters.mel(
            sr=self.config.sample_rate,
            n_fft=self.config.n_fft,
            n_mels=self.config.n_mels,
            fmin=self.config.f_min,
            fmax=self.config.f_max,
        )
        # Compute pseudo-inverse of the mel basis (shape: n_fft//2 + 1, n_mels)
        self.mel_pinv = pinv(self.mel_basis)
        
        # Linear frequency compression matrix to project (1 + n_fft//2) -> n_mels
        n_linear_bins = self.config.n_fft // 2 + 1
        self.linear_projector = np.zeros((self.config.n_mels, n_linear_bins), dtype=np.float32)
        step = n_linear_bins / self.config.n_mels
        for i in range(self.config.n_mels):
            start = int(i * step)
            end = int((i + 1) * step) if i < self.config.n_mels - 1 else n_linear_bins
            self.linear_projector[i, start:end] = 1.0 / max(1, end - start)

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Extracts Inverse-Mel reconstructed representation.
        Returns array of shape (n_mels, time_frames).
        """
        stft = librosa.stft(
            audio,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            window="hann",
            center=True,
        )
        power_spec = np.abs(stft) ** self.config.power
        # Mel projection
        mel_spec = np.dot(self.mel_basis, power_spec)
        mel_spec = np.maximum(mel_spec, 1e-10)
        
        # Approximate linear frequency reconstruction via pseudo-inverse
        reconstructed_linear = np.dot(self.mel_pinv, mel_spec)
        reconstructed_linear = np.maximum(reconstructed_linear, 1e-10)
        
        # Project back to compatible feature dimension (n_mels, time_frames)
        inv_mel_features = np.dot(self.linear_projector, reconstructed_linear)
        inv_mel_db = librosa.power_to_db(inv_mel_features, ref=np.max, top_db=self.config.top_db)

        if self.config.standardize:
            mean = np.mean(inv_mel_db)
            std = np.std(inv_mel_db) + 1e-7
            inv_mel_db = (inv_mel_db - mean) / std

        return inv_mel_db.astype(np.float32)


class DualFeatureExtractor:
    """Extracts both Mel and Inverse-Mel representations with validated alignment."""
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self.mel_extractor = LogMelExtractor(self.config)
        self.inv_mel_extractor = InverseMelExtractor(self.config)

    def extract(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        mel = self.mel_extractor.extract(audio)
        inv_mel = self.inv_mel_extractor.extract(audio)
        
        assert mel.shape == inv_mel.shape, f"Feature dimension mismatch: {mel.shape} vs {inv_mel.shape}"
        
        info = {
            "shape": list(mel.shape),
            "n_mels": self.config.n_mels,
            "time_frames": mel.shape[1],
            "mel_mean": float(np.mean(mel)),
            "mel_std": float(np.std(mel)),
            "inv_mel_mean": float(np.mean(inv_mel)),
            "inv_mel_std": float(np.std(inv_mel)),
        }
        return mel, inv_mel, info
