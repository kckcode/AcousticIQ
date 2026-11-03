"""
ML Inference Service for AcousticIQ.
Loads trained checkpoints, handles audio preprocessing, feature extraction,
reconstruction inference, threshold decisioning, and visual spectrogram generation.
"""

import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch

from backend.app.core.config import settings
from backend.app.schemas.analysis import AnalysisResponse, SpectrogramData
from ml.preprocessing.audio import AudioPreprocessor, PreprocessingConfig, AudioValidationError
from ml.features.extractor import DualFeatureExtractor, FeatureConfig
from ml.models.autoencoder import BaselineConvolutionalAutoencoder, ProposedAdaptiveFusionAutoencoder


class InferenceService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocessor = AudioPreprocessor(PreprocessingConfig(
            target_sample_rate=settings.DEFAULT_SAMPLE_RATE,
            target_duration_seconds=settings.TARGET_DURATION_SECONDS,
        ))
        self.feature_extractor = DualFeatureExtractor(FeatureConfig(
            sample_rate=settings.DEFAULT_SAMPLE_RATE,
            n_fft=settings.N_FFT,
            hop_length=settings.HOP_LENGTH,
            n_mels=settings.N_MELS,
            f_min=settings.F_MIN,
            f_max=settings.F_MAX,
        ))
        self._model_cache: Dict[str, Any] = {}

    def get_model(self, machine_category: str, model_type: str = "proposed") -> Tuple[Any, float, Dict[str, Any]]:
        """
        Retrieves cached model or loads from disk.
        Returns:
            model: PyTorch model in eval mode.
            threshold: Calibrated decision threshold.
            meta: Metadata dictionary.
        """
        cache_key = f"{machine_category}_{model_type}"
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        ckpt_file = f"{machine_category}_{model_type}_fusion.pt" if model_type == "proposed" else f"{machine_category}_baseline_ae.pt"
        ckpt_path = settings.MODEL_REGISTRY_DIR / ckpt_file

        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"No trained checkpoint found for machine '{machine_category}' and model '{model_type}' at {ckpt_path}. "
                f"Please run the training pipeline first: python scripts/train_and_evaluate.py --machine {machine_category}"
            )

        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        threshold = checkpoint.get("threshold", 0.05)

        if model_type == "proposed":
            model = ProposedAdaptiveFusionAutoencoder().to(self.device)
        else:
            model = BaselineConvolutionalAutoencoder().to(self.device)

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        meta_file = settings.MODEL_REGISTRY_DIR / f"{machine_category}_{model_type}_meta.json"
        meta = {}
        if meta_file.exists():
            with open(meta_file, "r") as f:
                meta = json.load(f)

        self._model_cache[cache_key] = (model, threshold, meta)
        return model, threshold, meta

    def downsample_matrix(self, matrix: np.ndarray, target_rows: int = 32, target_cols: int = 64) -> List[List[float]]:
        """Downsamples 2D spectrogram matrix for efficient web visualization."""
        rows, cols = matrix.shape
        row_indices = np.linspace(0, rows - 1, min(rows, target_rows)).astype(int)
        col_indices = np.linspace(0, cols - 1, min(cols, target_cols)).astype(int)
        sub = matrix[np.ix_(row_indices, col_indices)]
        return [[round(float(v), 3) for v in row] for row in sub]

    def analyze_audio(
        self,
        audio_path_or_buffer,
        filename: str,
        machine_category: str = "valve",
        model_type: str = "proposed",
        is_synthetic: bool = False,
    ) -> Dict[str, Any]:
        """
        Full inference pipeline:
        Audio loading -> Validation -> Feature extraction -> Model inference -> Anomaly scoring.
        """
        start_time = time.perf_counter()

        # Step 1: Preprocessing & Quality Checks
        audio, prep_meta = self.preprocessor.load_and_preprocess(audio_path_or_buffer)

        # Step 2: Feature Extraction (Mel + Inverse-Mel)
        mel, inv_mel, feat_info = self.feature_extractor.extract(audio)

        # Step 3: Model Inference & Anomaly Scoring
        model, threshold, meta = self.get_model(machine_category, model_type)

        with torch.no_grad():
            x_mel = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(self.device)
            if model_type == "proposed":
                x_inv = torch.from_numpy(inv_mel).unsqueeze(0).unsqueeze(0).to(self.device)
                score_tensor, err_map_tensor, alpha_tensor = model.compute_anomaly_score(x_mel, x_inv)
                anomaly_score = float(score_tensor.cpu().item())
                adaptive_gate_alpha = float(alpha_tensor.cpu().item())
            else:
                score_tensor, err_map_tensor = model.compute_anomaly_score(x_mel)
                anomaly_score = float(score_tensor.cpu().item())
                adaptive_gate_alpha = 1.0

        is_anomaly = bool(anomaly_score >= threshold)
        decision = "Potential Acoustic Anomaly" if is_anomaly else "Normal Acoustic Pattern"

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Confidence rating heuristic based on distance from threshold
        margin = abs(anomaly_score - threshold) / (threshold + 1e-7)
        if margin > 0.5:
            confidence = "High"
        elif margin > 0.2:
            confidence = "Moderate"
        else:
            confidence = "Borderline (Recommend manual inspection)"

        # Downsample waveform for UI playback scrub (e.g. 300 points)
        downsample_factor = max(1, len(audio) // 300)
        waveform_downsampled = [round(float(v), 3) for v in audio[::downsample_factor]]
        waveform_times = [
            round(float(i * downsample_factor / settings.DEFAULT_SAMPLE_RATE), 3)
            for i in range(len(waveform_downsampled))
        ]

        # Spectrogram visualization data
        mel_grid = self.downsample_matrix(mel, target_rows=32, target_cols=64)
        inv_grid = self.downsample_matrix(inv_mel, target_rows=32, target_cols=64)

        result = {
            "filename": filename,
            "machine_category": machine_category,
            "sample_rate": settings.DEFAULT_SAMPLE_RATE,
            "duration_seconds": round(len(audio) / settings.DEFAULT_SAMPLE_RATE, 2),
            "model_version": meta.get("model_type", model_type),
            "model_architecture": meta.get("architecture", type(model).__name__),
            "anomaly_score": round(anomaly_score, 6),
            "decision_threshold": round(threshold, 6),
            "is_anomaly": is_anomaly,
            "classification_decision": decision,
            "processing_time_ms": round(elapsed_ms, 2),
            "confidence_rating": confidence,
            "feature_config": {
                "n_mels": settings.N_MELS,
                "n_fft": settings.N_FFT,
                "hop_length": settings.HOP_LENGTH,
                "adaptive_gate_alpha": round(adaptive_gate_alpha, 4),
            },
            "warnings": prep_meta.get("warnings", []),
            "limitations_note": (
                "Anomaly detection indicates deviation from baseline acoustic normality; "
                "it does NOT pinpoint root-cause fault mechanisms or estimate remaining useful life (RUL)."
            ),
            "visualizations": {
                "frequencies": [round(float(f), 1) for f in np.linspace(0, settings.DEFAULT_SAMPLE_RATE / 2, 32)],
                "times": [round(float(t), 2) for t in np.linspace(0, settings.TARGET_DURATION_SECONDS, 64)],
                "magnitudes": mel_grid,
                "mel_spectrogram": mel_grid,
                "inverse_mel_spectrogram": inv_grid,
                "waveform_downsampled": waveform_downsampled,
                "waveform_times": waveform_times,
            },
            "is_synthetic_fixture": is_synthetic or filename.startswith("synthetic_"),
        }

        return result


inference_service = InferenceService()
