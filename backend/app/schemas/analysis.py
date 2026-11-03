"""
Pydantic Schemas for AcousticIQ API.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class HealthResponse(BaseModel):
    status: str = "healthy"
    app_name: str
    version: str
    environment: str
    database_connected: bool
    cuda_available: bool
    active_device: str
    loaded_models: Dict[str, str]


class MachineCategory(BaseModel):
    id: str
    name: str
    description: str
    trained_model_available: bool
    sample_count: Optional[int] = 0


class MachineListResponse(BaseModel):
    machines: List[MachineCategory]


class SpectrogramData(BaseModel):
    frequencies: List[float] # Center frequencies or mel bin indices
    times: List[float]       # Time steps (seconds)
    magnitudes: List[List[float]] # 2D array [frequency_bins, time_frames]
    mel_spectrogram: Optional[List[List[float]]] = None
    inverse_mel_spectrogram: Optional[List[List[float]]] = None
    waveform_downsampled: Optional[List[float]] = None
    waveform_times: Optional[List[float]] = None


class AnalysisResponse(BaseModel):
    analysis_id: str
    timestamp: datetime
    filename: str
    machine_category: str
    sample_rate: int
    duration_seconds: float
    model_version: str
    model_architecture: str
    anomaly_score: float
    decision_threshold: float
    is_anomaly: bool
    classification_decision: str
    processing_time_ms: float
    confidence_rating: str
    feature_config: Dict[str, Any]
    warnings: List[str]
    limitations_note: str
    visualizations: Optional[SpectrogramData] = None
    is_synthetic_fixture: bool = False


class AnalysisHistoryItem(BaseModel):
    id: str
    created_at: datetime
    filename: str
    machine_category: str
    duration_seconds: float
    anomaly_score: float
    decision_threshold: float
    is_anomaly: bool
    classification_decision: str
    processing_time_ms: float
    model_version: str
    is_synthetic: bool

    model_config = ConfigDict(from_attributes=True)


class PaginatedAnalysisResponse(BaseModel):
    total: int
    page: int
    page_size: int
    results: List[AnalysisHistoryItem]


class MetricSummary(BaseModel):
    roc_auc: float
    p_auc: float
    f1_score: float
    precision: float
    recall: float
    decision_threshold: float
    test_samples: int


class ModelInfo(BaseModel):
    machine_category: str
    model_type: str  # "baseline" or "proposed"
    version: str
    architecture: str
    trained_at: Optional[datetime] = None
    checkpoint_exists: bool
    checkpoint_path: Optional[str] = None
    input_shape: List[int]
    metrics: Optional[MetricSummary] = None


class ExperimentSummary(BaseModel):
    experiment_id: str
    machine_category: str
    timestamp: datetime
    dataset_subset: str
    baseline_metrics: MetricSummary
    proposed_metrics: MetricSummary
    fusion_mechanism: str
    relative_improvement_auc: float
    status: str
    notes: str
