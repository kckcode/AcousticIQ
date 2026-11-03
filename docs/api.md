# AcousticIQ API Specification

Base URL: `http://localhost:8000/api`

Interactive Documentation:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Endpoints

### 1. `GET /health`
Returns system health, PyTorch compute hardware, and loaded model checkpoints.

**Response (200 OK):**
```json
{
  "status": "healthy",
  "app_name": "AcousticIQ",
  "version": "1.0.0",
  "environment": "development",
  "database_connected": true,
  "cuda_available": false,
  "active_device": "cpu",
  "loaded_models": {
    "valve": "proposed_fusion",
    "pump": "proposed_fusion",
    "fan": "proposed_fusion",
    "slider": "proposed_fusion"
  }
}
```

### 2. `GET /machines`
Lists supported industrial machine classes and readiness.

### 3. `POST /analyze`
Accepts a WAV audio recording multipart upload and executes inference.

**Form Data:**
- `file`: WAV audio file binary.
- `machine_category`: `valve` | `pump` | `fan` | `slider` (default: `valve`)
- `model_type`: `proposed` | `baseline` (default: `proposed`)

**Response (200 OK):**
```json
{
  "analysis_id": "7f8b9e24-...",
  "timestamp": "2026-11-03T17:25:00Z",
  "filename": "valve_sample.wav",
  "machine_category": "valve",
  "sample_rate": 16000,
  "duration_seconds": 10.0,
  "model_version": "proposed",
  "model_architecture": "ProposedAdaptiveFusionAutoencoder",
  "anomaly_score": 0.03412,
  "decision_threshold": 0.08278,
  "is_anomaly": false,
  "classification_decision": "Normal Acoustic Pattern",
  "processing_time_ms": 11.2,
  "confidence_rating": "High",
  "feature_config": {
    "n_mels": 64,
    "n_fft": 1024,
    "hop_length": 512,
    "adaptive_gate_alpha": 0.582
  },
  "warnings": [],
  "limitations_note": "Anomaly detection indicates deviation from normal baseline; does not estimate RUL.",
  "visualizations": {
    "frequencies": [...],
    "times": [...],
    "magnitudes": [[...]],
    "waveform_downsampled": [...]
  }
}
```

### 4. `GET /results`
Paginated analysis history.
- Query params: `page` (int, default 1), `page_size` (int, default 20), `machine_category` (optional).

### 5. `GET /metrics`
Returns benchmark metrics from the latest experiment run.

### 6. `GET /models`
Catalog of model checkpoints with parameter counts, file sizes, and status.

### 7. `GET /experiments` & `GET /experiments/{id}`
Summaries and complete metric records of research experiments.

### 8. `GET /audio/samples` & `GET /audio/sample/{filename}`
Retrieval and streaming playback for pre-generated calibration fixtures.
