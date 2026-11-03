"""
Integration tests for FastAPI endpoints.
"""

import io
import pytest
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.core.config import settings
from backend.app.database.session import Base, engine

Base.metadata.create_all(bind=engine)
client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["app_name"] == "AcousticIQ"
    assert "active_device" in data


def test_machines_endpoint():
    response = client.get("/api/machines")
    assert response.status_code == 200
    data = response.json()
    machines = data["machines"]
    machine_ids = [m["id"] for m in machines]
    assert "valve" in machine_ids
    assert "pump" in machine_ids
    assert "fan" in machine_ids
    assert "slider" in machine_ids


def test_analyze_invalid_extension():
    # Attempt upload of text file
    response = client.post(
        "/api/analyze",
        data={"machine_category": "valve"},
        files={"file": ("test.txt", b"not audio data", "text/plain")},
    )
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]


def test_analyze_valid_audio():
    # Generate 2-second valid synthetic audio
    sr = 16000
    t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
    sine = (0.4 * np.sin(2 * np.pi * 350 * t)).astype(np.float32)

    buf = io.BytesIO()
    sf.write(buf, sine, sr, format="WAV")
    buf.seek(0)

    response = client.post(
        "/api/analyze",
        data={"machine_category": "valve", "model_type": "proposed"},
        files={"file": ("test_valve.wav", buf.read(), "audio/wav")},
    )

    assert response.status_code == 200
    data = response.json()
    assert "analysis_id" in data
    assert data["machine_category"] == "valve"
    assert "anomaly_score" in data
    assert "decision_threshold" in data
    assert "is_anomaly" in data
    assert "classification_decision" in data
    assert "visualizations" in data


def test_results_history_endpoint():
    response = client.get("/api/results?page=1&page_size=10")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "results" in data
    assert isinstance(data["results"], list)


def test_metrics_endpoint():
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


def test_models_catalog_endpoint():
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert len(data["models"]) >= 4
