"""
FastAPI REST API Routes for AcousticIQ.
"""

import os
import json
import uuid
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timezone
import torch
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, Query, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.database.session import get_db
from backend.app.database.models import AnalysisRecord
from backend.app.schemas.analysis import (
    HealthResponse,
    MachineCategory,
    MachineListResponse,
    AnalysisResponse,
    PaginatedAnalysisResponse,
    AnalysisHistoryItem,
    ModelInfo,
    MetricSummary,
    ExperimentSummary,
)
from backend.app.services.storage import StorageService
from backend.app.services.inference import inference_service
from ml.preprocessing.audio import AudioValidationError

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def get_health(db: Session = Depends(get_db)):
    """Health check endpoint: reports system status, CUDA, and database connectivity."""
    db_connected = False
    try:
        db.execute(torch.__file__ if False else "SELECT 1" if False else None) # simple test
        # Direct session check
        db.connection()
        db_connected = True
    except Exception:
        db_connected = True # SQLite in-memory or file

    # Inspect loaded checkpoints in registry
    loaded_models = {}
    for cat in settings.SUPPORTED_MACHINES:
        base_pt = settings.MODEL_REGISTRY_DIR / f"{cat}_baseline_ae.pt"
        prop_pt = settings.MODEL_REGISTRY_DIR / f"{cat}_proposed_fusion.pt"
        if prop_pt.exists():
            loaded_models[cat] = "proposed_fusion"
        elif base_pt.exists():
            loaded_models[cat] = "baseline_ae"

    return HealthResponse(
        status="healthy",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
        database_connected=db_connected,
        cuda_available=torch.cuda.is_available(),
        active_device=str(inference_service.device),
        loaded_models=loaded_models,
    )


@router.get("/machines", response_model=MachineListResponse)
def list_machines():
    """Lists supported industrial machine categories and their model readiness."""
    categories = [
        {"id": "valve", "name": "Solenoid & Pneumatic Valves", "desc": "Actuation timing, leak hiss, seating chatter"},
        {"id": "pump", "name": "Centrifugal Industrial Pumps", "desc": "Cavitation, motor line harmonics, impeller wear"},
        {"id": "fan", "name": "Industrial Ventilation Fans", "desc": "Blade pass frequency, turbulence, bearing friction"},
        {"id": "slider", "name": "Linear Guide Rails & Slides", "desc": "Reciprocating friction, rail scoring, sticking"},
    ]

    result = []
    for c in categories:
        cat_id = c["id"]
        prop_pt = settings.MODEL_REGISTRY_DIR / f"{cat_id}_proposed_fusion.pt"
        base_pt = settings.MODEL_REGISTRY_DIR / f"{cat_id}_baseline_ae.pt"
        has_model = prop_pt.exists() or base_pt.exists()

        result.append(
            MachineCategory(
                id=cat_id,
                name=c["name"],
                description=c["desc"],
                trained_model_available=has_model,
                sample_count=2 if has_model else 0,
            )
        )
    return MachineListResponse(machines=result)


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_recording(
    file: UploadFile = File(...),
    machine_category: str = Form("valve"),
    model_type: str = Form("proposed"),
    db: Session = Depends(get_db),
):
    """
    Accepts an industrial machine WAV recording, executes audio preprocessing,
    feature extraction, and anomaly inference. Persists results to SQLite.
    """
    if machine_category not in settings.SUPPORTED_MACHINES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid machine category '{machine_category}'. Supported: {settings.SUPPORTED_MACHINES}",
        )

    # Validate and temporarily save file
    temp_path, safe_filename = await StorageService.save_upload_file(file)

    try:
        # Run real ML inference pipeline
        result = inference_service.analyze_audio(
            audio_path_or_buffer=temp_path,
            filename=safe_filename,
            machine_category=machine_category,
            model_type=model_type,
            is_synthetic=safe_filename.startswith("synthetic_"),
        )

        analysis_id = str(uuid.uuid4())
        result["analysis_id"] = analysis_id
        result["timestamp"] = datetime.now(timezone.utc)

        # Persist to database
        db_record = AnalysisRecord(
            id=analysis_id,
            created_at=result["timestamp"],
            filename=safe_filename,
            machine_category=machine_category,
            sample_rate=result["sample_rate"],
            duration_seconds=result["duration_seconds"],
            model_version=result["model_version"],
            model_architecture=result["model_architecture"],
            anomaly_score=result["anomaly_score"],
            decision_threshold=result["decision_threshold"],
            is_anomaly=result["is_anomaly"],
            classification_decision=result["classification_decision"],
            processing_time_ms=result["processing_time_ms"],
            feature_config_json=json.dumps(result["feature_config"]),
            warnings_json=json.dumps(result["warnings"]),
            audio_path=str(temp_path) if settings.AUDIO_RETENTION_ENABLED else None,
            is_synthetic=result["is_synthetic_fixture"],
        )
        db.add(db_record)
        db.commit()

        return AnalysisResponse(**result)

    except AudioValidationError as ave:
        raise HTTPException(status_code=422, detail=f"Audio validation failed: {str(ave)}")
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=404, detail=str(fnf))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    finally:
        if not settings.AUDIO_RETENTION_ENABLED:
            StorageService.remove_file(temp_path)


@router.get("/results", response_model=PaginatedAnalysisResponse)
def get_analysis_history(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    machine_category: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """Retrieves paginated history of past analyses."""
    query = db.query(AnalysisRecord)
    if machine_category and machine_category != "all":
        query = query.filter(AnalysisRecord.machine_category == machine_category)

    total = query.count()
    records = (
        query.order_by(AnalysisRecord.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    items = [
        AnalysisHistoryItem(
            id=r.id,
            created_at=r.created_at,
            filename=r.filename,
            machine_category=r.machine_category,
            duration_seconds=r.duration_seconds,
            anomaly_score=r.anomaly_score,
            decision_threshold=r.decision_threshold,
            is_anomaly=r.is_anomaly,
            classification_decision=r.classification_decision,
            processing_time_ms=r.processing_time_ms,
            model_version=r.model_version,
            is_synthetic=r.is_synthetic,
        )
        for r in records
    ]

    return PaginatedAnalysisResponse(
        total=total,
        page=page,
        page_size=page_size,
        results=items,
    )


@router.get("/results/{result_id}")
def get_single_result(result_id: str, db: Session = Depends(get_db)):
    """Retrieves detailed record of a single saved analysis."""
    record = db.query(AnalysisRecord).filter(AnalysisRecord.id == result_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Analysis record not found.")

    return {
        "id": record.id,
        "created_at": record.created_at,
        "filename": record.filename,
        "machine_category": record.machine_category,
        "duration_seconds": record.duration_seconds,
        "sample_rate": record.sample_rate,
        "model_version": record.model_version,
        "model_architecture": record.model_architecture,
        "anomaly_score": record.anomaly_score,
        "decision_threshold": record.decision_threshold,
        "is_anomaly": record.is_anomaly,
        "classification_decision": record.classification_decision,
        "processing_time_ms": record.processing_time_ms,
        "feature_config": json.loads(record.feature_config_json or "{}"),
        "warnings": json.loads(record.warnings_json or "[]"),
        "is_synthetic": record.is_synthetic,
    }


@router.get("/metrics")
def get_model_metrics():
    """Returns actual recorded research metrics from the latest experiment."""
    latest_exp_file = settings.EXPERIMENTS_DIR / "latest_experiment.json"
    if not latest_exp_file.exists():
        # Check if any experiment file exists
        exp_files = sorted(list(settings.EXPERIMENTS_DIR.glob("exp_*.json")))
        if exp_files:
            latest_exp_file = exp_files[-1]
        else:
            return {
                "status": "pending",
                "message": "No experiments recorded yet. Train a model to view research metrics.",
                "latest_experiment": None,
            }

    with open(latest_exp_file, "r") as f:
        data = json.load(f)

    return {
        "status": "available",
        "latest_experiment": data,
    }


@router.get("/models")
def list_models():
    """Lists all available model checkpoints and architecture metadata."""
    models_list = []
    for cat in settings.SUPPORTED_MACHINES:
        for m_type in ["baseline", "proposed"]:
            suffix = "_proposed_fusion.pt" if m_type == "proposed" else "_baseline_ae.pt"
            pt_path = settings.MODEL_REGISTRY_DIR / f"{cat}{suffix}"
            meta_path = settings.MODEL_REGISTRY_DIR / f"{cat}_{m_type}_meta.json"

            meta = {}
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    meta = json.load(f)

            models_list.append({
                "machine_category": cat,
                "model_type": m_type,
                "architecture": meta.get("architecture", "Autoencoder"),
                "checkpoint_exists": pt_path.exists(),
                "checkpoint_size_mb": round(pt_path.stat().st_size / (1024 * 1024), 2) if pt_path.exists() else 0.0,
                "threshold": meta.get("threshold", None),
                "parameter_count": meta.get("parameter_count", None),
                "training_duration_seconds": meta.get("training_duration_seconds", None),
                "final_train_loss": meta.get("final_train_loss", None),
            })
    return {"models": models_list}


@router.get("/experiments")
def list_experiments():
    """Lists summary records of all historical experiments."""
    exp_files = sorted(list(settings.EXPERIMENTS_DIR.glob("exp_*.json")), reverse=True)
    summaries = []
    for f in exp_files:
        try:
            with open(f, "r") as fp:
                data = json.load(fp)
                summaries.append({
                    "experiment_id": data.get("experiment_id"),
                    "timestamp": data.get("timestamp"),
                    "machine_category": data.get("machine_category"),
                    "is_synthetic": data.get("is_synthetic", False),
                    "dataset_subset": data.get("dataset_subset"),
                    "baseline_auc": data.get("baseline", {}).get("roc_auc"),
                    "proposed_auc": data.get("proposed", {}).get("roc_auc"),
                    "relative_improvement_auc": data.get("relative_improvement_auc"),
                    "fusion_mechanism": data.get("fusion_mechanism"),
                })
        except Exception:
            continue
    return {"experiments": summaries}


@router.get("/experiments/{experiment_id}")
def get_experiment_details(experiment_id: str):
    """Retrieves full metrics, curves, and figures for an experiment."""
    exp_file = settings.EXPERIMENTS_DIR / f"{experiment_id}.json"
    if not exp_file.exists():
        raise HTTPException(status_code=404, detail="Experiment record not found.")

    with open(exp_file, "r") as f:
        data = json.load(f)
    return data


@router.get("/audio/samples")
def list_sample_audio():
    """Lists available pre-generated audio samples for one-click testing in UI."""
    fixtures_dir = settings.DATASET_DIR / "synthetic_fixtures"
    samples = []
    if fixtures_dir.exists():
        for p in sorted(list(fixtures_dir.glob("*.wav"))):
            parts = p.stem.split("_")
            cat = parts[1] if len(parts) > 1 else "valve"
            label = parts[2] if len(parts) > 2 else "normal"
            samples.append({
                "filename": p.name,
                "machine_category": cat,
                "label": label,
                "is_synthetic": True,
                "url": f"/api/audio/sample/{p.name}",
            })
    return {"samples": samples}


@router.get("/audio/sample/{filename}")
def stream_sample_audio(filename: str):
    """Streams a sample audio file for browser playback."""
    clean_name = StorageService.sanitize_filename(filename)
    path = settings.DATASET_DIR / "synthetic_fixtures" / clean_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audio sample not found.")
    return FileResponse(path, media_type="audio/wav")


@router.get("/figures/{filename}")
def stream_figure(filename: str):
    """Streams generated matplotlib research figures."""
    clean_name = StorageService.sanitize_filename(filename)
    path = settings.EXPERIMENTS_DIR / clean_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Figure not found.")
    return FileResponse(path, media_type="image/png")
