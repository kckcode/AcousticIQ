"""
SQLAlchemy ORM models for AcousticIQ.
"""

import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, Text
from .session import Base


class AnalysisRecord(Base):
    __tablename__ = "analysis_records"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    filename = Column(String(255), nullable=False)
    machine_category = Column(String(64), nullable=False, index=True)
    sample_rate = Column(Integer, nullable=False)
    duration_seconds = Column(Float, nullable=False)
    
    # Model details
    model_version = Column(String(64), nullable=False)
    model_architecture = Column(String(64), nullable=False)
    
    # Quantitative Scores
    anomaly_score = Column(Float, nullable=False)
    decision_threshold = Column(Float, nullable=False)
    is_anomaly = Column(Boolean, nullable=False, index=True)
    classification_decision = Column(String(64), nullable=False) # e.g. "Normal Acoustic Pattern" or "Potential Acoustic Anomaly"
    
    # Operational Metrics
    processing_time_ms = Column(Float, nullable=False)
    feature_config_json = Column(Text, nullable=True)
    warnings_json = Column(Text, nullable=True)
    
    # Optional audio path (if retention is enabled)
    audio_path = Column(String(512), nullable=True)
    is_synthetic = Column(Boolean, default=False)
