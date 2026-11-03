"""
Comprehensive Integration Tests for AcousticIQ Dataset Pipeline.
Tests:
- scripts/extract_dataset.py
- scripts/validate_dataset.py
- scripts/preprocess_dataset.py
- scripts/download_mimii.py (catalog and CLI interfaces)
"""

import os
import json
import stat
import zipfile
import tempfile
import numpy as np
import soundfile as sf
import pytest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
import sys
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.extract_dataset import extract_archive, make_readonly
from scripts.validate_dataset import (
    validate_single_audio_file,
    scan_and_validate_dataset,
)
from scripts.preprocess_dataset import preprocess_dataset, parse_metadata_from_path
from scripts.download_mimii import MIMII_ARCHIVES, print_catalog


@pytest.fixture
def temp_dataset_env(tmp_path):
    """Creates an isolated temporary directory tree for testing dataset operations."""
    raw_dir = tmp_path / "raw" / "MIMII"
    interim_dir = tmp_path / "interim"
    processed_dir = tmp_path / "processed"
    metadata_dir = tmp_path / "metadata"

    raw_dir.mkdir(parents=True)
    interim_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)

    # Generate synthetic 16kHz audio signals for testing
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)

    # 1. Normal valve sound (harmonic tone + low noise)
    sig_normal = (0.2 * np.sin(2 * np.pi * 350 * t) + 0.02 * np.random.randn(sr)).astype(np.float32)

    # 2. Anomaly valve sound (harsh friction tone + bursts)
    sig_anomaly = (0.5 * np.sin(2 * np.pi * 1200 * t) + 0.1 * np.random.randn(sr)).astype(np.float32)

    # Create dummy raw hierarchy
    valve_norm_dir = raw_dir / "valve" / "id_00" / "normal"
    valve_anom_dir = raw_dir / "valve" / "id_00" / "anomaly"
    valve_norm_dir.mkdir(parents=True)
    valve_anom_dir.mkdir(parents=True)

    for i in range(10):
        sf.write(valve_norm_dir / f"norm_{i:04d}.wav", sig_normal, sr, subtype="PCM_16")

    for i in range(4):
        sf.write(valve_anom_dir / f"anom_{i:04d}.wav", sig_anomaly, sr, subtype="PCM_16")

    return {
        "root": tmp_path,
        "raw_dir": raw_dir,
        "interim_dir": interim_dir,
        "processed_dir": processed_dir,
        "metadata_dir": metadata_dir,
        "sr": sr,
    }


def test_mimii_catalog_listing():
    """Verifies that all 8 major MIMII archives are cataloged with official Zenodo URLs."""
    assert len(MIMII_ARCHIVES) == 8
    assert "valve_6dB" in MIMII_ARCHIVES
    assert "pump_6dB" in MIMII_ARCHIVES
    assert "fan_6dB" in MIMII_ARCHIVES
    assert "slider_6dB" in MIMII_ARCHIVES

    for key, item in MIMII_ARCHIVES.items():
        assert "zenodo.org" in item["url"]
        assert item["size_gb"] > 0
        assert item["machine"] in ["valve", "pump", "fan", "slider"]


def test_extract_dataset_archive(tmp_path):
    """Tests archive extraction, category routing, and read-only permission setting."""
    zip_path = tmp_path / "mock_archive.zip"
    dest_raw_dir = tmp_path / "raw" / "MIMII"

    # Create a mock zip containing a MIMII path structure
    with zipfile.ZipFile(zip_path, "w") as z:
        sr = 16000
        data = (0.1 * np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, sr))).astype(np.float32)
        wav_buf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wav_buf.close()
        sf.write(wav_buf.name, data, sr, subtype="PCM_16")

        z.write(wav_buf.name, arcname="6_dB_valve/valve/id_00/normal/00000001.wav")
        z.write(wav_buf.name, arcname="6_dB_valve/valve/id_00/anomaly/00000001.wav")
        os.unlink(wav_buf.name)

    res = extract_archive(zip_path, dest_raw_dir, enforce_readonly=True)
    assert res["total_extracted"] == 2
    assert "valve" in res["categories"]

    extracted_normal = dest_raw_dir / "valve" / "id_00" / "normal" / "00000001.wav"
    extracted_anomaly = dest_raw_dir / "valve" / "id_00" / "anomaly" / "00000001.wav"

    assert extracted_normal.exists()
    assert extracted_anomaly.exists()

    # Verify read-only permission flag on extracted raw files
    file_stat = os.stat(extracted_normal)
    # File should not be writable by user (or readonly attribute on Windows)
    assert not (file_stat.st_mode & stat.S_IWOTH)


def test_validate_single_audio_file(temp_dataset_env, tmp_path):
    """Tests single file audio validation on valid, corrupt, empty, and silent files."""
    valid_file = temp_dataset_env["raw_dir"] / "valve" / "id_00" / "normal" / "norm_0000.wav"
    is_valid, stats, error = validate_single_audio_file(valid_file)
    assert is_valid is True
    assert error is None
    assert stats["sample_rate"] == 16000
    assert stats["duration_sec"] == 1.0

    # Test Empty file (0 bytes)
    empty_file = tmp_path / "empty.wav"
    empty_file.write_bytes(b"")
    is_valid, _, error = validate_single_audio_file(empty_file)
    assert is_valid is False
    assert "Empty" in error

    # Test Silent file
    silent_file = tmp_path / "silent.wav"
    sf.write(silent_file, np.zeros(16000, dtype=np.float32), 16000, subtype="PCM_16")
    is_valid, _, error = validate_single_audio_file(silent_file)
    assert is_valid is False
    assert "Silent" in error

    # Test NaN audio
    nan_file = tmp_path / "nan.wav"
    nan_arr = np.zeros(16000, dtype=np.float32)
    nan_arr[10] = np.nan
    sf.write(nan_file, nan_arr, 16000, subtype="FLOAT")
    is_valid, _, error = validate_single_audio_file(nan_file)
    assert is_valid is False
    assert "NaN" in error


def test_scan_and_validate_dataset(temp_dataset_env):
    """Tests batch scanning and validation report generation."""
    report = scan_and_validate_dataset(
        raw_dir=temp_dataset_env["raw_dir"],
        metadata_dir=temp_dataset_env["metadata_dir"],
    )

    assert report["status"] == "PASSED"
    assert report["total_files"] == 14
    assert report["valid_files"] == 14
    assert report["corrupt_files"] == 0
    assert "valve" in report["categories"]
    assert report["categories"]["valve"]["normal"] == 10
    assert report["categories"]["valve"]["anomaly"] == 4

    # Verify report JSON output
    report_file = temp_dataset_env["metadata_dir"] / "validation_report.json"
    assert report_file.exists()
    with open(report_file, "r") as f:
        data = json.load(f)
    assert data["total_files"] == 14


def test_dataset_preprocessing_and_manifest_generation(temp_dataset_env):
    """
    Tests complete preprocessing pipeline:
    - Standardizes audio into interim/
    - Extracts dual features into processed/
    - Writes zero-leakage split manifests in metadata/
    """
    result = preprocess_dataset(
        raw_dir=temp_dataset_env["raw_dir"],
        interim_dir=temp_dataset_env["interim_dir"],
        processed_dir=temp_dataset_env["processed_dir"],
        metadata_dir=temp_dataset_env["metadata_dir"],
        target_sr=16000,
        target_duration=1.0,
        n_mels=64,
        train_ratio=0.60,
        val_ratio=0.20,
        test_ratio=0.20,
    )

    assert result["status"] == "COMPLETED"
    assert result["total_files"] == 14
    assert result["processed_count"] == 14

    # 1. Check interim audio directory
    interim_wavs = list(temp_dataset_env["interim_dir"].glob("**/*_std.wav"))
    assert len(interim_wavs) == 14

    # 2. Check processed feature arrays
    processed_npz = list(temp_dataset_env["processed_dir"].glob("**/*.npz"))
    assert len(processed_npz) == 14

    # Inspect one processed file
    sample_npz = np.load(processed_npz[0])
    assert "log_mel" in sample_npz
    assert "inv_mel" in sample_npz
    assert "label" in sample_npz
    assert sample_npz["log_mel"].shape[0] == 64
    assert sample_npz["inv_mel"].shape[0] == 64

    # 3. Check split manifests
    meta_dir = temp_dataset_env["metadata_dir"]
    train_manifest = meta_dir / "train_manifest.json"
    val_manifest = meta_dir / "val_manifest.json"
    test_manifest = meta_dir / "test_manifest.json"
    all_manifest = meta_dir / "dataset_manifest.json"

    assert train_manifest.exists()
    assert val_manifest.exists()
    assert test_manifest.exists()
    assert all_manifest.exists()

    with open(train_manifest, "r") as f:
        tr = json.load(f)
    with open(val_manifest, "r") as f:
        vl = json.load(f)
    with open(test_manifest, "r") as f:
        ts = json.load(f)

    # CRITICAL CHECK: Training set MUST contain zero anomalies (semi-supervised anomaly detection)
    train_labels = [r["label"] for r in tr["records"]]
    assert all(label == 0 for label in train_labels), "Data leakage: anomaly found in training manifest!"
    assert len(train_labels) == 6  # 60% of 10 normal sounds

    # Validation & Test sets should contain both normal and anomaly sounds
    val_labels = [r["label"] for r in vl["records"]]
    test_labels = [r["label"] for r in ts["records"]]

    assert 1 in val_labels, "Validation manifest missing anomaly calibration samples"
    assert 1 in test_labels, "Test manifest missing evaluation anomaly samples"
    assert 0 in val_labels, "Validation manifest missing normal samples"
    assert 0 in test_labels, "Test manifest missing normal samples"
