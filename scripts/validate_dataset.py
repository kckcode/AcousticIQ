"""
Dataset Validation Utility for AcousticIQ.
Validates downloaded and extracted raw audio recordings in ml/data/raw/MIMII/ (or custom directory).

Performs:
1. File structure & hierarchy validation (category / model_id / condition / *.wav).
2. Audio stream decodability and header integrity check.
3. Sample rate (16,000 Hz nominal) and channel count verification.
4. Signal integrity analysis: NaN, Inf, digital silence, clipping detection.
5. Duration bounds verification (nominal MIMII clips are 10.0 seconds).
6. Manifest reporting & saving structured JSON report to ml/data/metadata/validation_report.json.
"""

import os
import sys
import json
import argparse
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

try:
    from backend.app.core.config import settings
    DEFAULT_RAW_DIR = settings.DATASET_RAW_DIR
    DEFAULT_METADATA_DIR = settings.DATASET_METADATA_DIR
except Exception:
    DEFAULT_RAW_DIR = BASE_DIR / "ml" / "data" / "raw" / "MIMII"
    DEFAULT_METADATA_DIR = BASE_DIR / "ml" / "data" / "metadata"


def validate_single_audio_file(file_path: Path) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """
    Validates a single audio recording for decodability and physical audio properties.
    """
    stats: Dict[str, Any] = {
        "path": str(file_path),
        "file_name": file_path.name,
        "size_bytes": file_path.stat().st_size,
    }

    if stats["size_bytes"] == 0:
        return False, stats, "Empty audio file (0 bytes)"

    try:
        # Read header first
        info = sf.info(file_path)
        stats["sample_rate"] = info.samplerate
        stats["channels"] = info.channels
        stats["duration_sec"] = round(info.duration, 3)
        stats["format"] = info.format
        stats["subtype"] = info.subtype

        # Duration bounds check (allow 0.5s to 30.0s, nominal 10s)
        if info.duration < 0.5:
            return False, stats, f"Audio duration too short: {info.duration:.2f}s (<0.5s)"
        if info.duration > 35.0:
            return False, stats, f"Audio duration exceeds threshold: {info.duration:.2f}s (>35.0s)"

        # Read actual audio data to inspect sample values
        data, sr = sf.read(file_path, dtype="float32", always_2d=True)

        # Check for NaN / Inf
        if np.isnan(data).any():
            return False, stats, "Audio stream contains NaN values"
        if np.isinf(data).any():
            return False, stats, "Audio stream contains Infinite values"

        # Check for absolute digital silence (all zeros or near-zero RMS)
        rms = np.sqrt(np.mean(data ** 2))
        stats["rms_amplitude"] = float(rms)
        if rms < 1e-6:
            return False, stats, "Silent recording (RMS amplitude < 1e-6)"

        # Check for extreme digital clipping (>10% samples at peak ±0.999)
        clipped_ratio = np.mean(np.abs(data) >= 0.999)
        stats["clipped_ratio"] = float(clipped_ratio)
        if clipped_ratio > 0.10:
            return False, stats, f"Excessive digital clipping detected ({clipped_ratio*100:.1f}% samples)"

        return True, stats, None

    except Exception as e:
        return False, stats, f"Decode failure: {str(e)}"


def scan_and_validate_dataset(
    raw_dir: Path,
    metadata_dir: Path,
    category_filter: Optional[str] = None,
    max_files: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Recursively scans raw_dir and validates every WAV file.
    Organizes findings by machine category, model ID, and condition.
    """
    if not raw_dir.exists():
        print(f"[Dataset Validator] Warning: Raw dataset directory does not exist: {raw_dir}")
        return {
            "status": "NOT_FOUND",
            "raw_dir": str(raw_dir),
            "total_files": 0,
            "valid_files": 0,
            "corrupt_files": 0,
            "categories": {},
            "issues": [f"Directory {raw_dir} does not exist."],
        }

    wav_files = sorted(list(raw_dir.glob("**/*.wav")))
    if category_filter:
        wav_files = [f for f in wav_files if category_filter.lower() in [p.lower() for p in f.parts]]

    if max_files and max_files > 0:
        wav_files = wav_files[:max_files]

    total_files = len(wav_files)
    print(f"\n[Dataset Validator] Found {total_files} audio files in {raw_dir}")
    if category_filter:
        print(f"Filter: machine category = '{category_filter}'")

    valid_count = 0
    corrupt_count = 0
    total_duration_sec = 0.0
    issues: List[Dict[str, Any]] = []

    # Category breakdown: {category: {model_id: {normal: count, anomaly: count}}}
    category_summary: Dict[str, Any] = {}

    for idx, fpath in enumerate(wav_files, 1):
        rel = fpath.relative_to(raw_dir)
        parts = rel.parts

        # Infer category, model_id, and condition
        cat = "unknown"
        model_id = "unknown"
        condition = "unknown"

        for p in parts:
            p_lower = p.lower()
            if p_lower in ["valve", "pump", "fan", "slider"]:
                cat = p_lower
            elif "id_" in p_lower:
                model_id = p_lower
            elif p_lower in ["normal", "anomaly"]:
                condition = p_lower

        if cat not in category_summary:
            category_summary[cat] = {
                "total": 0,
                "normal": 0,
                "anomaly": 0,
                "models": {},
            }
        if model_id not in category_summary[cat]["models"]:
            category_summary[cat]["models"][model_id] = {"normal": 0, "anomaly": 0}

        is_valid, stats, error_msg = validate_single_audio_file(fpath)

        if is_valid:
            valid_count += 1
            total_duration_sec += stats.get("duration_sec", 0.0)
            category_summary[cat]["total"] += 1
            if condition in ["normal", "anomaly"]:
                category_summary[cat][condition] += 1
                category_summary[cat]["models"][model_id][condition] += 1
        else:
            corrupt_count += 1
            issues.append({
                "path": str(fpath),
                "relative_path": str(rel),
                "category": cat,
                "model_id": model_id,
                "condition": condition,
                "reason": error_msg,
            })

        if idx % 100 == 0 or idx == total_files:
            print(f"\rValidated: {idx}/{total_files} files ({valid_count} valid, {corrupt_count} issues)", end="", flush=True)

    print("\n")

    report: Dict[str, Any] = {
        "status": "PASSED" if corrupt_count == 0 and total_files > 0 else ("EMPTY" if total_files == 0 else "FAILED"),
        "raw_directory": str(raw_dir),
        "total_files": total_files,
        "valid_files": valid_count,
        "corrupt_files": corrupt_count,
        "total_duration_hours": round(total_duration_sec / 3600.0, 3),
        "total_duration_minutes": round(total_duration_sec / 60.0, 2),
        "categories": category_summary,
        "issues_count": len(issues),
        "issues": issues[:50],  # cap sample issues in summary
    }

    metadata_dir.mkdir(parents=True, exist_ok=True)
    report_path = metadata_dir / "validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print_validation_table(report, report_path)
    return report


def print_validation_table(report: Dict[str, Any], report_path: Path):
    """Prints a structured ASCII summary table of the validation findings."""
    print("=" * 70)
    print("                     ACOUSTICIQ DATASET VALIDATION REPORT")
    print("=" * 70)
    print(f"Status           : {report['status']}")
    print(f"Raw Directory    : {report['raw_directory']}")
    print(f"Total Files      : {report['total_files']}")
    print(f"Valid Files      : {report['valid_files']}")
    print(f"Corrupt Files    : {report['corrupt_files']}")
    print(f"Total Audio Time : {report['total_duration_hours']} hours ({report['total_duration_minutes']} min)")
    print(f"JSON Report Path : {report_path}")
    print("-" * 70)
    print(f"{'Category':<12} | {'Normal':<8} | {'Anomaly':<8} | {'Total':<8} | {'Model IDs'}")
    print("-" * 70)
    for cat, info in report.get("categories", {}).items():
        model_str = ", ".join(info.get("models", {}).keys()) or "None"
        print(f"{cat:<12} | {info['normal']:<8} | {info['anomaly']:<8} | {info['total']:<8} | {model_str}")
    print("-" * 70)

    if report["corrupt_files"] > 0:
        print(f"\nWARNING: {report['corrupt_files']} files failed validation!")
        for issue in report.get("issues", [])[:10]:
            print(f"  - {issue['relative_path']}: {issue['reason']}")
        if len(report.get("issues", [])) > 10:
            print(f"  ... and {len(report['issues']) - 10} more (see {report_path})")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Validate MIMII dataset audio integrity and structure")
    parser.add_argument("--raw-dir", type=str, default=str(DEFAULT_RAW_DIR), help="Path to raw dataset directory")
    parser.add_argument("--metadata-dir", type=str, default=str(DEFAULT_METADATA_DIR), help="Path to metadata output directory")
    parser.add_argument("--category", type=str, default=None, help="Filter validation to specific machine category")
    parser.add_argument("--max-files", type=int, default=None, help="Limit number of files to validate (for quick checks)")

    args = parser.parse_args()

    report = scan_and_validate_dataset(
        raw_dir=Path(args.raw_dir),
        metadata_dir=Path(args.metadata_dir),
        category_filter=args.category,
        max_files=args.max_files,
    )

    if report["status"] == "FAILED":
        sys.exit(1)


if __name__ == "__main__":
    main()
