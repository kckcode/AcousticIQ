"""
Dataset Preprocessing and Feature Extraction Pipeline for AcousticIQ.

Pipeline Steps:
1. READ-ONLY intake of raw recordings from ml/data/raw/MIMII/ (or custom --raw-dir).
2. Standardization to intermediate audio in ml/data/interim/:
   - Resampling to target rate (16,000 Hz)
   - Multi-channel to mono downmixing
   - Fixed-length windowing / padding (10.0 seconds nominal)
   - Peak / RMS normalization
3. Feature extraction to ml/data/processed/:
   - Log-Mel spectrogram (64 mel filterbanks, 1024 FFT, 512 hop)
   - Inverse-Mel pseudo-inverse spectral representation
   - Saved as compressed .npz archives containing tensors and metadata
4. Manifest and split generation in ml/data/metadata/:
   - train_manifest.json: Normal sounds only (zero anomaly contamination, ~60%)
   - val_manifest.json: Normal (~20%) + Anomaly (~50%) for threshold calibration
   - test_manifest.json: Normal (~20%) + Anomaly (~50%) for final unbiased evaluation
   - dataset_manifest.json: Full catalog with labels and split tags
"""

import os
import sys
import json
import random
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
    DEFAULT_INTERIM_DIR = settings.DATASET_INTERIM_DIR
    DEFAULT_PROCESSED_DIR = settings.DATASET_PROCESSED_DIR
    DEFAULT_METADATA_DIR = settings.DATASET_METADATA_DIR
except Exception:
    DEFAULT_RAW_DIR = BASE_DIR / "ml" / "data" / "raw" / "MIMII"
    DEFAULT_INTERIM_DIR = BASE_DIR / "ml" / "data" / "interim"
    DEFAULT_PROCESSED_DIR = BASE_DIR / "ml" / "data" / "processed"
    DEFAULT_METADATA_DIR = BASE_DIR / "ml" / "data" / "metadata"

from ml.preprocessing.audio import AudioPreprocessor, PreprocessingConfig
from ml.features.extractor import DualFeatureExtractor, FeatureConfig


def parse_metadata_from_path(file_path: Path, raw_root: Path) -> Dict[str, str]:
    """
    Extracts category, model_id, condition, and base name from relative path.
    Example path: valve/id_00/normal/00000001.wav
    """
    rel = file_path.relative_to(raw_root)
    parts = rel.parts

    category = "unknown"
    model_id = "id_00"
    condition = "normal"

    for p in parts:
        p_lower = p.lower()
        if p_lower in ["valve", "pump", "fan", "slider"]:
            category = p_lower
        elif "id_" in p_lower:
            model_id = p_lower
        elif p_lower in ["normal", "anomaly"]:
            condition = p_lower

    return {
        "category": category,
        "model_id": model_id,
        "condition": condition,
        "relative_path": str(rel),
        "file_name": file_path.name,
        "stem": file_path.stem,
    }


def preprocess_dataset(
    raw_dir: Path,
    interim_dir: Path,
    processed_dir: Path,
    metadata_dir: Path,
    target_sr: int = 16000,
    target_duration: float = 10.0,
    n_mels: int = 64,
    category_filter: Optional[str] = None,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    test_ratio: float = 0.20,
    seed: int = 42,
    max_files: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Executes full end-to-end preprocessing, interim standardization,
    processed feature tensor extraction, and split manifest generation.
    """
    random.seed(seed)
    np.random.seed(seed)

    raw_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("           ACOUSTICIQ DATASET PREPROCESSING PIPELINE")
    print("=" * 70)
    print(f"Raw Input (Read-Only) : {raw_dir}")
    print(f"Interim Audio (WAV)   : {interim_dir}")
    print(f"Processed Features    : {processed_dir}")
    print(f"Metadata Manifests    : {metadata_dir}")
    print(f"Target Sampling Rate  : {target_sr} Hz")
    print(f"Target Duration       : {target_duration} s")
    print(f"Mel Frequency Bins    : {n_mels}")
    print("=" * 70)

    # Initialize Preprocessor & Extractor
    prep_config = PreprocessingConfig(target_sample_rate=target_sr, target_duration_seconds=target_duration)
    preprocessor = AudioPreprocessor(config=prep_config)
    feat_config = FeatureConfig(sample_rate=target_sr, n_mels=n_mels)
    extractor = DualFeatureExtractor(config=feat_config)

    wav_files = sorted(list(raw_dir.glob("**/*.wav")))
    if category_filter:
        wav_files = [f for f in wav_files if category_filter.lower() in [p.lower() for p in f.parts]]

    if max_files and max_files > 0:
        wav_files = wav_files[:max_files]

    total_files = len(wav_files)
    print(f"\nDiscovered {total_files} audio recordings to preprocess.")

    if total_files == 0:
        print("[Dataset Preprocessor] No audio files found in raw directory. Exiting.")
        return {
            "status": "EMPTY",
            "total_files": 0,
            "processed_count": 0,
        }

    # Structure records by (category, model_id) for stratified splitting
    categorized_records: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]] = {}

    all_processed_records: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for idx, raw_path in enumerate(wav_files, 1):
        try:
            meta = parse_metadata_from_path(raw_path, raw_dir)
            cat = meta["category"]
            model_id = meta["model_id"]
            condition = meta["condition"]
            label = 0 if condition == "normal" else 1

            # 1. Standardize Audio (Interim output)
            audio_np, meta_audio = preprocessor.load_and_preprocess(raw_path)

            rel_subpath = Path(cat) / model_id / condition
            interim_file = interim_dir / rel_subpath / f"{meta['stem']}_std.wav"
            interim_file.parent.mkdir(parents=True, exist_ok=True)

            sf.write(interim_file, audio_np, target_sr, subtype="PCM_16")

            # 2. Extract Acoustic Features (Processed .npz output)
            log_mel, inv_mel, feat_info = extractor.extract(audio_np)

            processed_file = processed_dir / rel_subpath / f"{meta['stem']}.npz"
            processed_file.parent.mkdir(parents=True, exist_ok=True)

            np.savez_compressed(
                processed_file,
                log_mel=log_mel,
                inv_mel=inv_mel,
                label=label,
                machine=cat,
                model_id=model_id,
                sample_rate=target_sr,
                duration=target_duration,
            )

            record = {
                "id": f"{cat}_{model_id}_{condition}_{meta['stem']}",
                "category": cat,
                "model_id": model_id,
                "condition": condition,
                "label": label,
                "raw_path": str(raw_path),
                "interim_path": str(interim_file),
                "processed_path": str(processed_file),
                "feature_shape": list(log_mel.shape),
                "duration_sec": target_duration,
                "sample_rate": target_sr,
            }

            if cat not in categorized_records:
                categorized_records[cat] = {}
            if model_id not in categorized_records[cat]:
                categorized_records[cat][model_id] = {"normal": [], "anomaly": []}

            categorized_records[cat][model_id][condition].append(record)
            all_processed_records.append(record)

        except Exception as e:
            errors.append({"path": str(raw_path), "error": str(e)})

        if idx % 50 == 0 or idx == total_files:
            print(f"\rProcessed: {idx}/{total_files} files (Errors: {len(errors)})", end="", flush=True)

    print("\nFeature extraction and standardization complete.")

    # 3. Create Partition Manifests (Zero Target Leakage)
    # Train: 60% of normal sounds (0% anomaly)
    # Val: 20% of normal sounds + 50% anomaly (for threshold calibration)
    # Test: 20% of normal sounds + 50% anomaly (for unbiased evaluation)

    train_set: List[Dict[str, Any]] = []
    val_set: List[Dict[str, Any]] = []
    test_set: List[Dict[str, Any]] = []

    for cat, models in categorized_records.items():
        for model_id, conds in models.items():
            normals = conds.get("normal", [])
            anomalies = conds.get("anomaly", [])

            random.shuffle(normals)
            random.shuffle(anomalies)

            n_total = len(normals)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)

            train_normals = normals[:n_train]
            val_normals = normals[n_train:n_train + n_val]
            test_normals = normals[n_train + n_val:]

            for r in train_normals:
                r["split"] = "train"
            for r in val_normals:
                r["split"] = "val"
            for r in test_normals:
                r["split"] = "test"

            train_set.extend(train_normals)
            val_set.extend(val_normals)
            test_set.extend(test_normals)

            # Split anomalies equally between validation and test
            a_total = len(anomalies)
            a_half = a_total // 2

            val_anomalies = anomalies[:a_half]
            test_anomalies = anomalies[a_half:]

            for r in val_anomalies:
                r["split"] = "val"
            for r in test_anomalies:
                r["split"] = "test"

            val_set.extend(val_anomalies)
            test_set.extend(test_anomalies)

    # Save Manifest Files
    manifest_paths = {
        "train": metadata_dir / "train_manifest.json",
        "val": metadata_dir / "val_manifest.json",
        "test": metadata_dir / "test_manifest.json",
        "all": metadata_dir / "dataset_manifest.json",
    }

    with open(manifest_paths["train"], "w", encoding="utf-8") as f:
        json.dump({"split": "train", "total": len(train_set), "records": train_set}, f, indent=2)

    with open(manifest_paths["val"], "w", encoding="utf-8") as f:
        json.dump({"split": "val", "total": len(val_set), "records": val_set}, f, indent=2)

    with open(manifest_paths["test"], "w", encoding="utf-8") as f:
        json.dump({"split": "test", "total": len(test_set), "records": test_set}, f, indent=2)

    with open(manifest_paths["all"], "w", encoding="utf-8") as f:
        json.dump({"total_records": len(all_processed_records), "records": all_processed_records}, f, indent=2)

    # Print Summary Table
    print("\n" + "=" * 70)
    print("                     DATASET SPLIT SUMMARY")
    print("=" * 70)
    print(f"Train Manifest : {len(train_set)} records (Normal: {len([r for r in train_set if r['label'] == 0])}, Anomaly: {len([r for r in train_set if r['label'] == 1])}) -> {manifest_paths['train'].name}")
    print(f"Val Manifest   : {len(val_set)} records (Normal: {len([r for r in val_set if r['label'] == 0])}, Anomaly: {len([r for r in val_set if r['label'] == 1])}) -> {manifest_paths['val'].name}")
    print(f"Test Manifest  : {len(test_set)} records (Normal: {len([r for r in test_set if r['label'] == 0])}, Anomaly: {len([r for r in test_set if r['label'] == 1])}) -> {manifest_paths['test'].name}")
    print(f"Full Inventory : {len(all_processed_records)} records -> {manifest_paths['all'].name}")
    print("=" * 70 + "\n")

    return {
        "status": "COMPLETED",
        "total_files": total_files,
        "processed_count": len(all_processed_records),
        "train_count": len(train_set),
        "val_count": len(val_set),
        "test_count": len(test_set),
        "errors_count": len(errors),
        "manifests": {k: str(v) for k, v in manifest_paths.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="AcousticIQ Dataset Preprocessing and Feature Generation")
    parser.add_argument("--raw-dir", type=str, default=str(DEFAULT_RAW_DIR), help="Path to raw recordings")
    parser.add_argument("--interim-dir", type=str, default=str(DEFAULT_INTERIM_DIR), help="Path to interim standardized audio")
    parser.add_argument("--processed-dir", type=str, default=str(DEFAULT_PROCESSED_DIR), help="Path to processed feature tensors")
    parser.add_argument("--metadata-dir", type=str, default=str(DEFAULT_METADATA_DIR), help="Path to metadata manifests")
    parser.add_argument("--category", type=str, default=None, help="Process only a specific machine category")
    parser.add_argument("--max-files", type=int, default=None, help="Limit files for dry run / smoke test")
    parser.add_argument("--sr", type=int, default=16000, help="Target sample rate (Hz)")
    parser.add_argument("--duration", type=float, default=10.0, help="Target audio duration (seconds)")
    parser.add_argument("--n-mels", type=int, default=64, help="Number of Mel frequency bins")

    args = parser.parse_args()

    preprocess_dataset(
        raw_dir=Path(args.raw_dir),
        interim_dir=Path(args.interim_dir),
        processed_dir=Path(args.processed_dir),
        metadata_dir=Path(args.metadata_dir),
        target_sr=args.sr,
        target_duration=args.duration,
        n_mels=args.n_mels,
        category_filter=args.category,
        max_files=args.max_files,
    )


if __name__ == "__main__":
    main()
