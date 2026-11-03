"""
Unified End-to-End Training and Evaluation Pipeline for AcousticIQ.
Executes Phase 3 and Phase 4:
- Preprocesses audio.
- Extracts dual feature representations (Mel + Inverse-Mel).
- Trains Baseline Convolutional Autoencoder.
- Trains Proposed Dual-Branch Adaptive Fusion Autoencoder.
- Evaluates both models fairly on held-out test split.
- Calibrates thresholds on validation set (no leakage into test evaluation).
- Computes ROC-AUC, pAUC (max FPR=0.1), F1-Score, Precision, Recall, Latency.
- Saves checkpoints and metadata.
- Generates reproducible research figures.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np

# Ensure root workspace is on python path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

try:
    from backend.app.core.config import settings
    DEFAULT_RAW_DIR = settings.DATASET_RAW_DIR
    DEFAULT_INTERIM_DIR = settings.DATASET_INTERIM_DIR
    DEFAULT_PROCESSED_DIR = settings.DATASET_PROCESSED_DIR
    DEFAULT_METADATA_DIR = settings.DATASET_METADATA_DIR
    DEFAULT_CHECKPOINT_DIR = settings.MODEL_REGISTRY_DIR
    DEFAULT_EXPERIMENTS_DIR = settings.EXPERIMENTS_DIR
except Exception:
    DEFAULT_RAW_DIR = BASE_DIR / "ml" / "data" / "raw" / "MIMII"
    DEFAULT_INTERIM_DIR = BASE_DIR / "ml" / "data" / "interim"
    DEFAULT_PROCESSED_DIR = BASE_DIR / "ml" / "data" / "processed"
    DEFAULT_METADATA_DIR = BASE_DIR / "ml" / "data" / "metadata"
    DEFAULT_CHECKPOINT_DIR = BASE_DIR / "ml" / "checkpoints"
    DEFAULT_EXPERIMENTS_DIR = BASE_DIR / "ml" / "experiments"

from ml.preprocessing.audio import AudioPreprocessor, PreprocessingConfig
from ml.features.extractor import DualFeatureExtractor, FeatureConfig
from ml.models.autoencoder import BaselineConvolutionalAutoencoder, ProposedAdaptiveFusionAutoencoder
from ml.training.trainer import ModelTrainer
from ml.evaluation.evaluator import ModelEvaluator
from ml.data.synthetic_fixture import generate_synthetic_machine_sound


def prepare_dataset(
    machine_category: str,
    raw_dir: Optional[Path] = None,
    processed_dir: Optional[Path] = None,
    metadata_dir: Optional[Path] = None,
    data_dir: Optional[Path] = None,
    use_synthetic_if_missing: bool = True,
    n_train_normal: int = 40,
    n_val_normal: int = 15,
    n_val_anomaly: int = 15,
    n_test_normal: int = 25,
    n_test_anomaly: int = 25,
) -> Tuple[Dict[str, Any], bool]:
    """
    Prepares training, validation, and test datasets for machine_category.
    Priority order:
    1. Pre-extracted feature arrays (.npz) via split manifests in metadata_dir.
    2. Raw audio recordings in raw_dir (or fallback data_dir).
    3. Calibrated synthetic acoustic fixtures (when real audio has not yet been downloaded).
    """
    raw_dir = raw_dir or DEFAULT_RAW_DIR
    processed_dir = processed_dir or DEFAULT_PROCESSED_DIR
    metadata_dir = metadata_dir or DEFAULT_METADATA_DIR

    train_mels, train_invs = [], []
    val_mels, val_invs, val_labels = [], [], []
    test_mels, test_invs, test_labels = [], [], []

    # Priority 1: Check metadata manifests & processed .npz features
    train_manifest = metadata_dir / "train_manifest.json"
    val_manifest = metadata_dir / "val_manifest.json"
    test_manifest = metadata_dir / "test_manifest.json"

    if train_manifest.exists() and val_manifest.exists() and test_manifest.exists():
        try:
            with open(train_manifest, "r", encoding="utf-8") as f:
                tr_data = json.load(f)
            with open(val_manifest, "r", encoding="utf-8") as f:
                vl_data = json.load(f)
            with open(test_manifest, "r", encoding="utf-8") as f:
                ts_data = json.load(f)

            tr_recs = [r for r in tr_data.get("records", []) if r.get("category") == machine_category]
            vl_recs = [r for r in vl_data.get("records", []) if r.get("category") == machine_category]
            ts_recs = [r for r in ts_data.get("records", []) if r.get("category") == machine_category]

            if len(tr_recs) > 0 and len(ts_recs) > 0:
                print(f"[Dataset] Loading processed feature tensors from {processed_dir} for {machine_category}...")
                for r in tr_recs:
                    npz = np.load(r["processed_path"])
                    train_mels.append(npz["log_mel"])
                    train_invs.append(npz["inv_mel"])

                for r in vl_recs:
                    npz = np.load(r["processed_path"])
                    val_mels.append(npz["log_mel"])
                    val_invs.append(npz["inv_mel"])
                    val_labels.append(int(r.get("label", 0)))

                for r in ts_recs:
                    npz = np.load(r["processed_path"])
                    test_mels.append(npz["log_mel"])
                    test_invs.append(npz["inv_mel"])
                    test_labels.append(int(r.get("label", 0)))

                print(f"[Dataset] Successfully loaded manifests: Train={len(train_mels)}, Val={len(val_mels)}, Test={len(test_mels)}")
                return {
                    "train_mel": train_mels,
                    "train_inv": train_invs,
                    "val_mel": val_mels,
                    "val_inv": val_invs,
                    "val_labels": val_labels,
                    "test_mel": test_mels,
                    "test_inv": test_invs,
                    "test_labels": test_labels,
                }, False
        except Exception as e:
            print(f"[Dataset] Manifest loading encountered error ({e}), falling back to raw scan.")

    # Priority 2: Check raw audio directory
    search_dirs = [raw_dir / machine_category, raw_dir]
    if data_dir:
        search_dirs.insert(0, data_dir / machine_category)
        search_dirs.insert(1, data_dir)

    found_raw_normal = []
    found_raw_anomaly = []
    for sdir in search_dirs:
        if sdir.exists():
            for p in sdir.glob("**/*.wav"):
                parts_lower = [part.lower() for part in p.parts]
                if machine_category in parts_lower or sdir.name.lower() == machine_category:
                    if "normal" in parts_lower:
                        found_raw_normal.append(p)
                    elif "anomaly" in parts_lower:
                        found_raw_anomaly.append(p)

    if len(found_raw_normal) >= 5:
        print(f"[Dataset] Loading and extracting features from raw audio in {raw_dir}...")
        preprocessor = AudioPreprocessor()
        extractor = DualFeatureExtractor()

        n_norm = len(found_raw_normal)
        n_tr = int(n_norm * 0.6)
        n_vl = int(n_norm * 0.2)

        for p in found_raw_normal[:n_tr]:
            audio, _ = preprocessor.load_and_preprocess(p)
            mel, inv, _ = extractor.extract(audio)
            train_mels.append(mel)
            train_invs.append(inv)

        for p in found_raw_normal[n_tr:n_tr + n_vl]:
            audio, _ = preprocessor.load_and_preprocess(p)
            mel, inv, _ = extractor.extract(audio)
            val_mels.append(mel)
            val_invs.append(inv)
            val_labels.append(0)

        for p in found_raw_normal[n_tr + n_vl:]:
            audio, _ = preprocessor.load_and_preprocess(p)
            mel, inv, _ = extractor.extract(audio)
            test_mels.append(mel)
            test_invs.append(inv)
            test_labels.append(0)

        n_anom = len(found_raw_anomaly)
        n_anom_val = n_anom // 2
        for p in found_raw_anomaly[:n_anom_val]:
            audio, _ = preprocessor.load_and_preprocess(p)
            mel, inv, _ = extractor.extract(audio)
            val_mels.append(mel)
            val_invs.append(inv)
            val_labels.append(1)

        for p in found_raw_anomaly[n_anom_val:]:
            audio, _ = preprocessor.load_and_preprocess(p)
            mel, inv, _ = extractor.extract(audio)
            test_mels.append(mel)
            test_invs.append(inv)
            test_labels.append(1)

        return {
            "train_mel": train_mels,
            "train_inv": train_invs,
            "val_mel": val_mels,
            "val_inv": val_invs,
            "val_labels": val_labels,
            "test_mel": test_mels,
            "test_inv": test_invs,
            "test_labels": test_labels,
        }, False

    # Priority 3: Fall back to calibrated synthetic fixtures
    if use_synthetic_if_missing:
        print(f"[Dataset] Genuine MIMII audio not found in {raw_dir}.")
        print(f"[Dataset] Generating calibrated multi-condition synthetic dataset ({machine_category})...")
        print("  NOTE: Explicitly marked as synthetic fixtures for testing & verification.")

        extractor = DualFeatureExtractor()

        # Normal training recordings
        for i in range(n_train_normal):
            audio, _ = generate_synthetic_machine_sound(machine_category, is_anomaly=False, seed=1000 + i)
            mel, inv, _ = extractor.extract(audio)
            train_mels.append(mel)
            train_invs.append(inv)

        # Validation set (normal + anomalies)
        for i in range(n_val_normal):
            audio, _ = generate_synthetic_machine_sound(machine_category, is_anomaly=False, seed=2000 + i)
            mel, inv, _ = extractor.extract(audio)
            val_mels.append(mel)
            val_invs.append(inv)
            val_labels.append(0)

        for i in range(n_val_anomaly):
            audio, _ = generate_synthetic_machine_sound(machine_category, is_anomaly=True, seed=3000 + i)
            mel, inv, _ = extractor.extract(audio)
            val_mels.append(mel)
            val_invs.append(inv)
            val_labels.append(1)

        # Test set (independent seeds)
        for i in range(n_test_normal):
            audio, _ = generate_synthetic_machine_sound(machine_category, is_anomaly=False, seed=4000 + i)
            mel, inv, _ = extractor.extract(audio)
            test_mels.append(mel)
            test_invs.append(inv)
            test_labels.append(0)

        for i in range(n_test_anomaly):
            audio, _ = generate_synthetic_machine_sound(machine_category, is_anomaly=True, seed=5000 + i)
            mel, inv, _ = extractor.extract(audio)
            test_mels.append(mel)
            test_invs.append(inv)
            test_labels.append(1)

    data = {
        "train_mel": train_mels,
        "train_inv": train_invs,
        "val_mel": val_mels,
        "val_inv": val_invs,
        "val_labels": val_labels,
        "test_mel": test_mels,
        "test_inv": test_invs,
        "test_labels": test_labels,
    }
    return data, True


def run_experiment(
    machine_category: str = "valve",
    epochs: int = 20,
    batch_size: int = 16,
    raw_dir: Path = DEFAULT_RAW_DIR,
    interim_dir: Path = DEFAULT_INTERIM_DIR,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    metadata_dir: Path = DEFAULT_METADATA_DIR,
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
    experiments_dir: Path = DEFAULT_EXPERIMENTS_DIR,
    data_dir: Optional[Path] = None,
):
    print(f"\n========================================================")
    print(f" AcousticIQ Training & Evaluation Pipeline: {machine_category.upper()}")
    print(f"========================================================")

    data, is_synthetic = prepare_dataset(
        machine_category=machine_category,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        metadata_dir=metadata_dir,
        data_dir=data_dir,
    )
    print(f"Dataset split: Train={len(data['train_mel'])}, Val={len(data['val_mel'])}, Test={len(data['test_mel'])}")
    print(f"Synthetic data status: {is_synthetic}")

    trainer = ModelTrainer(machine_category, checkpoint_dir=checkpoint_dir, seed=42)

    # 1. Train Baseline Model (Log-Mel Autoencoder)
    print(f"\n[Phase 3] Training Baseline Convolutional Autoencoder ({epochs} epochs)...")
    baseline_model, baseline_meta = trainer.train_baseline(
        train_mels=data["train_mel"],
        val_mels=data["val_mel"],
        val_labels=data["val_labels"],
        epochs=epochs,
        batch_size=batch_size,
    )
    print(f"Baseline Training Complete. Threshold: {baseline_meta['threshold']:.6f}")

    # 2. Train Proposed Model (Adaptive Fusion Autoencoder)
    print(f"\n[Phase 4] Training Proposed Adaptive Feature Fusion Autoencoder ({epochs} epochs)...")
    proposed_model, proposed_meta = trainer.train_proposed(
        train_mels=data["train_mel"],
        train_invs=data["train_inv"],
        val_mels=data["val_mel"],
        val_invs=data["val_inv"],
        val_labels=data["val_labels"],
        epochs=epochs,
        batch_size=batch_size,
    )
    print(f"Proposed Model Training Complete. Threshold: {proposed_meta['threshold']:.6f}")

    # 3. Evaluate Baseline on Held-out Test Set
    evaluator = ModelEvaluator(output_dir=experiments_dir)

    print(f"\n[Evaluation] Evaluating Baseline on {len(data['test_labels'])} test samples...")
    base_scores, base_latencies = [], []
    baseline_model.eval()
    import torch
    device = trainer.device
    for mel in data["test_mel"]:
        t0 = time.perf_counter()
        x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(device)
        score, _ = baseline_model.compute_anomaly_score(x)
        t1 = time.perf_counter()
        base_scores.append(float(score.cpu().item()))
        base_latencies.append((t1 - t0) * 1000.0)

    base_eval = evaluator.evaluate_model(
        y_true=data["test_labels"],
        y_scores=base_scores,
        threshold=baseline_meta["threshold"],
        model_name="Baseline Convolutional Autoencoder",
        inference_times_ms=base_latencies,
    )

    # 4. Evaluate Proposed on Held-out Test Set
    print(f"[Evaluation] Evaluating Proposed on {len(data['test_labels'])} test samples...")
    prop_scores, prop_latencies = [], []
    proposed_model.eval()
    for mel, inv in zip(data["test_mel"], data["test_inv"]):
        t0 = time.perf_counter()
        xm = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(device)
        xi = torch.from_numpy(inv).unsqueeze(0).unsqueeze(0).to(device)
        score, _, _ = proposed_model.compute_anomaly_score(xm, xi)
        t1 = time.perf_counter()
        prop_scores.append(float(score.cpu().item()))
        prop_latencies.append((t1 - t0) * 1000.0)

    prop_eval = evaluator.evaluate_model(
        y_true=data["test_labels"],
        y_scores=prop_scores,
        threshold=proposed_meta["threshold"],
        model_name="Proposed Adaptive Feature Fusion Autoencoder",
        inference_times_ms=prop_latencies,
    )

    # Relative AUC delta
    auc_diff = prop_eval["roc_auc"] - base_eval["roc_auc"]
    pauc_diff = prop_eval["p_auc"] - base_eval["p_auc"]

    print("\n---------------- EXPERIMENTAL RESULTS ----------------")
    print(f"Metric                 | Baseline      | Proposed Adaptive Fusion | Delta")
    print("-------------------------------------------------------------------------")
    print(f"ROC-AUC                | {base_eval['roc_auc']:<13.4f} | {prop_eval['roc_auc']:<24.4f} | {auc_diff:+.4f}")
    print(f"Partial AUC (FPR=0.1)  | {base_eval['p_auc']:<13.4f} | {prop_eval['p_auc']:<24.4f} | {pauc_diff:+.4f}")
    print(f"F1-Score               | {base_eval['f1_score']:<13.4f} | {prop_eval['f1_score']:<24.4f} | {prop_eval['f1_score'] - base_eval['f1_score']:+.4f}")
    print(f"Precision              | {base_eval['precision']:<13.4f} | {prop_eval['precision']:<24.4f} | {prop_eval['precision'] - base_eval['precision']:+.4f}")
    print(f"Recall                 | {base_eval['recall']:<13.4f} | {prop_eval['recall']:<24.4f} | {prop_eval['recall'] - base_eval['recall']:+.4f}")
    print(f"Inference Latency (ms) | {base_eval['mean_inference_latency_ms']:<13.2f} | {prop_eval['mean_inference_latency_ms']:<24.2f} | {prop_eval['mean_inference_latency_ms'] - base_eval['mean_inference_latency_ms']:+.2f} ms")
    print("-------------------------------------------------------------------------")

    experiment_id = f"exp_{machine_category}_{int(time.time())}"

    # Generate Research Figures
    figures = evaluator.generate_research_plots(
        experiment_id=experiment_id,
        machine_category=machine_category,
        baseline_metrics=base_eval,
        proposed_metrics=prop_eval,
        baseline_history=baseline_meta.get("history"),
        proposed_history=proposed_meta.get("history"),
    )
    print(f"Generated research figures in {experiments_dir}:")
    for k, v in figures.items():
        print(f"  - {k}: {v}")

    # Save Experiment Record JSON
    experiment_record = {
        "experiment_id": experiment_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "machine_category": machine_category,
        "is_synthetic": is_synthetic,
        "dataset_subset": "Synthetic Multi-Condition Calibration" if is_synthetic else f"MIMII {machine_category}",
        "fusion_mechanism": "Adaptive Latent Gating Network (Sigmoid)",
        "epochs": epochs,
        "batch_size": batch_size,
        "baseline": base_eval,
        "proposed": prop_eval,
        "relative_improvement_auc": round(auc_diff, 4),
        "relative_improvement_pauc": round(pauc_diff, 4),
        "figures": figures,
        "status": "completed",
        "notes": "Fair evaluation on identical held-out test split. Threshold selected on validation split.",
    }

    exp_json_path = experiments_dir / f"{experiment_id}.json"
    with open(exp_json_path, "w") as f:
        json.dump(experiment_record, f, indent=2)

    # Also update latest experiment pointer for UI
    latest_path = experiments_dir / "latest_experiment.json"
    with open(latest_path, "w") as f:
        json.dump(experiment_record, f, indent=2)

    print(f"\nExperiment saved to {exp_json_path}")
    return experiment_record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate AcousticIQ models")
    parser.add_argument("--machine", type=str, default="valve", choices=["valve", "pump", "fan", "slider"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--raw-dir", type=str, default=str(DEFAULT_RAW_DIR), help="Path to raw recordings")
    parser.add_argument("--interim-dir", type=str, default=str(DEFAULT_INTERIM_DIR), help="Path to interim standardized audio")
    parser.add_argument("--processed-dir", type=str, default=str(DEFAULT_PROCESSED_DIR), help="Path to processed feature tensors")
    parser.add_argument("--metadata-dir", type=str, default=str(DEFAULT_METADATA_DIR), help="Path to metadata manifests")
    parser.add_argument("--checkpoint-dir", type=str, default=str(DEFAULT_CHECKPOINT_DIR), help="Checkpoint directory")
    parser.add_argument("--experiments-dir", type=str, default=str(DEFAULT_EXPERIMENTS_DIR), help="Experiments output directory")

    args = parser.parse_args()

    run_experiment(
        machine_category=args.machine,
        epochs=args.epochs,
        batch_size=args.batch_size,
        raw_dir=Path(args.raw_dir),
        interim_dir=Path(args.interim_dir),
        processed_dir=Path(args.processed_dir),
        metadata_dir=Path(args.metadata_dir),
        checkpoint_dir=Path(args.checkpoint_dir),
        experiments_dir=Path(args.experiments_dir),
    )

