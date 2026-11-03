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

from ml.preprocessing.audio import AudioPreprocessor, PreprocessingConfig
from ml.features.extractor import DualFeatureExtractor, FeatureConfig
from ml.models.autoencoder import BaselineConvolutionalAutoencoder, ProposedAdaptiveFusionAutoencoder
from ml.training.trainer import ModelTrainer
from ml.evaluation.evaluator import ModelEvaluator
from ml.data.synthetic_fixture import generate_synthetic_machine_sound


def prepare_dataset(
    machine_category: str,
    data_dir: Path,
    use_synthetic_if_missing: bool = True,
    n_train_normal: int = 40,
    n_val_normal: int = 15,
    n_val_anomaly: int = 15,
    n_test_normal: int = 25,
    n_test_anomaly: int = 25,
) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, List[int]], bool]:
    """
    Loads real audio files if available in data_dir, otherwise generates controlled
    calibration fixtures.
    """
    preprocessor = AudioPreprocessor()
    extractor = DualFeatureExtractor()

    real_normal_dir = data_dir / machine_category / "normal"
    real_anomaly_dir = data_dir / machine_category / "anomaly"

    is_synthetic = True
    train_mels, train_invs = [], []
    val_mels, val_invs, val_labels = [], [], []
    test_mels, test_invs, test_labels = [], [], []

    # Check for genuine MIMII files
    if real_normal_dir.exists() and any(real_normal_dir.glob("*.wav")):
        print(f"[Dataset] Loading genuine audio recordings from {data_dir / machine_category}...")
        is_synthetic = False
        normal_files = sorted(list(real_normal_dir.glob("*.wav")))
        anomaly_files = sorted(list(real_anomaly_dir.glob("*.wav"))) if real_anomaly_dir.exists() else []

        # Split 60% train (normal only), 20% val, 20% test
        n_norm = len(normal_files)
        n_tr = int(n_norm * 0.6)
        n_vl = int(n_norm * 0.2)

        for p in normal_files[:n_tr]:
            audio, _ = preprocessor.load_and_preprocess(p)
            mel, inv, _ = extractor.extract(audio)
            train_mels.append(mel)
            train_invs.append(inv)

        for p in normal_files[n_tr:n_tr + n_vl]:
            audio, _ = preprocessor.load_and_preprocess(p)
            mel, inv, _ = extractor.extract(audio)
            val_mels.append(mel)
            val_invs.append(inv)
            val_labels.append(0)

        for p in normal_files[n_tr + n_vl:]:
            audio, _ = preprocessor.load_and_preprocess(p)
            mel, inv, _ = extractor.extract(audio)
            test_mels.append(mel)
            test_invs.append(inv)
            test_labels.append(0)

        # Split anomalies 50/50 between val and test
        n_anom = len(anomaly_files)
        n_anom_val = n_anom // 2
        for p in anomaly_files[:n_anom_val]:
            audio, _ = preprocessor.load_and_preprocess(p)
            mel, inv, _ = extractor.extract(audio)
            val_mels.append(mel)
            val_invs.append(inv)
            val_labels.append(1)

        for p in anomaly_files[n_anom_val:]:
            audio, _ = preprocessor.load_and_preprocess(p)
            mel, inv, _ = extractor.extract(audio)
            test_mels.append(mel)
            test_invs.append(inv)
            test_labels.append(1)

    elif use_synthetic_if_missing:
        print(f"[Dataset] Real MIMII audio not found in {data_dir / machine_category}.")
        print(f"[Dataset] Generating calibrated multi-condition synthetic dataset ({machine_category})...")
        print("  NOTE: Explicitly marked as synthetic fixtures for testing & verification.")
        
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
    return data, is_synthetic


def run_experiment(
    machine_category: str = "valve",
    epochs: int = 20,
    batch_size: int = 16,
    data_dir: Path = BASE_DIR / "ml" / "data",
    checkpoint_dir: Path = BASE_DIR / "ml" / "checkpoints",
    experiments_dir: Path = BASE_DIR / "ml" / "experiments",
):
    print(f"\n========================================================")
    print(f" AcousticIQ Training & Evaluation Pipeline: {machine_category.upper()}")
    print(f"========================================================")

    data, is_synthetic = prepare_dataset(machine_category, data_dir)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine", type=str, default="valve", choices=["valve", "pump", "fan", "slider"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    run_experiment(
        machine_category=args.machine,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
