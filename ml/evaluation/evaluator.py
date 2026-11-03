"""
Evaluation and Experiment Tracking Pipeline for AcousticIQ.
Implements:
- Standard ROC-AUC and Partial ROC-AUC (pAUC at max FPR = 0.1, DCASE/MIMII protocol).
- Precision, Recall, F1-Score, Confusion Matrix.
- Reproducible experiment tracking and publication-ready matplotlib figure generation.
"""

import time
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_fscore_support


def compute_pauc(y_true: np.ndarray, y_score: np.ndarray, max_fpr: float = 0.1) -> float:
    """
    Computes partial Area Under the ROC Curve (pAUC) for FPR in [0, max_fpr].
    Standard DCASE evaluation metric for industrial anomaly detection.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    # Clip to max_fpr
    stop_idx = np.searchsorted(fpr, max_fpr, side="right")
    
    # Sub-curve points
    sub_fpr = list(fpr[:stop_idx]) + [max_fpr]
    # Linear interpolation of TPR at max_fpr
    if stop_idx < len(fpr):
        x0, x1 = fpr[stop_idx - 1], fpr[stop_idx]
        y0, y1 = tpr[stop_idx - 1], tpr[stop_idx]
        tpr_at_max = y0 + (y1 - y0) * (max_fpr - x0) / max(1e-9, (x1 - x0))
    else:
        tpr_at_max = tpr[-1]
    sub_tpr = list(tpr[:stop_idx]) + [tpr_at_max]

    if hasattr(np, "trapezoid"):
        raw_pauc = np.trapezoid(sub_tpr, sub_fpr)
    else:
        raw_pauc = np.trapz(sub_tpr, sub_fpr)
    # Normalize to [0, 1] relative to the maximum possible area max_fpr
    normalized_pauc = raw_pauc / max_fpr
    return float(np.clip(normalized_pauc, 0.0, 1.0))


class ModelEvaluator:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_model(
        self,
        y_true: List[int], # 0 = normal, 1 = anomaly
        y_scores: List[float],
        threshold: float,
        model_name: str,
        inference_times_ms: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates predictions against ground truth labels.
        """
        y_true_arr = np.array(y_true)
        y_scores_arr = np.array(y_scores)
        y_pred_arr = (y_scores_arr >= threshold).astype(int)

        # ROC & AUC
        fpr, tpr, _ = roc_curve(y_true_arr, y_scores_arr)
        roc_auc_val = float(auc(fpr, tpr))
        pauc_val = compute_pauc(y_true_arr, y_scores_arr, max_fpr=0.1)

        # Classification metrics at configured threshold
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_arr, y_pred_arr, average="binary", zero_division=0
        )
        cm = confusion_matrix(y_true_arr, y_pred_arr).tolist()
        tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr).ravel()
        fpr_at_threshold = float(fp / max(1, (fp + tn)))

        latency_mean = float(np.mean(inference_times_ms)) if inference_times_ms else 0.0

        metrics = {
            "model_name": model_name,
            "roc_auc": round(roc_auc_val, 4),
            "p_auc": round(pauc_val, 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
            "false_positive_rate": round(fpr_at_threshold, 4),
            "decision_threshold": round(threshold, 6),
            "confusion_matrix": cm,
            "total_test_samples": len(y_true),
            "normal_test_samples": int(np.sum(y_true_arr == 0)),
            "anomaly_test_samples": int(np.sum(y_true_arr == 1)),
            "mean_inference_latency_ms": round(latency_mean, 2),
            "fpr_curve": [round(x, 4) for x in fpr.tolist()],
            "tpr_curve": [round(x, 4) for x in tpr.tolist()],
        }
        return metrics

    def generate_research_plots(
        self,
        experiment_id: str,
        machine_category: str,
        baseline_metrics: Dict[str, Any],
        proposed_metrics: Dict[str, Any],
        baseline_history: Optional[Dict[str, Any]] = None,
        proposed_history: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """
        Generates publication-quality research figures:
        1. ROC curves comparing Baseline vs Proposed.
        2. Metrics bar chart comparison.
        3. Training loss curves.
        4. Confusion matrices.
        """
        plots = {}

        # 1. ROC Curves
        fig, ax = plt.subplots(figsize=(6.5, 5), dpi=150)
        ax.plot(
            baseline_metrics["fpr_curve"],
            baseline_metrics["tpr_curve"],
            color="#2563eb",
            lw=2,
            label=f"Baseline Mel-AE (AUC = {baseline_metrics['roc_auc']:.3f}, pAUC = {baseline_metrics['p_auc']:.3f})",
        )
        ax.plot(
            proposed_metrics["fpr_curve"],
            proposed_metrics["tpr_curve"],
            color="#059669",
            lw=2.2,
            linestyle="--",
            label=f"Proposed Adaptive Fusion (AUC = {proposed_metrics['roc_auc']:.3f}, pAUC = {proposed_metrics['p_auc']:.3f})",
        )
        ax.plot([0, 1], [0, 1], color="#94a3b8", linestyle=":", lw=1.2, label="Random Chance (AUC = 0.500)")
        ax.axvline(0.1, color="#dc2626", linestyle=":", lw=1.2, alpha=0.7, label="pAUC boundary (FPR = 0.1)")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate (FPR)", fontsize=11)
        ax.set_ylabel("True Positive Rate (TPR / Recall)", fontsize=11)
        ax.set_title(f"ROC Performance Comparison — {machine_category.upper()} Acoustic Monitoring", fontsize=12, fontweight="bold")
        ax.legend(loc="lower right", fontsize=8.5, frameon=True)
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        roc_path = self.output_dir / f"{experiment_id}_roc_curve.png"
        fig.savefig(roc_path)
        plt.close(fig)
        plots["roc_curve"] = str(roc_path)

        # 2. Metric Comparison Bar Chart
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
        metric_names = ["ROC-AUC", "pAUC (0.1)", "F1-Score", "Precision", "Recall"]
        base_vals = [
            baseline_metrics["roc_auc"],
            baseline_metrics["p_auc"],
            baseline_metrics["f1_score"],
            baseline_metrics["precision"],
            baseline_metrics["recall"],
        ]
        prop_vals = [
            proposed_metrics["roc_auc"],
            proposed_metrics["p_auc"],
            proposed_metrics["f1_score"],
            proposed_metrics["precision"],
            proposed_metrics["recall"],
        ]

        x = np.arange(len(metric_names))
        width = 0.35
        rects1 = ax.bar(x - width/2, base_vals, width, label="Baseline Mel-AE", color="#3b82f6")
        rects2 = ax.bar(x + width/2, prop_vals, width, label="Proposed Adaptive Fusion", color="#10b981")

        ax.set_ylabel("Score", fontsize=11)
        ax.set_title(f"Diagnostic Metrics Comparison ({machine_category.title()})", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, fontsize=10)
        ax.set_ylim([0.0, 1.15])
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        for rect in rects1:
            h = rect.get_height()
            ax.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8)
        for rect in rects2:
            h = rect.get_height()
            ax.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

        plt.tight_layout()
        bar_path = self.output_dir / f"{experiment_id}_metrics_comparison.png"
        fig.savefig(bar_path)
        plt.close(fig)
        plots["metrics_bar"] = str(bar_path)

        # 3. Training Loss Curves
        if baseline_history and proposed_history:
            fig, ax = plt.subplots(figsize=(6.5, 4), dpi=150)
            epochs = range(1, len(baseline_history.get("train_loss", [])) + 1)
            ax.plot(epochs, baseline_history.get("train_loss", []), label="Baseline MSE Loss", color="#2563eb", lw=1.8)
            ax.plot(epochs, proposed_history.get("train_loss", []), label="Proposed Composite Loss", color="#059669", lw=1.8, linestyle="--")
            ax.set_xlabel("Epoch", fontsize=10)
            ax.set_ylabel("Reconstruction Error (Loss)", fontsize=10)
            ax.set_title(f"Unsupervised Training Convergence ({machine_category.title()})", fontsize=11, fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.4)
            plt.tight_layout()
            loss_path = self.output_dir / f"{experiment_id}_loss_curves.png"
            fig.savefig(loss_path)
            plt.close(fig)
            plots["loss_curves"] = str(loss_path)

        return plots
