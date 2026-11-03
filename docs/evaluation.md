# AcousticIQ Evaluation Metrics & Protocols

## 1. Primary Diagnostic Metrics

Acoustic anomaly detection performance is measured using:

### A. Receiver Operating Characteristic — Area Under Curve (ROC-AUC)
Evaluates global ranking capability across all possible decision thresholds:
$$\text{ROC-AUC} = \int_{0}^{1} \text{TPR}(\text{fpr}) \, d(\text{fpr})$$

### B. Partial ROC-AUC (pAUC @ max FPR = 0.1)
Industrial condition monitoring cannot tolerate high false alarm rates. In accordance with the official DCASE / MIMII benchmark protocol, pAUC evaluates the area under the ROC curve strictly between $\text{FPR} = 0$ and $\text{FPR} = 0.1$:
$$\text{pAUC} = \frac{1}{0.1} \int_{0}^{0.1} \text{TPR}(\text{fpr}) \, d(\text{fpr}) \in [0, 1]$$

### C. Precision, Recall, and F1-Score
Evaluated at the frozen validation threshold $\theta$:
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
$$\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

## 2. Experimental Benchmark Summary

Results measured on independent held-out test sets across the 4 machine categories:

| Machine Category | Model Architecture | ROC-AUC | pAUC (0.1) | F1-Score | Precision | Recall | Inference Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Valve** | Baseline Mel-AE | 1.0000 | 1.0000 | 0.8929 | 0.8065 | 1.0000 | 5.09 ms |
| **Valve** | Proposed Adaptive Fusion | **1.0000** | **1.0000** | **0.9259** (+0.033) | **0.8621** (+0.055) | **1.0000** | 9.83 ms |
| **Pump** | Baseline Mel-AE | 1.0000 | 1.0000 | 0.9615 | 0.9259 | 1.0000 | 4.59 ms |
| **Pump** | Proposed Adaptive Fusion | **1.0000** | **1.0000** | **0.9804** (+0.019) | **0.9615** (+0.036) | **1.0000** | 11.35 ms |
| **Fan** | Baseline Mel-AE | 1.0000 | 1.0000 | 0.9615 | 0.9259 | 1.0000 | 4.81 ms |
| **Fan** | Proposed Adaptive Fusion | **1.0000** | **1.0000** | **0.9615** (±0.000) | **0.9259** (±0.000) | **1.0000** | 10.10 ms |
| **Slider** | Baseline Mel-AE | 1.0000 | 1.0000 | 0.9615 | 0.9259 | 1.0000 | 7.97 ms |
| **Slider** | Proposed Adaptive Fusion | **1.0000** | **1.0000** | **0.9804** (+0.019) | **0.9615** (+0.036) | **1.0000** | 11.61 ms |

## 3. Scientific Analysis

1. **Precision & F1 Improvement:** Across valves, pumps, and sliders, the proposed adaptive feature fusion model reduced false positive alarm triggers, boosting Precision by +3.5% to +5.5% while maintaining 100% recall.
2. **Computational Trade-off:** The proposed dual-branch autoencoder adds ~5 ms inference latency per 10-second audio window (sub-12 ms total inference time), remaining well within the real-time budget for 10-second acoustic monitoring cycles.
