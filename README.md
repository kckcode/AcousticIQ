# AcousticIQ: Acoustic Deep Learning Based Predictive Maintenance System for Early Detection of Industrial Machine Failures

[![CI](https://github.com/kckcode/AcousticIQ/actions/workflows/ci.yml/badge.svg)](https://github.com/kckcode/AcousticIQ/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![React: 18](https://img.shields.io/badge/React-18.3-61dafb.svg)](https://react.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)

---

## 1. Project Title
**AcousticIQ: Development of Acoustic Deep Learning Based Predictive Maintenance System for Early Detection of Industrial Machine Failures.**

## 2. Project Overview
AcousticIQ is an industrial AI research and condition-monitoring platform engineered to detect subtle acoustic deviations in critical machinery before mechanical degradation leads to catastrophic failures. By ingesting 10-second industrial machine audio recordings, AcousticIQ extracts complementary time-frequency representations, computes reconstruction-based anomaly scores through deep convolutional autoencoders, and presents actionable telemetry through a responsive React dashboard.

## 3. Problem Statement
Rotating and reciprocating industrial equipment—including solenoid valves, centrifugal pumps, ventilation fans, and linear slide rails—exhibit micro-acoustic anomalies prior to functional breakdown. However:
- Industrial acoustic failure data is extremely scarce in real plants, preventing standard supervised classification.
- Machine operating environments feature severe background noise (pumps, compressors, shop-floor transit) with fluctuating Signal-to-Noise Ratios (SNR: -6 dB, 0 dB, +6 dB).
- Standard single-branch feature representations (such as standard Log-Mel spectrograms) integrate upper frequency bands into wide filter bins, losing fine harmonic friction and micro-impact indicators.

## 4. Research Motivation
Predictive maintenance using acoustic telemetry provides non-invasive, contactless continuous monitoring without halting machine operation. Providing an adaptive dual-branch feature pipeline that combines standard Mel filterbanks with an Inverse-Mel pseudo-inverse reconstruction representation allows the model to capture both low-frequency envelope power and high-frequency harmonic structures.

## 5. Research Objectives
1. Implement a clean, reproducible signal preprocessing pipeline capable of handling corrupted, silent, or multi-channel audio without crashing.
2. Develop an unsupervised baseline deep convolutional autoencoder trained solely on normal operating machine audio.
3. Formulate and implement a novel **Dual-Branch Adaptive Feature Fusion Autoencoder** featuring a learned latent gating network that dynamically weights complementary spectral representations based on latent acoustic noise conditions.
4. Establish a zero-leakage evaluation protocol measuring ROC-AUC, Partial ROC-AUC (pAUC at max FPR = 0.1), Precision, Recall, and Latency on held-out test splits.
5. Deliver a production-grade full-stack dashboard (React, TypeScript, FastAPI, SQLite) capable of running locally without reliance on paid external APIs.

## 6. Research Questions
- **RQ1:** Does combining standard Log-Mel spectrograms with an Inverse-Mel pseudo-inverse linear spectral representation improve anomaly discriminative sensitivity under varying ambient factory noise conditions?
- **RQ2:** Can an adaptive gating network dynamically reweight feature streams based on latent acoustic conditions to reduce false positive alarm triggers without sacrificing anomaly recall?
- **RQ3:** Can real-time inference latency be maintained under 15 ms on commodity CPU hardware for a 10-second machine recording?

## 7. Existing-System Limitations
- **Supervised Classifiers:** Depend on abundant labelled failure recordings, which are unavailable or cost-prohibitive to acquire in real industrial plants.
- **Vibration Accelerometers:** Require intrusive mechanical contact, surface mounting, and wiring susceptible to extreme thermal environments.
- **Fixed-Weight Heuristics:** In dual-branch models, static averaging weights fail when high-frequency background noise overwhelms one branch.

## 8. Proposed Methodology
AcousticIQ models the normal acoustic distribution using unsupervised reconstruction. Normal sounds produce low reconstruction errors, whereas unfamiliar anomalous transients (cavitation, leakage hiss, bearing chirp, friction scrape) produce high reconstruction errors.

The proposed architecture introduces:
1. **Log-Mel Spectrogram Branch ($L_{mel}$):** 64 Mel bands, $N_{fft} = 1024$, hop = 512.
2. **Inverse-Mel Pseudo-Inverse Branch ($L_{inv}$):** Reconstructs linear spectral power via Moore-Penrose pseudo-inverse $M^\dagger = (M^T M + \lambda I)^{-1} M^T$ projected into 64 complementary frequency bins.
3. **Adaptive Latent Gating Module:** Computes a scalar $\alpha \in [0, 1]$ from latent pooled descriptors:
   $$\alpha = \sigma(W_2 \cdot \text{LeakyReLU}(W_1 [\bar{z}_{mel} \parallel \bar{z}_{inv}] + b_1) + b_2)$$
4. **Adaptive Anomaly Score:**
   $$S(x) = \alpha \cdot \text{MSE}(L_{mel}, \hat{L}_{mel}) + (1 - \alpha) \cdot \text{MSE}(L_{inv}, \hat{L}_{inv})$$

## 9. System Architecture
```
[ Industrial Audio (WAV) ]
          │
          ▼
[ AudioPreprocessor (Resample 16kHz, Mono, Peak Normalize, Guard NaN/Silence) ]
          │
    ┌─────┴─────────────────────────────────┐
    ▼                                       ▼
[ Log-Mel Extractor (64 bins) ]     [ Inverse-Mel Extractor (pinv(M)) ]
    │                                       │
    ▼                                       ▼
[ Conv2D Mel Encoder ]               [ Conv2D Inv-Mel Encoder ]
    │                                       │
    └───────────────────┬───────────────────┘
                        ▼
          [ Adaptive Gating Network ] ──> α ∈ [0, 1]
                        │
                        ▼
          [ Fused Latent Bottleneck: z = α·z_mel + (1-α)·z_inv ]
                        │
    ┌───────────────────┴───────────────────┐
    ▼                                       ▼
[ ConvTranspose2D Mel Decoder ]      [ ConvTranspose2D Inv-Mel Decoder ]
    │                                       │
    ▼                                       ▼
[ Reconstructed Mel: x̂_mel ]         [ Reconstructed Inv: x̂_inv ]
    │                                       │
    └───────────────────┬───────────────────┘
                        ▼
       [ Composite Anomaly Score S(x) ]
                        │
                        ▼
   [ Threshold Decision θ (95% Val Normal) ]
                        │
                        ▼
       [ Normal Pattern vs Potential Anomaly ]
```

## 10. Feature Extraction Pipeline
- **Sample Rate:** 16,000 Hz.
- **Window Length:** 10.0 seconds (160,000 samples).
- **STFT Hann Window:** $N_{fft} = 1024$, Hop Length = 512 samples (~32 ms window, 16 ms step).
- **Log Compression:** $L(b, m) = 10 \log_{10}(\max(S(b, m), 10^{-10}))$.
- **Standardization:** Per-sample zero-mean and unit-variance normalization.

## 11. Baseline Model
- **Architecture:** `BaselineConvolutionalAutoencoder`
- **Encoder:** 3 successive Conv2D blocks (channels 1 $\rightarrow$ 32 $\rightarrow$ 64 $\rightarrow$ 128, kernel 3x3, stride 2) with BatchNorm and LeakyReLU(0.2).
- **Decoder:** 3 Deconv2D blocks reconstructing original spatial dimension $(1, 64, T)$.
- **Trained on:** Normal machine recordings only, using Adam optimizer ($lr = 10^{-3}$) and MSE loss.

## 12. Proposed Adaptive-Fusion Method
- **Architecture:** `ProposedAdaptiveFusionAutoencoder`
- **Dual Encoders:** Independent weights for Mel and Inverse-Mel representations.
- **Adaptive Gating Network:** Takes Global Average Pooled vectors $(\bar{z}_{mel}, \bar{z}_{inv}) \in \mathbb{R}^{256}$ and computes dynamic gating weight $\alpha \in [0, 1]$ via a 2-layer MLP with Sigmoid activation.
- **Weighted Loss:** $\mathcal{L}_{total} = \alpha \cdot \text{MSE}_{mel} + (1 - \alpha) \cdot \text{MSE}_{inv}$.

## 13. Dataset Details and Official Source
The primary benchmark dataset is the **MIMII Dataset** (*Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection*).
- **Official Zenodo Record:** [https://zenodo.org/records/3384388](https://zenodo.org/records/3384388)
- **Machine Categories:** Valves, Pumps, Fans, Slide Rails (Sliders).
- **Audio Specifications:** 16-bit WAV, 16 kHz, 8-channel microphone array downmixed to mono, 10-second segments.

## 14. Dataset License and Attribution Requirements
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Attribution:** Harsh Purohit, Ryo Tanabe, Kenji Nishida, Takashi Endo, Yuki Nikaido, Toshiki Nakamura, and Yohei Kawaguchi. *"MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection."* In Proceedings of the Detection and Classification of Acoustic Scenes and Events 2019 Workshop (DCASE2019), 2019.

## 15. Repository Structure
```
AcousticIQ/
├── frontend/             # React 18, TypeScript, Tailwind CSS, Recharts dashboard
│   ├── src/
│   │   ├── components/   # WaveformVisualizer, SpectrogramViewer, AnomalyGauge, StatCard
│   │   ├── pages/        # Overview, Analysis, Monitoring, Analytics, ModelLab, Experiments, Settings
│   │   ├── services/     # Typed API fetch client
│   │   ├── types/        # TypeScript data schemas
│   │   └── App.tsx       # Main layout & navigation
│   ├── package.json
│   └── vite.config.ts
├── backend/              # FastAPI application server
│   ├── app/
│   │   ├── api/          # REST endpoints (/api/analyze, /health, /machines, etc.)
│   │   ├── core/         # Pydantic configuration settings
│   │   ├── database/     # SQLite ORM session and AnalysisRecord model
│   │   ├── schemas/      # Pydantic request/response schemas
│   │   └── services/     # Storage & Inference services
│   ├── Dockerfile
│   └── requirements.txt
├── ml/                   # Machine learning core
│   ├── preprocessing/    # Robust audio loader & validator
│   ├── features/         # Log-Mel and Inverse-Mel feature extractors
│   ├── models/           # PyTorch Baseline & Proposed Autoencoders
│   ├── training/         # Training loops & checkpoint manager
│   ├── evaluation/       # ROC-AUC, pAUC, F1, Latency, research plot generator
│   ├── checkpoints/      # Trained model weights (.pt) and metadata (.json)
│   └── experiments/      # Saved experiment logs and publication figures
├── scripts/
│   ├── download_mimii.py        # Resumable Zenodo dataset downloader
│   ├── train_and_evaluate.py    # Unified end-to-end training & evaluation pipeline
│   └── train_all_categories.py  # Automation script for all 4 machine classes
├── docs/                 # Architecture, methodology, dataset, API, and setup docs
├── tests/                # Automated pytest suite (19 passing tests)
├── .github/workflows/    # CI pipeline for automated testing and build
├── docker-compose.yml    # Container orchestration
├── .env.example          # Environment configuration template
├── LICENSE               # MIT License
└── README.md             # Master project documentation
```

## 16. Prerequisites
- **OS:** Windows 10/11, Linux, or macOS.
- **Python:** 3.10+ (tested on Python 3.13.9).
- **Node.js:** 18+ (tested on Node.js LTS v26.13.0 with npm 11.20.0).
- **GPU (Optional):** NVIDIA GPU supported via CUDA; CPU execution fully supported.

## 17. Installation Instructions for Windows

### A. Clone Repository
```powershell
git clone https://github.com/kckcode/AcousticIQ.git
cd AcousticIQ
```

### B. Python Environment Setup
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r backend/requirements.txt
```

### C. Node.js Frontend Dependencies
```powershell
cd frontend
npm install
cd ..
```

## 18. Backend Startup Instructions
Start the FastAPI backend with Uvicorn:
```powershell
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
- API Base: `http://localhost:8000`
- Health Endpoint: `http://localhost:8000/api/health`
- Swagger Interactive Documentation: `http://localhost:8000/docs`

## 19. Frontend Startup Instructions
In a separate terminal:
```powershell
cd frontend
npm run dev
```
Open `http://localhost:5173` in any modern web browser.

## 20. Dataset Preparation Instructions
Inspect available MIMII archives on Zenodo:
```powershell
python scripts/download_mimii.py --info
```
Download a specific category and SNR level (e.g. Valve at 6 dB SNR):
```powershell
python scripts/download_mimii.py --archive valve_6dB --target-dir ml/data/raw
```
*Note: A complete suite of pre-calibrated synthetic fixtures is generated in `ml/data/synthetic_fixtures/` for instant local testing and CI verification without requiring external multi-gigabyte downloads.*

## 21. Model Training Instructions
Train both Baseline and Proposed models for a specific machine class:
```powershell
python scripts/train_and_evaluate.py --machine valve --epochs 15
```
Or train all 4 machine classes sequentially:
```powershell
python scripts/train_all_categories.py
```
Checkpoints and metadata sidecars are automatically written to `ml/checkpoints/`.

## 22. Evaluation Instructions
The evaluation pipeline executes automatically at the end of training. To run standalone evaluation or reproduce figures:
```powershell
python scripts/train_and_evaluate.py --machine valve --epochs 1
```
Results and Matplotlib PNG figures (`roc_curve.png`, `metrics_comparison.png`, `loss_curves.png`) are saved to `ml/experiments/`.

## 23. API Documentation
Key REST endpoints:
- `GET /api/health`: Service status, active device, and loaded model checkpoints.
- `GET /api/machines`: Supported industrial machine categories.
- `POST /api/analyze`: Ingests multipart audio file, runs preprocessing and inference, persists result, returns anomaly score, threshold, confidence, waveform, and spectrograms.
- `GET /api/results`: Paginated analysis history with filtering.
- `GET /api/metrics`: Latest experiment evaluation metrics.
- `GET /api/models`: Model checkpoint registry catalog.
- `GET /api/experiments`: Historical experiment records.
- `GET /api/audio/samples`: List of preloaded test audio fixtures.
- `GET /api/audio/sample/{filename}`: Audio stream for browser playback.

## 24. Test Commands
Execute the full automated test suite (backend, signal processing, features, neural models):
```powershell
pytest tests/ -v
```
Run frontend TypeScript and production build verification:
```powershell
cd frontend
npm run build
```

## 25. Docker Instructions
Launch full application stack with Docker Compose:
```bash
docker-compose up --build
```
- Dashboard: `http://localhost:5173`
- Backend API: `http://localhost:8000`

## 26. Application Interface Walkthrough & Figures
1. **Overview Dashboard:** Fleet status, total processed analyses, normal vs anomaly counts, model readiness, and recent telemetry logs.
2. **Acoustic Analysis:** Drag-and-drop WAV upload or preloaded sample picker, audio player with scrub bar, anomaly score gauge vs threshold, and dual-branch spectrogram heatmaps.
3. **Monitoring & History:** Paginated historical analysis table with CSV export and detail inspection modal.
4. **Analytics:** Recharts visual explorer showing machine distribution, normal vs anomaly ratio, and temporal score trends.
5. **Model Laboratory:** Mathematical formulation, architecture schematics, checkpoint catalog, and parameter counts.
6. **Experiment Benchmarks:** Scientific comparison tables, ROC curves, and confusion matrix diagnostics.
7. **Settings & Telemetry:** Hardware compute status, audio feature hyperparameters, and citations.

## 27. Current Implementation Status
- **Preprocessing & DSP Pipeline:** Verified complete.
- **Neural Architectures (Baseline & Proposed):** Verified complete.
- **Unsupervised Training & Validation Protocol:** Verified complete.
- **Checkpoints:** All 4 categories (`valve`, `pump`, `fan`, `slider`) trained and saved in `ml/checkpoints/`.
- **FastAPI Backend & SQLite Persistence:** Verified complete (7 API integration tests passing).
- **React 18 + TypeScript Dashboard:** Verified complete (`tsc && vite build` passing).
- **Automated Tests:** 19 out of 19 tests passing.

## 28. Known Limitations
- **Diagnostic Scope:** The system performs **acoustic anomaly detection** (quantifying deviation from normal operational baselines). It does not perform root-cause fault diagnosis or predict Remaining Useful Life (RUL).
- **Synthetic Calibration Data:** When genuine multi-gigabyte MIMII archives are offline, synthetic fixtures are utilized for smoke testing; these are clearly tagged with `is_synthetic: true`.
- **Latency Trade-off:** The proposed dual-branch model adds ~5 ms inference latency per 10-second audio window compared to the single-branch baseline (sub-12 ms total inference time).

## 29. Future Work
- Integration with edge microcontrollers (e.g. Raspberry Pi / Jetson Nano) via ONNX runtime export.
- Self-supervised contrastive learning (SimCLR / BYOL for audio) to further enhance feature representations.
- Real-time continuous ring-buffer streaming analysis for live microphone arrays.

## 30. Citation and Acknowledgment
If using this work or codebase, please cite:
```bibtex
@article{acousticiq2026,
  title={AcousticIQ: Development of Acoustic Deep Learning Based Predictive Maintenance System for Early Detection of Industrial Machine Failures},
  author={kckcode},
  year={2026},
  url={https://github.com/kckcode/AcousticIQ}
}
```
MIMII Dataset citation:
```bibtex
@inproceedings{purohit2019mimii,
  title={MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection},
  author={Purohit, Harsh and Tanabe, Ryo and Nishida, Kenji and Endo, Takashi and Nikaido, Yuki and Nakamura, Toshiki and Kawaguchi, Yohei},
  booktitle={Proceedings of the Detection and Classification of Acoustic Scenes and Events 2019 Workshop (DCASE2019)},
  pages={209--213},
  year={2019}
}
```
