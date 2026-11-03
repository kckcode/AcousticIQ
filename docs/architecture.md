# AcousticIQ System Architecture

## 1. High-Level System Architecture

AcousticIQ is structured as a decoupled, micro-service oriented predictive maintenance system designed for on-premise industrial telemetry and research environments:

```
+-----------------------------------------------------------------------------------+
|                                 CLIENT TIER                                       |
|                                                                                   |
|  React 18 + TypeScript + Vite + Tailwind CSS Dashboard                            |
|  - Real-time Waveform Scrubbing (HTML5 Web Audio API)                             |
|  - 2D Dual-Branch Spectrogram Heatmap Rendering                                  |
|  - Quantitative Score vs Threshold Gauges (Recharts)                              |
|  - Analysis History, Analytics Explorer, Model Laboratory, Experiments Catalog    |
+-----------------------------------------+-----------------------------------------+
                                          | JSON over REST (FastAPI)
                                          v
+-----------------------------------------------------------------------------------+
|                                BACKEND API TIER                                   |
|                                                                                   |
|  FastAPI Application (Uvicorn Async Worker)                                       |
|  - Routing: /api/analyze, /api/health, /api/machines, /api/results, etc.          |
|  - Upload Sanitization & Memory-Bounded Chunking (StorageService)                 |
|  - ACID Persistence: SQLite (SQLAlchemy ORM)                                      |
|  - Model Registry & Checkpoint Cache (PyTorch Device Dispatcher)                  |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                          ML INFERENCE & DSP PIPELINE                              |
|                                                                                   |
|  1. Preprocessing Engine (AudioPreprocessor)                                      |
|     - WAV Header Validation & NaN/Inf Sanitization                                |
|     - Multi-channel downmixing (Mono conversion)                                  |
|     - Polyphase Resampling (16 kHz target)                                        |
|     - Amplitude Peak & Silence Guarding                                           |
|     - Sliding Window Segmentation (10.0s window)                                  |
|                                                                                   |
|  2. Dual-Feature Extraction Pipeline                                              |
|     - Branch A: 64-band Log-Mel Spectrogram (n_fft=1024, hop=512)                 |
|     - Branch B: Inverse-Mel Pseudo-Inverse Spectral Representation                |
|                                                                                   |
|  3. Deep Convolutional Anomaly Detectors                                          |
|     - Baseline Single-Branch Mel ConvAutoencoder                                  |
|     - Proposed Dual-Branch Adaptive Feature Fusion Autoencoder                   |
|     - Dynamic Latent Gating Network (condition-adaptive weighting alpha)          |
+-----------------------------------------------------------------------------------+
```

## 2. Directory Structure

```
AcousticIQ/
├── frontend/             # React + TypeScript + Vite web dashboard
│   ├── src/
│   │   ├── components/   # Reusable UI widgets (Waveform, Spectrogram, Gauges)
│   │   ├── pages/        # 7 modular pages (Overview, Analysis, History, etc.)
│   │   ├── services/     # Typed API clients
│   │   └── types/        # TypeScript interfaces matching backend schemas
├── backend/              # FastAPI application server
│   ├── app/
│   │   ├── api/          # REST route handlers
│   │   ├── core/         # Pydantic settings & configuration
│   │   ├── database/     # SQLAlchemy SQLite session & models
│   │   ├── schemas/      # Pydantic validation schemas
│   │   └── services/     # Storage & inference managers
├── ml/                   # Machine Learning research core
│   ├── preprocessing/    # Robust audio loading & signal conditioning
│   ├── features/         # Log-Mel and Inverse-Mel feature extractors
│   ├── models/           # PyTorch Baseline & Proposed Autoencoders
│   ├── training/         # Reproducible unsupervised training loops
│   ├── evaluation/       # ROC-AUC, pAUC (0.1), F1, Confusion Matrix
│   ├── checkpoints/      # Model weights (.pt) and metadata (.json)
│   └── experiments/      # Evaluation records and Matplotlib figures
├── scripts/              # CLI tooling for download, training, and benchmarking
├── docs/                 # Architectural and methodological documentation
└── tests/                # Automated pytest suite (Preprocessing, Features, Models, API)
```
