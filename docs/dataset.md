# MIMII Dataset Documentation & Management Guide

## 1. Official Dataset Record

- **Record Title:** Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection (MIMII Dataset)
- **Official URL:** https://zenodo.org/records/3384388
- **Authors:** Harsh Purohit, Ryo Tanabe, Kenji Nishida, Takashi Endo, Yuki Nikaido, Toshiki Nakamura, Yohei Kawaguchi (Hitachi, Ltd.)
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

---

## 2. Machine Categories & Failure Modes

The MIMII dataset comprises audio recordings from four industrial machine categories:

1. **Valves:**
   - **Operating Dynamics:** Fast solenoid opening/closing strokes with pneumatic flow transients.
   - **Failure Modes:** Air leakage, mechanical friction, contamination, delayed seating chatter.
2. **Pumps:**
   - **Operating Dynamics:** Centrifugal motor operation with 50 Hz power harmonics.
   - **Failure Modes:** Impeller cavitation, clogging, bearing race spall, rotor unbalance.
3. **Fans:**
   - **Operating Dynamics:** Multi-blade rotation with Blade Pass Frequency ($BPF = \text{blades} \times \frac{\text{RPM}}{60}$) and aerodynamic turbulence.
   - **Failure Modes:** Blade deformation, airflow blockage, unbalanced shaft flutter.
4. **Slide Rails (Sliders):**
   - **Operating Dynamics:** Linear reciprocating motion with acceleration, constant velocity, and deceleration segments.
   - **Failure Modes:** Rail surface scoring, insufficient lubrication, mechanical sticking.

---

## 3. Signal-to-Noise Ratios (SNR) & Acoustic Conditions

Real factory background noise recordings were blended with machine sounds at three distinct SNR levels:
- **6 dB:** Cleanest industrial setting with minimal background interference.
- **0 dB:** Moderate factory background noise where noise energy equals machine signal energy.
- **-6 dB:** Severe ambient noise environment where machine sounds are partially masked by background machinery.

---

## 4. Archive Organization

The official release is partitioned into 12 ZIP archives:
- **Valve:** `6_dB_valve.zip` (6.9 GB), `0_dB_valve.zip` (7.5 GB), `-6_dB_valve.zip` (8.0 GB)
- **Pump:** `6_dB_pump.zip` (7.7 GB), `0_dB_pump.zip` (7.9 GB), `-6_dB_pump.zip` (8.2 GB)
- **Fan:** `6_dB_fan.zip` (10.2 GB), `0_dB_fan.zip` (10.4 GB), `-6_dB_fan.zip` (10.9 GB)
- **Slide Rail:** `6_dB_slider.zip` (7.1 GB), `0_dB_slider.zip` (7.5 GB), `-6_dB_slider.zip` (8.0 GB)

---

## 5. Four-Tier Dataset Architecture

AcousticIQ enforces a strict four-tier directory structure under `ml/data/` (or external storage):

```
ml/data/
├── raw/
│   └── MIMII/                     # Original downloaded & extracted audio files (STRICTLY READ-ONLY)
│       ├── valve/
│       │   ├── id_00/
│       │   │   ├── normal/        # Raw 16kHz multi-channel or mono WAVs
│       │   │   └── anomaly/       # Raw anomaly WAVs
│       │   ├── id_02/
│       │   ├── id_04/
│       │   └── id_06/
│       ├── pump/
│       ├── fan/
│       └── slider/
├── interim/                       # Standardized intermediate audio
│   └── <category>/<id>/<cond>/    # 16kHz, mono downmixed, 10.0s windowed WAVs (*_std.wav)
├── processed/                     # Feature tensors for machine learning
│   └── <category>/<id>/<cond>/    # Compressed NumPy arrays (*.npz)
│                                  # Contains: log_mel (64, T), inv_mel (64, T), label, meta
└── metadata/                      # Split manifests and validation audit reports
    ├── validation_report.json     # Decodability, SNR, duration, and clipping audit
    ├── train_manifest.json        # Normal-only split (60% normal, 0% anomaly: ZERO LEAKAGE)
    ├── val_manifest.json          # Normal (20%) + Anomaly (50%) for threshold calibration
    ├── test_manifest.json         # Normal (20%) + Anomaly (50%) for held-out evaluation
    └── dataset_manifest.json      # Complete inventory with metadata
```

### Protection Principles:
1. **Never Modify Raw Audio:** Files placed in `raw/MIMII/` are treated as immutable ground truth and marked read-only upon extraction.
2. **Deterministic Preprocessing:** Intermediate standardized audio is written to `interim/`, ensuring raw recordings are never touched.
3. **Optimized Training:** Pre-extracted Log-Mel and Inverse-Mel feature tensors in `processed/` allow instantaneous training epochs without redundant STFT operations.
4. **Git Cleanliness:** All data directories, audio binaries, and `.npz` feature tensors are ignored in `.gitignore`. Only `.gitkeep` files are tracked in version control.

---

## 6. Configurable Dataset Paths

AcousticIQ supports both repository-local storage and external high-capacity drive storage. Paths are configurable via environment variables in `.env` or system environment:

```env
# Optional external drive path overrides
DATASET_RAW_DIR=D:/ExternalDatasets/MIMII/raw
DATASET_INTERIM_DIR=D:/ExternalDatasets/MIMII/interim
DATASET_PROCESSED_DIR=D:/ExternalDatasets/MIMII/processed
DATASET_METADATA_DIR=D:/ExternalDatasets/MIMII/metadata
```

If not specified, directories default to `ml/data/raw/MIMII/`, `ml/data/interim/`, `ml/data/processed/`, and `ml/data/metadata/`.

---

## 7. Dataset Management CLI Utilities

### 7.1 Inspect Archive Catalog
```bash
python scripts/download_mimii.py --info
```

### 7.2 Download Specific Machine Archive
```bash
# Downloads valve 6dB archive (~6.9 GB) into raw/MIMII/ with resume support
python scripts/download_mimii.py --archive valve_6dB --raw-dir ml/data/raw/MIMII
```

### 7.3 Extract Archive
```bash
# Extracts archive into structured category hierarchy and enforces read-only protection
python scripts/extract_dataset.py --archive ml/data/raw/MIMII/6_dB_valve.zip --raw-dir ml/data/raw/MIMII
```

### 7.4 Validate Dataset Integrity
```bash
# Audits decodability, duration bounds, silent files, NaNs, and digital clipping
python scripts/validate_dataset.py --raw-dir ml/data/raw/MIMII --metadata-dir ml/data/metadata
```

### 7.5 Preprocess and Generate Feature Manifests
```bash
# Standardizes to 16kHz mono, extracts dual Log-Mel & Inverse-Mel features, and creates manifests
python scripts/preprocess_dataset.py \
  --raw-dir ml/data/raw/MIMII \
  --interim-dir ml/data/interim \
  --processed-dir ml/data/processed \
  --metadata-dir ml/data/metadata \
  --sr 16000 \
  --duration 10.0 \
  --n-mels 64
```

### 7.6 Train Directly From Preprocessed Manifests
```bash
python scripts/train_and_evaluate.py \
  --machine valve \
  --epochs 20 \
  --batch-size 16 \
  --raw-dir ml/data/raw/MIMII \
  --processed-dir ml/data/processed \
  --metadata-dir ml/data/metadata
```

---

## 8. Calibrated Synthetic Fixture Fallback

For environments without downloaded multi-gigabyte archives (such as automated CI runners and rapid smoke testing), AcousticIQ provides physics-based synthetic fixtures in `ml/data/synthetic_fixture.py`.

These mathematical generators simulate realistic acoustic profiles:
- **Valve:** Solenoid transient impact + high-velocity pneumatic flow + friction bursts.
- **Pump:** Rotor rotation harmonics + 50 Hz power hum + impeller cavitation noise.
- **Fan:** Aerodynamic turbulence + Blade Pass Frequency harmonics + blade flutter.
- **Slider:** Linear acceleration slide glide + rail friction squeal + impact transient.

Whenever synthetic data is used, the system explicitly labels the records and experimental results with `is_synthetic: true` to maintain absolute scientific transparency.
