# AcousticIQ Setup and Execution Guide

## Prerequisites

- **Operating System:** Windows 10/11, Linux, or macOS
- **Python:** 3.10+ (tested on Python 3.13)
- **Node.js:** 18+ (tested on Node.js LTS v26.13.0)
- **Audio Codec Libraries:** `libsndfile` / `soxr`

## 1. Backend Setup & Startup

### Clone and Navigate
```bash
git clone https://github.com/kckcode/AcousticIQ.git
cd AcousticIQ
```

### Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### Run Automated Test Suite
```bash
pytest tests/ -v
```

### Start the FastAPI Backend Server
```bash
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
Health Endpoint: `http://localhost:8000/api/health`

## 2. Frontend Setup & Startup

### Navigate to Frontend
```bash
cd frontend
```

### Install Node Dependencies
```bash
npm install
```

### Start Vite Development Server
```bash
npm run dev
```
Dashboard UI: `http://localhost:5173`

### Production Build
```bash
npm run build
```

## 3. Dataset & Model Training Workflows

### Inspect Available MIMII Archives
```bash
python scripts/download_mimii.py --info
```

### Download Specific MIMII Subset (e.g. Valve 6 dB)
```bash
python scripts/download_mimii.py --archive valve_6dB --target-dir ml/data/raw
```

### Run End-to-End Training and Evaluation
```bash
# Train and evaluate specific category:
python scripts/train_and_evaluate.py --machine valve --epochs 15

# Or train all 4 machine classes:
python scripts/train_all_categories.py
```
This automatically updates checkpoints in `ml/checkpoints/` and publishes experiment results and plots to `ml/experiments/`.

## 4. Docker Deployment

```bash
docker-compose up --build
```
- Dashboard UI: `http://localhost:5173`
- Backend API: `http://localhost:8000`
