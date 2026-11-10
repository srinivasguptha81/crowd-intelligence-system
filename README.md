# CrowdGuard AI — Real-Time Crowd Behavior Analysis, Panic Detection & Risk Forecasting (v1.0)

An end-to-end crowd safety decision-support platform featuring:
- **React + Vite Operations Dashboard** with Video Intelligence, Multi-Camera Live Feeds, and Persistent Incident Triage
- **FastAPI + WebSocket Backend** with SQLite persistence (`data/crowdguard.db`) and async background video job queue
- **YOLOv8 + Kalman-Hungarian Multi-Object Tracking** (`app/tracker.py`) with persistent Track IDs, velocity vectors, trajectory trails, and posture-collapse fall detection
- **Vector Optical Flow & Crowd Physics** (`app/crowd_physics.py`) computing 8-bin Shannon flow entropy, counter-flow collision index, spatial divergence (radial scattering vs. compression), **Helbing Crowd Pressure ($P = \rho \cdot \text{Var}(\vec{v})$)**, and **Fruin Level of Service (LOS A–F)** using perspective-calibrated floor area ($\text{people}/\text{m}^2$)
- **Dedicated Panic Detection Engine**: identifies `RADIAL_DISPERSION`, `TURBULENT_PANIC`, `COUNTER_FLOW_CONFLICT`, `CRUSH_STAGNATION` (high density + near-zero deadlock velocity), and `FALL_TRAMPLING_RISK`
- **Temporal +30s Risk Forecasting & Supervised ML Pipeline**: sliding-window temporal extrapolation (`TemporalRiskEngine`) with optional `RandomForest` + `HistGradientBoosting` ensemble training (`ml/training/extract_features.py` + `ml/training/train_risk_model.py`)
- **Multi-Camera & Simulated Scenarios**: live RTSP/HTTP/webcam streams plus built-in synthetic simulators (`demo://crowd-sim`, `demo://panic-sim`, `demo://crush-sim`) streamed over WebSockets (`/ws/cameras/{camera_id}`)

> **Scientific & Safety Integrity:** Physics-based heuristics and temporal forecasts are decision-support signals, not standalone validated stampede guarantees. Always calibrate `monitored_area_m2` and perspective weights per camera, validate on authorized venue footage, and verify alerts with trained safety operators.

## Project Structure

```text
CrowdGuardAI/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI REST & WebSocket API entrypoint
│   │   ├── config.py            # Environment-driven configuration settings
│   │   ├── database.py          # SQLite persistence (incidents, analyses, cameras)
│   │   ├── detector.py          # Thread-safe YOLOv8 person detector
│   │   ├── tracker.py           # Kalman-velocity + Hungarian multi-object tracker
│   │   ├── crowd_physics.py     # Optical flow entropy, divergence, panic & +30s forecasting
│   │   ├── camera_manager.py    # Multi-camera workers & WebSocket broadcaster
│   │   ├── simulator.py         # Synthetic crowd/panic/crush test stream generator
│   │   ├── video_processor.py   # Recorded video sampling & H.264/WebM MP4 annotation
│   │   └── security.py          # SSRF URL checks, credential masking & API auth
│   ├── tests/
│   │   └── test_crowdguard.py   # Unit & end-to-end API/WebSocket test suite
│   ├── setup.ps1                # Windows environment provisioning (venv)
│   ├── run.ps1                  # Windows backend launcher
│   ├── requirements.txt         # Python dependencies
│   ├── Dockerfile & .dockerignore
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── App.tsx              # Operations dashboard UI
│   │   ├── main.tsx             # React mount entrypoint
│   │   ├── styles.css           # Formatted dark operations-center styles
│   │   └── vite-env.d.ts
│   ├── index.html
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── Dockerfile
│   └── .env.example
├── ml/
│   ├── data/README.md           # 15-feature spatio-temporal CSV schema
│   └── training/
│       ├── extract_features.py  # Extracts behavioral/physics windows from video
│       └── train_risk_model.py  # Group-split RF + HistGB temporal ensemble trainer
├── deployment/
│   └── docker-compose.yml       # Full-stack Docker orchestration
└── docs/
    ├── FEATURES.md              # Architecture & v1.0 feature specification
    ├── SECURITY.md              # Security & privacy guidelines
    └── VALIDATION_PLAN.md       # Calibration & site-trial validation plan
```

## Requirements
- Python 3.10–3.12
- Node.js 20+
- Optional RTSP camera reachable over permitted LAN for live CCTV mode (or use built-in `demo://*` scenarios for offline evaluation)

## Windows PowerShell Quick Start

### 1. Backend terminal
```powershell
cd backend
powershell -ExecutionPolicy Bypass -File .\run.ps1
```
`run.ps1` automatically selects a working virtual environment (`venv`) or runs `setup.ps1` if needed.

Verify:
- http://127.0.0.1:8000/health
- http://127.0.0.1:8000/docs

### 2. Frontend terminal
```powershell
cd frontend
npm install
Copy-Item .env.example .env -ErrorAction SilentlyContinue
npm run dev
```
Open http://localhost:5173.

## Running Automated Tests
```powershell
cd backend
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

## ML Feature Extraction & Temporal Model Training
Extract the 15 spatio-temporal crowd-physics features from authorized venue footage:
```powershell
cd backend
.\venv\Scripts\python.exe ..\ml\training\extract_features.py --video path\to\venue_video.mp4 --out ..\ml\data\features.csv --area-m2 36.0
```
After populating `event_within_horizon` (`0` or `1`) with verified horizon labels across independent `video_id` sequences:
```powershell
.\venv\Scripts\python.exe ..\ml\training\train_risk_model.py --csv ..\ml\data\features.csv --out ..\models\crowd_risk_model.joblib --group-col video_id
```

## Key API & WebSocket Endpoints
- `GET /health` — service, detector, tracker, and ML model status
- `POST /api/v1/analyze` & `POST /api/v1/analyze/async` — recorded video analysis with calibration parameters
- `GET /api/v1/analysis/{analysis_id}` & `/video` — job progress, full telemetry JSON, and browser-playable H.264/WebM MP4
- `GET /api/v1/cameras`, `POST /api/v1/cameras/start`, `PUT /api/v1/cameras/{id}/calibration` — multi-camera & calibration lifecycle
- `WS /ws/cameras/{camera_id}` — low-latency WebSocket push stream (annotated JPEG + live physics/panic/forecast metrics)
- `GET /api/v1/incidents`, `PATCH /api/v1/incidents/{id}` — SQLite-backed incident review (`OPEN` / `ACKNOWLEDGED` / `RESOLVED`)
- Legacy compatibility: `POST /api/v1/rtsp/start`, `/stop`, `/status`, `/frame`

## Deploying to Vercel + Cloud Backend

Because the backend requires **PyTorch, Ultralytics YOLO, OpenCV, persistent WebSockets, background RTSP ingestion threads, and large video uploads** (which exceed Vercel Serverless's 250 MB bundle limit, 4.5 MB payload cap, and stateless execution limits), CrowdGuard AI uses a standard split-cloud deployment:

1. **Deploy Backend (`backend/Dockerfile`) to Render / Railway / Cloud VM**:
   - **Render**: Use the included `render.yaml` Blueprint (points to `backend/Dockerfile`, exposes `/health`, and enables `CORS_ORIGIN_REGEX=https://.*\.vercel\.app`).
   - **Railway**: Deploy `backend/` using `backend/railway.toml` and `backend/Dockerfile`.
   - Enable API authentication on public deployments by setting `AUTH_ENABLED=true` and a strong `API_AUTH_TOKEN`.

2. **Deploy Frontend (`frontend/`) to Vercel**:
   - Import the project into Vercel (works with Root Directory set to either `.` via `vercel.json` or `frontend` via `frontend/vercel.json`).
   - Optionally set `VITE_API_BASE_URL=https://your-backend-url` in Vercel Environment Variables, **or** enter your hosted backend URL and optional API token directly in the dashboard's **Configure Backend Server** bar after deployment.
