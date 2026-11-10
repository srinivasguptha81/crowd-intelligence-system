# Features in CrowdGuard AI (v1.0)

## Real-Time Crowd Behavior Analysis
- YOLOv8 person detection (`yolov8n.pt`) coupled with a SciPy **Kalman-velocity + Hungarian (`linear_sum_assignment`) Multi-Object Tracker** (`app/tracker.py`) assigning persistent Track IDs, velocity vectors, normalized body-height speeds, and trajectory trails.
- **Vector Optical Flow Dynamics** (`app/crowd_physics.py`) using Farnebäck dense optical flow retaining both magnitude and orientation angle to compute:
  - **8-bin Shannon Flow Entropy** $H(\theta)$ for chaotic vs. laminar motion.
  - **Counter-Flow Collision Index** for opposing crowd streams.
  - **Spatial Divergence ($\nabla \cdot \vec{v}$)** separating outward panic scattering from inward bottleneck compression.
- **Perspective-Calibrated Spatial Density**: $3 \times 3$ zone grid calibrated with configurable `monitored_area_m2` and top/mid/bottom perspective row weights, reporting physical density ($\text{people}/\text{m}^2$) and **Fruin Level of Service (LOS A–F)**.
- **Multi-Camera & WebSocket Streaming** (`app/camera_manager.py`): Concurrent camera workers (`rtsp://`, `rtsps://`, `http(s)://`, `webcam:N`, and offline test simulators `demo://crowd-sim`, `demo://panic-sim`, `demo://crush-sim`) streaming live frames and telemetry via `/ws/cameras/{camera_id}`.

## Panic Detection
- Dedicated behavioral panic evaluation (`detect_panic_signals`) combining optical flow dynamics and multi-object tracks:
  - **Radial Scattering / Dispersion Panic**: high outward divergence + acceleration + direction disorder.
  - **Turbulent & Counter-Flow Panic**: high flow entropy + opposing stream collisions.
  - **Crowd Crush / Deadlock Stagnation**: critical density ($\ge 1.0\text{–}2.17\text{ p/m}^2$) coupled with near-zero velocity and compression.
  - **Posture Collapse / Fall Anomaly**: sudden horizontal aspect-ratio collapse on tracked individuals inside dense crowd zones.
  - **Kinetic Energy Spike**: temporal $z$-score anomaly against the rolling kinetic energy baseline.

## Temporal Stampede-Risk Forecasting & ML Pipeline
- **Sliding-Window Temporal Risk Engine** (`TemporalRiskEngine`): tracks density growth rate, **Helbing Crowd Pressure ($P = \rho \cdot \text{Var}(\vec{v})$)**, and panic momentum to project risk over a **+30s forecast horizon**, classify trend (`STABLE`, `RISING`, `RAPID_ESCALATION`, `DEESCALATING`), and estimate **Time-to-Critical-Threshold (ETCT)**.
- **Feature Extraction & Ensemble Training** (`ml/training/extract_features.py`, `ml/training/train_risk_model.py`): extracts 15 spatio-temporal physics features and trains a `RandomForestClassifier` + `HistGradientBoostingClassifier` soft-voting ensemble with `GroupShuffleSplit` by `video_id`.

## Operational Persistence & Security
- **SQLite Persistence** (`app/database.py`): retains video analyses, browser-playable H.264/WebM annotated outputs, camera profiles, and incident lifecycle states (`OPEN` → `ACKNOWLEDGED` → `RESOLVED`).
- **Security Controls** (`app/security.py`): URL credential redaction, SSRF network checks, and optional bearer/API-key authentication (`AUTH_ENABLED=true`).

## Scientific Integrity Notice
All physics heuristics and temporal forecasts are decision-support signals. Do not deploy without venue-specific perspective calibration, local validation on authorized venue footage, and human-in-the-loop verification.
