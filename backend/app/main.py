"""CrowdGuard AI FastAPI Application — Real-Time Crowd Behavior Analysis, Panic Detection & Stampede Risk Forecasting."""

from __future__ import annotations

import asyncio
import os
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .camera_manager import camera_manager
from .config import settings
from .crowd_physics import PerspectiveConfig
from .database import db
from .detector import detector_status_info
from .security import require_auth
from .video_processor import analyze_video_file

_analysis_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="video-analyzer")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    camera_manager.set_event_loop(asyncio.get_running_loop())
    yield
    camera_manager.stop_all()
    _analysis_executor.shutdown(wait=False)


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Real-time crowd behavior analysis, panic detection, perspective-calibrated density, "
        "and temporal risk forecasting for public venues."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[x.strip() for x in settings.cors_origins.split(",") if x.strip()],
    allow_origin_regex=settings.cors_origin_regex.strip() or None,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _check_risk_model_exists() -> bool:
    return settings.resolve_path(settings.model_path).exists()


# --- Schemas ---


class RTSPRequest(BaseModel):
    url: str


class CameraStartRequest(BaseModel):
    camera_id: str | None = None
    name: str = "Venue Camera"
    venue_zone: str = "Main Concourse"
    url: str
    calibration: dict[str, Any] | None = None


class CalibrationUpdateRequest(BaseModel):
    monitored_area_m2: float = 36.0
    row_weights: list[float] = [1.35, 1.0, 0.85]


class IncidentStatusUpdate(BaseModel):
    status: str


# --- Core Health & Summary ---


@app.get("/")
def root():
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health",
        "capabilities": [
            "multi_object_tracking",
            "vector_optical_flow_entropy",
            "panic_dispersion_and_crush_detection",
            "perspective_calibrated_density_los",
            "temporal_30s_risk_forecasting",
            "multi_camera_websocket_streaming",
        ],
    }


@app.get("/health")
def health():
    cams = camera_manager.list_cameras()
    running = sum(1 for c in cams if c["status"] == "running")
    det = detector_status_info()
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.app_version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "yolo_loaded": det["loaded"],
        "yolo_enabled": det["enabled"],
        "risk_model_loaded": _check_risk_model_exists(),
        "tracker": "kalman_hungarian",
        "active_cameras": running,
        "auth_enabled": settings.auth_enabled,
    }


@app.get("/api/v1/summary")
def summary():
    cams = camera_manager.list_cameras()
    running = sum(1 for c in cams if c["status"] == "running")
    all_incidents = db.list_incidents(limit=100)
    open_incidents = [x for x in all_incidents if x.get("status") == "OPEN"]
    return {
        "status": "active" if running > 0 else "ready",
        "active_cameras": running,
        "registered_cameras": len(cams),
        "zones_monitored": running * 9,
        "open_incidents": len(open_incidents),
        "latest_incidents": all_incidents[:5],
        "cameras": cams,
    }


# --- Incident Persistence & Workflow ---


@app.get("/api/v1/incidents", dependencies=[Depends(require_auth)])
def get_incidents(status: str | None = None, limit: int = 100):
    items = db.list_incidents(limit=min(500, max(1, limit)), status=status)
    return {"items": items, "count": len(items)}


@app.patch("/api/v1/incidents/{incident_id}", dependencies=[Depends(require_auth)])
def update_incident(incident_id: str, payload: IncidentStatusUpdate):
    normalized = payload.status.strip().upper()
    if normalized not in {"OPEN", "ACKNOWLEDGED", "RESOLVED"}:
        raise HTTPException(status_code=400, detail="Status must be OPEN, ACKNOWLEDGED, or RESOLVED.")
    updated = db.update_incident_status(incident_id, normalized)
    if not updated:
        raise HTTPException(status_code=404, detail="Incident not found.")
    return {"id": incident_id, "status": normalized}


@app.delete("/api/v1/incidents", dependencies=[Depends(require_auth)])
def clear_incidents():
    db.clear_incidents()
    return {"status": "cleared"}


# --- Video Upload & Analysis (Sync + Async Jobs) ---


def _parse_calibration_form(
    monitored_area_m2: float | None,
    row_weights: str | None,
) -> PerspectiveConfig:
    cfg: dict[str, Any] = {}
    if monitored_area_m2 is not None:
        cfg["monitored_area_m2"] = monitored_area_m2
    if row_weights:
        parts = [x.strip() for x in row_weights.split(",") if x.strip()]
        if len(parts) == 3:
            try:
                cfg["row_weights"] = [float(p) for p in parts]
            except ValueError:
                pass
    return PerspectiveConfig.from_dict(cfg if cfg else None)


async def _save_upload_temp(file: UploadFile) -> tuple[str, str]:
    allowed = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Use: {', '.join(sorted(allowed))}")
    content = await file.read()
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_mb} MB upload limit.")
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    return tmp_path, (file.filename or Path(tmp_path).name)


def _run_background_job(
    analysis_id: str,
    tmp_path: str,
    filename: str,
    perspective: PerspectiveConfig,
) -> None:
    try:
        analyze_video_file(
            tmp_path,
            original_filename=filename,
            analysis_id=analysis_id,
            perspective=perspective,
            progress_cb=lambda p: db.update_analysis_progress(analysis_id, p),
        )
    except Exception as exc:
        detail = getattr(exc, "detail", None) or str(exc)
        db.fail_analysis(analysis_id, str(detail))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/api/v1/analyze", dependencies=[Depends(require_auth)])
async def analyze_sync(
    file: UploadFile = File(...),
    monitored_area_m2: float | None = Form(default=None),
    row_weights: str | None = Form(default=None),
):
    tmp_path, filename = await _save_upload_temp(file)
    persp = _parse_calibration_form(monitored_area_m2, row_weights)
    aid = str(uuid.uuid4())
    db.save_analysis_job(aid, filename, status="PROCESSING", progress=0.05)
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            _analysis_executor,
            lambda: analyze_video_file(
                tmp_path,
                original_filename=filename,
                analysis_id=aid,
                perspective=persp,
            ),
        )
        return result
    except Exception as exc:
        db.fail_analysis(aid, str(exc))
        raise
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        await file.close()


@app.post("/api/v1/analyze/async", dependencies=[Depends(require_auth)])
async def analyze_async(
    file: UploadFile = File(...),
    monitored_area_m2: float | None = Form(default=None),
    row_weights: str | None = Form(default=None),
):
    tmp_path, filename = await _save_upload_temp(file)
    await file.close()
    persp = _parse_calibration_form(monitored_area_m2, row_weights)
    aid = str(uuid.uuid4())
    db.save_analysis_job(aid, filename, status="PROCESSING", progress=0.02)
    _analysis_executor.submit(_run_background_job, aid, tmp_path, filename, persp)
    return {"analysis_id": aid, "status": "PROCESSING", "progress": 0.02, "filename": filename}


@app.get("/api/v1/analyses", dependencies=[Depends(require_auth)])
def list_analyses(limit: int = 20):
    return {"items": db.list_analyses(limit=min(100, max(1, limit)))}


@app.get("/api/v1/analysis/{analysis_id}", dependencies=[Depends(require_auth)])
def get_analysis_status(analysis_id: str):
    rec = db.get_analysis(analysis_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Analysis job not found.")
    return {
        "analysis_id": rec["analysis_id"],
        "filename": rec["filename"],
        "status": rec["status"],
        "progress": rec["progress"],
        "error": rec["error"],
        "result": rec["result"],
    }


@app.get("/api/v1/analysis/{analysis_id}/video")
def get_annotated_video(analysis_id: str):
    rec = db.get_analysis(analysis_id)
    path = rec.get("video_path") if rec else None
    mime = (rec.get("media_type") if rec else None) or "video/mp4"
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Annotated video not found. Run analysis again.")
    ext = Path(path).suffix or ".mp4"
    return FileResponse(path, media_type=mime, filename=f"crowdguard-{analysis_id}{ext}")


# --- Multi-Camera Real-Time API & WebSocket Streaming ---


@app.get("/api/v1/cameras", dependencies=[Depends(require_auth)])
def list_cameras():
    return {"items": camera_manager.list_cameras()}


@app.post("/api/v1/cameras/start", dependencies=[Depends(require_auth)])
def start_camera(request: CameraStartRequest):
    if not settings.rtsp_enabled:
        raise HTTPException(status_code=403, detail="Live stream ingestion is disabled in configuration.")
    status = camera_manager.start_camera(
        camera_id=request.camera_id,
        name=request.name,
        venue_zone=request.venue_zone,
        url=request.url,
        calibration=request.calibration,
    )
    return status


@app.post("/api/v1/cameras/{camera_id}/stop", dependencies=[Depends(require_auth)])
def stop_camera(camera_id: str):
    ok = camera_manager.stop_camera(camera_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Camera not found.")
    return {"camera_id": camera_id, "status": "stopped"}


@app.delete("/api/v1/cameras/{camera_id}", dependencies=[Depends(require_auth)])
def delete_camera(camera_id: str):
    ok = camera_manager.remove_camera(camera_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Camera not found.")
    return {"camera_id": camera_id, "deleted": True}


@app.put("/api/v1/cameras/{camera_id}/calibration", dependencies=[Depends(require_auth)])
def update_camera_calibration(camera_id: str, payload: CalibrationUpdateRequest):
    updated = camera_manager.update_calibration(camera_id, payload.model_dump())
    if not updated:
        raise HTTPException(status_code=404, detail="Camera not found.")
    return updated


@app.get("/api/v1/cameras/{camera_id}/status", dependencies=[Depends(require_auth)])
def get_camera_status(camera_id: str):
    cam = camera_manager.get_camera(camera_id)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found.")
    return cam.snapshot_status()


@app.get("/api/v1/cameras/{camera_id}/frame", dependencies=[Depends(require_auth)])
def get_camera_frame(camera_id: str):
    cam = camera_manager.get_camera(camera_id)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found.")
    snap = cam.snapshot_frame()
    if not snap:
        raise HTTPException(status_code=404, detail="No live frame available yet.")
    return snap


@app.websocket("/ws/cameras/{camera_id}")
async def camera_websocket(websocket: WebSocket, camera_id: str):
    if settings.auth_enabled:
        token = websocket.query_params.get("token") or ""
        if token != settings.api_auth_token.strip():
            await websocket.close(code=4401)
            return

    await websocket.accept()
    queue = camera_manager.subscribe(camera_id)
    try:
        cam = camera_manager.get_camera(camera_id)
        if cam:
            snap = cam.snapshot_frame()
            if snap:
                await websocket.send_json({"type": "frame", **snap})
            else:
                await websocket.send_json({"type": "status", **cam.snapshot_status()})

        while True:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=2.0)
                await websocket.send_json(msg)
            except asyncio.TimeoutError:
                cam = camera_manager.get_camera(camera_id)
                status_payload = cam.snapshot_status() if cam else {"camera_id": camera_id, "status": "stopped"}
                await websocket.send_json({"type": "heartbeat", **status_payload})
    except WebSocketDisconnect:
        pass
    finally:
        camera_manager.unsubscribe(camera_id, queue)


# --- Backward-Compatible Legacy Single-Camera RTSP Endpoints ---


@app.post("/api/v1/rtsp/start", dependencies=[Depends(require_auth)])
def start_rtsp(request: RTSPRequest):
    if not settings.rtsp_enabled:
        raise HTTPException(status_code=403, detail="RTSP support is disabled in backend configuration.")
    status = camera_manager.start_camera(
        camera_id=camera_manager.DEFAULT_CAMERA_ID,
        name="Primary Venue Camera",
        venue_zone="Main Concourse",
        url=request.url,
    )
    return {"status": status["status"], "camera_id": status["camera_id"], "message": "Stream connection started."}


@app.post("/api/v1/rtsp/stop", dependencies=[Depends(require_auth)])
def stop_rtsp():
    camera_manager.stop_camera(camera_manager.DEFAULT_CAMERA_ID)
    return {"status": "stopped"}


@app.get("/api/v1/rtsp/status", dependencies=[Depends(require_auth)])
def rtsp_status():
    cam = camera_manager.get_camera(camera_manager.DEFAULT_CAMERA_ID)
    if not cam:
        return {"status": "stopped", "error": None, "metrics": {}, "started_at": None}
    snap = cam.snapshot_status()
    return {
        "status": snap["status"],
        "error": snap["error"],
        "metrics": snap["metrics"],
        "started_at": snap["started_at"],
        "fps": snap["fps"],
    }


@app.get("/api/v1/rtsp/frame", dependencies=[Depends(require_auth)])
def rtsp_frame():
    cam = camera_manager.get_camera(camera_manager.DEFAULT_CAMERA_ID)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera is not started.")
    snap = cam.snapshot_frame()
    if not snap:
        raise HTTPException(status_code=404, detail="No live frame available yet.")
    return {"image_base64": snap["image_base64"], "metrics": snap["metrics"], "fps": snap["fps"]}
