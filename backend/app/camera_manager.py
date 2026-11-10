"""Multi-camera real-time ingestion, tracking, panic detection, and WebSocket broadcast manager."""

from __future__ import annotations

import asyncio
import base64
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np

from .config import settings
from .crowd_physics import (
    PerspectiveConfig,
    TemporalRiskEngine,
    compute_flow_dynamics,
    compute_zone_metrics,
    draw_annotations,
)
from .database import db
from .detector import detect_people
from .security import mask_parsed_url, validate_stream_source
from .simulator import SyntheticCrowdStream
from .tracker import CrowdTracker


class CameraWorker:
    """Thread-safe worker managing a single real-time camera or simulated stream."""

    def __init__(
        self,
        camera_id: str,
        name: str,
        venue_zone: str,
        raw_url: str,
        calibration: PerspectiveConfig,
        on_frame_cb: Any = None,
    ) -> None:
        self.camera_id = camera_id
        self.name = name
        self.venue_zone = venue_zone
        self.raw_url = raw_url
        self.masked_url = mask_parsed_url(raw_url)
        self.calibration = calibration
        self.on_frame_cb = on_frame_cb

        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

        self.status = "stopped"
        self.error: str | None = None
        self.started_at: str | None = None
        self.latest_jpeg_b64: str = ""
        self.latest_metrics: dict[str, Any] = {}
        self.fps_actual: float = 0.0
        self._last_incident_ts: float = 0.0

    def start(self) -> None:
        self.stop()
        self.error = None
        self.status = "connecting"
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name=f"cam-{self.camera_id}")
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.5)
        with self.lock:
            self.status = "stopped"
            self.latest_jpeg_b64 = ""
            self.latest_metrics = {}
            self.fps_actual = 0.0

    def update_calibration(self, calibration: PerspectiveConfig) -> None:
        with self.lock:
            self.calibration = calibration

    def snapshot_status(self) -> dict[str, Any]:
        with self.lock:
            return {
                "camera_id": self.camera_id,
                "name": self.name,
                "venue_zone": self.venue_zone,
                "masked_url": self.masked_url,
                "status": self.status,
                "error": self.error,
                "started_at": self.started_at,
                "fps": round(self.fps_actual, 2),
                "calibration": self.calibration.to_dict(),
                "metrics": self.latest_metrics,
            }

    def snapshot_frame(self) -> dict[str, Any] | None:
        with self.lock:
            if not self.latest_jpeg_b64:
                return None
            return {
                "camera_id": self.camera_id,
                "image_base64": self.latest_jpeg_b64,
                "metrics": self.latest_metrics,
                "fps": round(self.fps_actual, 2),
            }

    def _maybe_log_incident(
        self,
        risk: dict[str, Any],
        panic: dict[str, Any],
        spatial: dict[str, Any],
        zones: list[dict[str, Any]],
    ) -> None:
        now = time.time()
        if now - self._last_incident_ts < settings.incident_cooldown_seconds:
            return
        score = int(risk["score"])
        p_score = int(panic["panic_score"])
        pred_score = int(risk["predicted_score_30s"])
        if score < 60 and p_score < 55 and pred_score < 70:
            return

        self._last_incident_ts = now
        hotspot = max(zones, key=lambda z: z["density_p_m2"], default={"zone_id": "R1C1"})
        category = (
            panic["panic_type"]
            if panic["panic_type"] != "NORMAL_FLOW"
            else "STAMPEDE_PRECURSOR"
            if pred_score >= 70
            else "CROWD_DENSITY"
        )
        level = "CRITICAL" if max(score, p_score, pred_score) >= 78 else "HIGH"
        db.add_incident(
            {
                "id": f"live-{self.camera_id}-{uuid.uuid4().hex[:8]}",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "camera_id": self.camera_id,
                "source": f"{self.name} ({self.venue_zone})",
                "category": category,
                "level": level,
                "score": score,
                "panic_score": p_score,
                "predicted_score": pred_score,
                "status": "OPEN",
                "message": "; ".join(risk["reasons"][:2]),
                "details": {
                    "hotspot_zone": hotspot["zone_id"],
                    "peak_density_p_m2": spatial["peak_zone_density_p_m2"],
                    "peak_los": spatial["peak_los"],
                    "prediction_trend": risk["prediction_trend"],
                    "etct_seconds": risk["estimated_seconds_to_critical"],
                },
            }
        )

    def _open_capture(self) -> tuple[cv2.VideoCapture | None, SyntheticCrowdStream | None]:
        lower = self.raw_url.lower()
        if lower.startswith("demo://"):
            return None, SyntheticCrowdStream(mode=lower)
        if lower.startswith("webcam:"):
            idx = int(lower.split(":", 1)[1])
            cap = cv2.VideoCapture(idx)
            return cap, None
        cap = cv2.VideoCapture(self.raw_url, cv2.CAP_FFMPEG)
        return cap, None

    def _run(self) -> None:
        cap, sim = self._open_capture()
        if sim is None and (cap is None or not cap.isOpened()):
            if cap is not None:
                cap.release()
            with self.lock:
                self.status = "error"
                self.error = (
                    "Could not connect to camera stream. Check URL, credentials, network, "
                    "and OpenCV FFmpeg support."
                )
            return

        with self.lock:
            self.status = "running"
            self.started_at = datetime.now(timezone.utc).isoformat()
            self.error = None

        tracker = CrowdTracker()
        risk_engine = TemporalRiskEngine(window_size=24)
        previous_frame: np.ndarray | None = None
        previous_count = 0
        min_interval = 1.0 / max(1.0, settings.live_target_fps)
        last_proc = 0.0
        t_start = time.monotonic()

        try:
            while not self.stop_event.is_set():
                now = time.monotonic()
                if sim is not None:
                    elapsed = now - last_proc
                    if elapsed < min_interval:
                        time.sleep(max(0.01, min_interval - elapsed))
                        now = time.monotonic()
                    frame, sim_boxes = sim.next_frame()
                    boxes = sim_boxes
                    detector_status = "simulator"
                else:
                    assert cap is not None
                    ok, frame = cap.read()
                    if not ok:
                        with self.lock:
                            self.error = "Stream interrupted; attempting reconnection..."
                            self.status = "reconnecting"
                        cap.release()
                        time.sleep(1.0)
                        if self.stop_event.is_set():
                            break
                        cap, _ = self._open_capture()
                        if cap is None or not cap.isOpened():
                            with self.lock:
                                self.status = "error"
                                self.error = "Reconnection failed. Stream closed."
                            break
                        with self.lock:
                            self.status = "running"
                            self.error = None
                        continue

                    if now - last_proc < min_interval:
                        continue
                    boxes, detector_status = detect_people(frame)

                dt = max(0.05, now - last_proc) if last_proc > 0 else min_interval
                last_proc = now
                count = len(boxes)

                with self.lock:
                    cal = self.calibration

                track_dyn = tracker.update(boxes, frame.shape, dt=dt)
                flow_dyn = compute_flow_dynamics(previous_frame, frame)
                zones, spatial_summary = compute_zone_metrics(frame, boxes, flow_dyn, cal)
                risk, panic = risk_engine.evaluate(
                    timestamp_s=now - t_start,
                    count=count,
                    count_delta=count - previous_count,
                    spatial_summary=spatial_summary,
                    flow_dyn=flow_dyn,
                    track_dyn=track_dyn,
                )
                _, jpg_b64 = draw_annotations(
                    frame=frame,
                    zones=zones,
                    track_dyn=track_dyn,
                    flow_dyn=flow_dyn,
                    spatial_summary=spatial_summary,
                    risk=risk,
                    panic=panic,
                    source_tag=self.name,
                )

                metrics = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "people_detected": count,
                    "motion_magnitude": flow_dyn["mean_mag"],
                    "flow_dynamics": {k: v for k, v in flow_dyn.items() if k not in {"flow_grid", "vector_samples"}},
                    "tracking_dynamics": {k: v for k, v in track_dyn.items() if k != "tracks"},
                    "spatial_summary": spatial_summary,
                    "panic": panic,
                    "risk": risk,
                    "zone_metrics": zones,
                    "detector": detector_status,
                }

                with self.lock:
                    self.latest_jpeg_b64 = jpg_b64
                    self.latest_metrics = metrics
                    self.fps_actual = round(1.0 / dt, 2)

                self._maybe_log_incident(risk, panic, spatial_summary, zones)

                if self.on_frame_cb is not None and jpg_b64:
                    self.on_frame_cb(
                        self.camera_id,
                        {
                            "type": "frame",
                            "camera_id": self.camera_id,
                            "image_base64": jpg_b64,
                            "metrics": metrics,
                            "fps": round(1.0 / dt, 2),
                        },
                    )

                previous_frame = frame.copy()
                previous_count = count
        finally:
            if cap is not None:
                cap.release()
            with self.lock:
                if self.status != "error":
                    self.status = "stopped"


class CameraManager:
    """Registry managing multiple concurrent venue cameras and WebSocket subscribers."""

    DEFAULT_CAMERA_ID = "cam-primary"

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._workers: dict[str, CameraWorker] = {}
        self._subscribers: dict[str, set[asyncio.Queue]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def _broadcast(self, camera_id: str, payload: dict[str, Any]) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        with self._lock:
            queues = list(self._subscribers.get(camera_id, set()))
        if not queues:
            return

        def _push() -> None:
            for q in queues:
                if q.full():
                    try:
                        q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                try:
                    q.put_nowait(payload)
                except asyncio.QueueFull:
                    pass

        loop.call_soon_threadsafe(_push)

    def subscribe(self, camera_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=4)
        with self._lock:
            self._subscribers.setdefault(camera_id, set()).add(q)
        return q

    def unsubscribe(self, camera_id: str, q: asyncio.Queue) -> None:
        with self._lock:
            subs = self._subscribers.get(camera_id)
            if subs and q in subs:
                subs.remove(q)

    def start_camera(
        self,
        *,
        camera_id: str | None = None,
        name: str = "Venue Camera",
        venue_zone: str = "Main Concourse",
        url: str,
        calibration: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        validated_url = validate_stream_source(url)
        cid = (camera_id or f"cam-{uuid.uuid4().hex[:6]}").strip()
        cal = PerspectiveConfig.from_dict(calibration)

        with self._lock:
            existing = self._workers.get(cid)
            if existing:
                existing.stop()
            worker = CameraWorker(
                camera_id=cid,
                name=name,
                venue_zone=venue_zone,
                raw_url=validated_url,
                calibration=cal,
                on_frame_cb=self._broadcast,
            )
            self._workers[cid] = worker
            db.upsert_camera(cid, name, venue_zone, worker.masked_url, cal.to_dict())

        worker.start()
        return worker.snapshot_status()

    def stop_camera(self, camera_id: str) -> bool:
        with self._lock:
            worker = self._workers.get(camera_id)
        if not worker:
            return False
        worker.stop()
        return True

    def remove_camera(self, camera_id: str) -> bool:
        with self._lock:
            worker = self._workers.pop(camera_id, None)
        if worker:
            worker.stop()
        db.delete_camera(camera_id)
        return worker is not None

    def update_calibration(self, camera_id: str, calibration: dict[str, Any]) -> dict[str, Any] | None:
        cal = PerspectiveConfig.from_dict(calibration)
        with self._lock:
            worker = self._workers.get(camera_id)
            if not worker:
                return None
            worker.update_calibration(cal)
            db.upsert_camera(camera_id, worker.name, worker.venue_zone, worker.masked_url, cal.to_dict())
            return worker.snapshot_status()

    def list_cameras(self) -> list[dict[str, Any]]:
        with self._lock:
            workers = list(self._workers.values())
        return [w.snapshot_status() for w in workers]

    def get_camera(self, camera_id: str) -> CameraWorker | None:
        with self._lock:
            return self._workers.get(camera_id)

    def stop_all(self) -> None:
        with self._lock:
            workers = list(self._workers.values())
        for w in workers:
            w.stop()


camera_manager = CameraManager()

