"""Comprehensive unit & integration tests for CrowdGuard AI."""

from __future__ import annotations

import csv
import math
import subprocess
import sys
import tempfile
import time
import unittest
from collections import deque
from pathlib import Path

import cv2
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.camera_manager import camera_manager
from app.config import settings
from app.crowd_physics import (
    PerspectiveConfig,
    TemporalRiskEngine,
    compute_flow_dynamics,
    compute_zone_metrics,
    detect_panic_signals,
    fruin_los,
)
from app.database import Database, db
from app.main import (
    CameraStartRequest,
    IncidentStatusUpdate,
    delete_camera,
    get_annotated_video,
    get_camera_frame,
    health,
    start_camera,
    stop_camera,
    summary,
    update_incident,
)
from app.security import mask_parsed_url, redact_url_credentials, validate_stream_source
from app.tracker import CrowdTracker
from app.video_processor import analyze_video_file
from fastapi import HTTPException


class TestTrackerAndPhysics(unittest.TestCase):
    def test_kalman_hungarian_tracking_and_fall_detection(self) -> None:
        tracker = CrowdTracker()
        frame_shape = (360, 640, 3)

        for step in range(3):
            x = 100 + step * 15
            res = tracker.update([[x, 100, x + 24, 164]], frame_shape, dt=0.2)
        self.assertEqual(res["active_count"], 1)
        tid = res["tracks"][0]["track_id"]
        self.assertGreater(res["tracks"][0]["vx"], 20.0)
        self.assertEqual(res["fallen_count"], 0)

        # Posture collapse from vertical to horizontal
        res_fall = tracker.update([[130, 140, 194, 164]], frame_shape, dt=0.2)
        self.assertEqual(res_fall["tracks"][0]["track_id"], tid)
        self.assertEqual(res_fall["fallen_count"], 1)
        self.assertTrue(res_fall["tracks"][0]["fallen"])

    def test_radial_dispersion_panic_detection(self) -> None:
        h, w = 180, 320
        f1 = np.zeros((h, w, 3), dtype=np.uint8)
        f2 = np.zeros((h, w, 3), dtype=np.uint8)
        cx, cy = w // 2, h // 2
        for ang in [i * math.pi / 4 for i in range(8)]:
            x1, y1 = int(cx + 22 * math.cos(ang)), int(cy + 22 * math.sin(ang))
            x2, y2 = int(cx + 55 * math.cos(ang)), int(cy + 55 * math.sin(ang))
            cv2.circle(f1, (x1, y1), 10, (230, 230, 230), -1)
            cv2.circle(f2, (x2, y2), 10, (230, 230, 230), -1)

        flow_dyn = compute_flow_dynamics(f1, f2)
        self.assertGreater(flow_dyn["mean_mag"], 0.4)
        self.assertGreater(flow_dyn["positive_divergence"], 0.15)
        self.assertGreater(flow_dyn["flow_entropy"], 0.6)

        persp = PerspectiveConfig(monitored_area_m2=24.0)
        boxes = [[cx - 60, cy - 60, cx - 40, cy - 10] for _ in range(8)]
        _, spatial = compute_zone_metrics(f2, boxes, flow_dyn, persp)
        track_dyn = {
            "active_count": 8,
            "mean_norm_speed": 1.9,
            "speed_variance": 0.45,
            "direction_coherence": 0.25,
            "fallen_count": 0,
        }
        panic = detect_panic_signals(8, 2, spatial, flow_dyn, track_dyn, deque([0.2] * 5))
        self.assertGreaterEqual(panic["panic_score"], 45)
        self.assertIn(panic["panic_type"], {"RADIAL_DISPERSION", "TURBULENT_PANIC"})

    def test_crush_stagnation_detection_at_zero_motion(self) -> None:
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        flow_dyn = compute_flow_dynamics(frame, frame)
        self.assertEqual(flow_dyn["mean_mag"], 0.0)

        boxes = [
            [30 + (i % 4) * 18, 20 + (i // 4) * 20, 45 + (i % 4) * 18, 55 + (i // 4) * 20]
            for i in range(12)
        ]
        persp = PerspectiveConfig(monitored_area_m2=36.0)
        zones, spatial = compute_zone_metrics(frame, boxes, flow_dyn, persp)
        self.assertEqual(fruin_los(spatial["peak_zone_density_p_m2"]), "F")
        self.assertEqual(zones[0]["level"], "HIGH")

        track_dyn = {
            "active_count": 12,
            "mean_norm_speed": 0.05,
            "speed_variance": 0.0,
            "direction_coherence": 1.0,
            "fallen_count": 0,
        }
        panic = detect_panic_signals(12, 4, spatial, flow_dyn, track_dyn, deque())
        self.assertEqual(panic["panic_type"], "CRUSH_STAGNATION")
        self.assertGreaterEqual(panic["panic_score"], 40)

    def test_temporal_forecasting_detects_rising_escalation(self) -> None:
        engine = TemporalRiskEngine(window_size=20)
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        persp = PerspectiveConfig(monitored_area_m2=27.0)

        last_risk: dict = {}
        for step in range(8):
            count = 3 + step * 3
            boxes = [
                [40 + (i % 5) * 20, 40 + (i // 5) * 20, 56 + (i % 5) * 20, 80 + (i // 5) * 20]
                for i in range(count)
            ]
            flow_dyn = {
                "mean_mag": 0.8 + step * 0.45,
                "flow_entropy": 0.3 + step * 0.08,
                "counter_flow": 0.1 + step * 0.06,
                "positive_divergence": 0.05 + step * 0.03,
                "negative_divergence": 0.05,
                "velocity_variance": 0.2 + step * 0.35,
                "flow_grid": [[0.8 + step * 0.4] * 3 for _ in range(3)],
            }
            _, spatial = compute_zone_metrics(frame, boxes, flow_dyn, persp)
            track_dyn = {
                "active_count": count,
                "mean_norm_speed": 0.3 + step * 0.2,
                "speed_variance": 0.05 + step * 0.06,
                "direction_coherence": max(0.2, 1.0 - step * 0.08),
                "fallen_count": 0,
            }
            last_risk, _ = engine.evaluate(step * 1.5, count, 3, spatial, flow_dyn, track_dyn)

        self.assertIn(last_risk["prediction_trend"], {"RISING", "RAPID_ESCALATION"})
        self.assertGreaterEqual(last_risk["predicted_score_30s"], last_risk["score"])


class TestSecurityAndDatabase(unittest.TestCase):
    def test_url_redaction_and_ssrf_policy(self) -> None:
        masked = mask_parsed_url("rtsp://admin:SuperSecret123@192.168.1.55:554/h264")
        self.assertNotIn("SuperSecret123", masked)
        self.assertIn("***:***@", masked)
        self.assertNotIn("pw123", redact_url_credentials("rtsp://user:pw123@10.0.0.5/live"))

        with self.assertRaises(HTTPException) as ctx:
            validate_stream_source("http://169.254.169.254/latest/meta-data/")
        self.assertEqual(ctx.exception.status_code, 403)

        with self.assertRaises(HTTPException) as ctx2:
            validate_stream_source("rtsp://192.168.1.10/stream", allow_private=False)
        self.assertEqual(ctx2.exception.status_code, 403)

        self.assertEqual(validate_stream_source("demo://crowd-sim"), "demo://crowd-sim")
        self.assertEqual(
            validate_stream_source("rtsp://192.168.1.10/stream", allow_private=True),
            "rtsp://192.168.1.10/stream",
        )

    def test_sqlite_persistence_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            test_db = Database(Path(tmpdir) / "test.db")
            test_db.add_incident(
                {
                    "id": "inc-1",
                    "source": "Gate A",
                    "category": "RADIAL_DISPERSION",
                    "level": "HIGH",
                    "score": 78,
                    "panic_score": 82,
                    "predicted_score": 88,
                    "status": "OPEN",
                    "message": "Panic scatter detected",
                }
            )
            items = test_db.list_incidents()
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0]["category"], "RADIAL_DISPERSION")
            self.assertTrue(test_db.update_incident_status("inc-1", "RESOLVED"))
            self.assertEqual(test_db.list_incidents(status="RESOLVED")[0]["status"], "RESOLVED")


class TestIntegrationAndML(unittest.TestCase):
    def test_video_pipeline_live_camera_and_endpoints(self) -> None:
        orig_db_path = db._path
        orig_proc_dir = settings.processed_dir

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            db._path = tmp_path / "e2e_test.db"
            db._init_schema()
            settings.processed_dir = str(tmp_path / "processed")

            try:
                # 1. Health & summary endpoints
                h = health()
                self.assertEqual(h["status"], "ok")
                self.assertEqual(h["tracker"], "kalman_hungarian")

                # 2. Generate synthetic MP4 & run full video processor (YOLO + Tracker + Physics + VideoWriter)
                vid_path = tmp_path / "sample_crowd.mp4"
                writer = cv2.VideoWriter(str(vid_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 240))
                for i in range(6):
                    frame = np.full((240, 320, 3), 30, dtype=np.uint8)
                    cv2.rectangle(frame, (20 + i * 16, 80), (50 + i * 16, 160), (200, 200, 200), -1)
                    writer.write(frame)
                writer.release()

                result = analyze_video_file(
                    str(vid_path),
                    original_filename="sample_crowd.mp4",
                    perspective=PerspectiveConfig(monitored_area_m2=40.0),
                )
                self.assertGreater(result["frames_analyzed"], 0)
                self.assertIn("max_panic_score", result["summary"])
                self.assertIn("max_predicted_risk_30s", result["summary"])

                file_resp = get_annotated_video(result["analysis_id"])
                self.assertTrue(Path(file_resp.path).exists())
                self.assertGreater(Path(file_resp.path).stat().st_size, 200)

                # 3. Live multi-camera manager with simulated panic stream
                cam_req = CameraStartRequest(
                    camera_id="test-cam-live",
                    name="Test Panic Sim",
                    venue_zone="Gate Test",
                    url="demo://panic-sim",
                )
                start_res = start_camera(cam_req)
                self.assertEqual(start_res["camera_id"], "test-cam-live")

                snap = None
                for _ in range(25):
                    try:
                        snap = get_camera_frame("test-cam-live")
                        if snap and snap.get("image_base64"):
                            break
                    except HTTPException:
                        time.sleep(0.1)
                self.assertIsNotNone(snap)
                self.assertIn("panic", snap["metrics"])
                self.assertIn("predicted_score_30s", snap["metrics"]["risk"])

                # 4. Incident lifecycle via endpoint helper
                db.add_incident(
                    {
                        "id": "inc-live-1",
                        "source": "Test Panic Sim",
                        "category": "RADIAL_DISPERSION",
                        "level": "HIGH",
                        "score": 76,
                        "panic_score": 80,
                        "predicted_score": 84,
                        "status": "OPEN",
                        "message": "Test dispersion alert",
                    }
                )
                upd = update_incident("inc-live-1", IncidentStatusUpdate(status="ACKNOWLEDGED"))
                self.assertEqual(upd["status"], "ACKNOWLEDGED")
                self.assertGreaterEqual(summary()["registered_cameras"], 1)

                stop_camera("test-cam-live")
                delete_camera("test-cam-live")
            finally:
                camera_manager.stop_all()
                db._path = orig_db_path
                settings.processed_dir = orig_proc_dir

    def test_ml_temporal_training_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "train.csv"
            out_model = Path(tmpdir) / "model.joblib"
            fieldnames = [
                "video_id",
                "density_signal",
                "motion_signal",
                "surge_signal",
                "peak_density_p_m2",
                "flow_entropy",
                "counter_flow",
                "positive_divergence",
                "negative_divergence",
                "crowd_pressure",
                "crush_stagnation_signal",
                "panic_score_norm",
                "density_slope",
                "pressure_slope",
                "panic_rolling_mean",
                "pressure_rolling_max",
                "event_within_horizon",
            ]
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for vid in range(4):
                    for i in range(12):
                        label = 1 if i >= 7 else 0
                        w.writerow(
                            {
                                "video_id": f"vid_{vid}",
                                "density_signal": 0.2 + 0.6 * label,
                                "motion_signal": 0.2 + 0.5 * label,
                                "surge_signal": 0.1 + 0.5 * label,
                                "peak_density_p_m2": 0.5 + 1.8 * label,
                                "flow_entropy": 0.25 + 0.5 * label,
                                "counter_flow": 0.1 + 0.4 * label,
                                "positive_divergence": 0.05 + 0.3 * label,
                                "negative_divergence": 0.05 + 0.2 * label,
                                "crowd_pressure": 0.3 + 2.5 * label,
                                "crush_stagnation_signal": 0.05 + 0.5 * label,
                                "panic_score_norm": 0.15 + 0.65 * label,
                                "density_slope": 0.01 + 0.15 * label,
                                "pressure_slope": 0.02 + 0.35 * label,
                                "panic_rolling_mean": 0.15 + 0.55 * label,
                                "pressure_rolling_max": 0.4 + 2.8 * label,
                                "event_within_horizon": label,
                            }
                        )

            script = PROJECT_ROOT / "ml" / "training" / "train_risk_model.py"
            proc = subprocess.run(
                [sys.executable, str(script), "--csv", str(csv_path), "--out", str(out_model)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
            self.assertTrue(out_model.exists())


if __name__ == "__main__":
    unittest.main()
