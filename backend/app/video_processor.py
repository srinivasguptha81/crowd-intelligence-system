"""Recorded video analysis pipeline with tracking, crowd physics, panic detection, and browser-safe encoding."""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from fastapi import HTTPException

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
from .tracker import CrowdTracker


def _open_browser_writer(
    output_dir: Path,
    analysis_id: str,
    fps: float,
    size: tuple[int, int],
) -> tuple[cv2.VideoWriter | None, Path, str]:
    """Try browser-compatible video codecs before falling back to mp4v."""
    candidates: list[tuple[int, str, str, str]] = []
    if os.name == "nt":
        candidates.append((cv2.CAP_MSMF, "H264", ".mp4", "video/mp4"))
    candidates.extend(
        [
            (cv2.CAP_ANY, "avc1", ".mp4", "video/mp4"),
            (cv2.CAP_ANY, "VP80", ".webm", "video/webm"),
            (cv2.CAP_ANY, "mp4v", ".mp4", "video/mp4"),
        ]
    )
    for api_pref, fourcc_str, ext, mime in candidates:
        out_path = output_dir / f"{analysis_id}{ext}"
        try:
            writer = cv2.VideoWriter(
                str(out_path),
                api_pref,
                cv2.VideoWriter_fourcc(*fourcc_str),
                max(1.0, float(fps)),
                size,
            )
            if writer.isOpened():
                return writer, out_path, mime
            writer.release()
            if out_path.exists():
                out_path.unlink(missing_ok=True)
        except Exception:
            continue
    return None, output_dir / f"{analysis_id}.mp4", "video/mp4"


def analyze_video_file(
    path: str,
    *,
    original_filename: str | None = None,
    analysis_id: str | None = None,
    perspective: PerspectiveConfig | None = None,
    progress_cb: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """Analyze a recorded video file end-to-end and persist results."""
    aid = analysis_id or str(uuid.uuid4())
    persp = perspective or PerspectiveConfig.from_dict(None)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Could not open video. Try a valid MP4/MOV/AVI/MKV/WEBM file.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = round(total_frames / fps, 2) if total_frames > 0 else None
    sample_count = min(settings.analysis_max_frames, max(1, total_frames)) if total_frames else settings.analysis_max_frames
    stride = max(1, total_frames // sample_count) if total_frames else 4
    dt_step = stride / max(1.0, fps)

    tracker = CrowdTracker()
    risk_engine = TemporalRiskEngine(window_size=24)

    output_dir = settings.resolve_path(settings.processed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    annotated_frames: list[dict[str, Any]] = []
    previous_frame: np.ndarray | None = None
    previous_count = 0
    idx = 0
    sampled = 0
    writer: cv2.VideoWriter | None = None
    out_path = output_dir / f"{aid}.mp4"
    media_type = "video/mp4"

    try:
        while sampled < settings.analysis_max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride != 0:
                idx += 1
                continue

            time_s = round(idx / fps, 2)
            boxes, detector_status = detect_people(frame)
            count = len(boxes)

            track_dyn = tracker.update(boxes, frame.shape, dt=dt_step)
            flow_dyn = compute_flow_dynamics(previous_frame, frame)
            zones, spatial_summary = compute_zone_metrics(frame, boxes, flow_dyn, persp)
            risk, panic = risk_engine.evaluate(
                timestamp_s=time_s,
                count=count,
                count_delta=count - previous_count,
                spatial_summary=spatial_summary,
                flow_dyn=flow_dyn,
                track_dyn=track_dyn,
            )
            annotated_bgr, annotated_jpg = draw_annotations(
                frame=frame,
                zones=zones,
                track_dyn=track_dyn,
                flow_dyn=flow_dyn,
                spatial_summary=spatial_summary,
                risk=risk,
                panic=panic,
                source_tag=f"Analyze #{aid[:6]}",
            )

            if writer is None:
                h, w = frame.shape[:2]
                writer_fps = max(1.0, fps / stride)
                writer, out_path, media_type = _open_browser_writer(output_dir, aid, writer_fps, (w, h))

            if writer is not None and writer.isOpened():
                writer.write(annotated_bgr)

            row = {
                "frame": idx,
                "time_seconds": time_s,
                "people_detected": count,
                "motion_magnitude": flow_dyn["mean_mag"],
                "flow_dynamics": {k: v for k, v in flow_dyn.items() if k not in {"flow_grid", "vector_samples"}},
                "tracking_dynamics": {k: v for k, v in track_dyn.items() if k != "tracks"},
                "spatial_summary": spatial_summary,
                "panic": panic,
                "risk": risk,
                "zone_metrics": zones,
                "boxes": boxes[:150],
                "detector": detector_status,
            }
            rows.append(row)

            if len(annotated_frames) < 12 or sampled % max(1, sample_count // 12) == 0:
                if len(annotated_frames) < 12:
                    annotated_frames.append(
                        {
                            "time_seconds": time_s,
                            "image_base64": annotated_jpg,
                            "panic_type": panic["panic_type"],
                            "risk_score": risk["score"],
                        }
                    )

            previous_frame = frame.copy()
            previous_count = count
            sampled += 1
            idx += 1

            if progress_cb:
                progress_cb(min(0.95, sampled / float(max(1, sample_count))))
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if not rows:
        raise HTTPException(status_code=400, detail="No readable frames were found in the video.")

    scores = [r["risk"]["score"] for r in rows]
    pred_scores = [r["risk"]["predicted_score_30s"] for r in rows]
    panic_scores = [r["panic"]["panic_score"] for r in rows]

    peak = max(rows, key=lambda r: r["risk"]["score"])
    peak_panic_row = max(rows, key=lambda r: r["panic"]["panic_score"])
    yolo_used = any(r["detector"] == "yolo" for r in rows)
    hotspot = max(peak["zone_metrics"], key=lambda z: z["density_p_m2"])
    display_name = original_filename or Path(path).name

    result = {
        "analysis_id": aid,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "filename": display_name,
        "duration_seconds": duration,
        "fps_source": round(fps, 2),
        "annotated_video_url": f"/api/v1/analysis/{aid}/video",
        "annotated_video_mime": media_type,
        "frames_analyzed": len(rows),
        "detector": "YOLO + Kalman/Hungarian Tracker" if yolo_used else "unavailable",
        "warning": None if yolo_used else "Person detector unavailable. Zero detections must not be interpreted as an empty crowd.",
        "calibration": persp.to_dict(),
        "summary": {
            "max_people_detected": max(r["people_detected"] for r in rows),
            "average_people_detected": round(float(np.mean([r["people_detected"] for r in rows])), 2),
            "peak_density_p_m2": max(r["spatial_summary"]["peak_zone_density_p_m2"] for r in rows),
            "peak_los": peak["spatial_summary"]["peak_los"],
            "max_heuristic_risk": max(scores),
            "max_predicted_risk_30s": max(pred_scores),
            "max_panic_score": max(panic_scores),
            "dominant_panic_type": peak_panic_row["panic"]["panic_type"],
            "peak_crowd_pressure": max(r["panic"]["crowd_pressure"] for r in rows),
            "peak_flow_entropy": max(r["flow_dynamics"]["flow_entropy"] for r in rows),
            "total_fallen_alerts": sum(1 for r in rows if r["tracking_dynamics"]["fallen_count"] > 0),
            "peak_time_seconds": peak["time_seconds"],
            "peak_level": peak["risk"]["level"],
            "hotspot_zone": hotspot["zone_id"],
        },
        "zone_grid": {
            "rows": 3,
            "cols": 3,
            "note": "Perspective-weighted spatial density in people/m^2 (Fruin LOS calibrated to monitored_area_m2).",
        },
        "timeline": rows,
        "annotated_frames": annotated_frames,
        "safety_notice": (
            "Decision-support physics & ML indicators. Requires venue-specific calibration and operator "
            "verification before emergency intervention."
        ),
    }

    db.complete_analysis(aid, str(out_path), media_type, result)

    # Record high-priority incident automatically if threshold crossed
    peak_risk = result["summary"]["max_heuristic_risk"]
    peak_panic = result["summary"]["max_panic_score"]
    peak_pred = result["summary"]["max_predicted_risk_30s"]
    if peak_risk >= 45 or peak_panic >= 50 or peak_pred >= 65:
        cat = (
            peak_panic_row["panic"]["panic_type"]
            if peak_panic_row["panic"]["panic_type"] != "NORMAL_FLOW"
            else "STAMPEDE_PRECURSOR"
            if peak_pred >= 65
            else "CROWD_DENSITY"
        )
        db.add_incident(
            {
                "id": aid,
                "created_at": result["created_at"],
                "camera_id": None,
                "source": display_name,
                "category": cat,
                "level": result["summary"]["peak_level"],
                "score": peak_risk,
                "panic_score": peak_panic,
                "predicted_score": peak_pred,
                "status": "OPEN",
                "message": "; ".join(peak["risk"]["reasons"][:2]),
                "details": {
                    "hotspot_zone": hotspot["zone_id"],
                    "peak_density_p_m2": result["summary"]["peak_density_p_m2"],
                    "peak_los": result["summary"]["peak_los"],
                    "peak_time_seconds": peak["time_seconds"],
                },
            }
        )

    return result

