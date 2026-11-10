"""
Extract calibrated crowd-physics and temporal behavior feature windows from video files.

Usage:
    python ml/training/extract_features.py --video path/to/video.mp4 --out ml/data/features.csv --area-m2 36.0

Outputs a CSV containing all 15 behavioral/temporal features plus timestamp and video_id,
with a placeholder `event_within_horizon` column (default empty) for manual ground-truth labeling.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.crowd_physics import (  # noqa: E402
    TEMPORAL_FEATURES,
    PerspectiveConfig,
    TemporalRiskEngine,
    compute_flow_dynamics,
    compute_zone_metrics,
)
from app.detector import detect_people  # noqa: E402
from app.tracker import CrowdTracker  # noqa: E402


def extract_from_video(video_path: Path, area_m2: float, max_frames: int) -> list[dict]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, total // max_frames) if total > 0 else 4
    dt = stride / max(1.0, fps)

    tracker = CrowdTracker()
    engine = TemporalRiskEngine(window_size=24)
    persp = PerspectiveConfig(monitored_area_m2=area_m2)

    rows: list[dict] = []
    prev_frame = None
    prev_count = 0
    idx = 0
    sampled = 0

    while sampled < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride != 0:
            idx += 1
            continue

        t_s = round(idx / fps, 3)
        boxes, _ = detect_people(frame)
        count = len(boxes)
        track_dyn = tracker.update(boxes, frame.shape, dt=dt)
        flow_dyn = compute_flow_dynamics(prev_frame, frame)
        _, spatial_summary = compute_zone_metrics(frame, boxes, flow_dyn, persp)
        risk, _ = engine.evaluate(
            timestamp_s=t_s,
            count=count,
            count_delta=count - prev_count,
            spatial_summary=spatial_summary,
            flow_dyn=flow_dyn,
            track_dyn=track_dyn,
        )

        row = {"video_id": video_path.stem, "time_seconds": t_s}
        row.update(risk["feature_vector"])
        row["event_within_horizon"] = ""
        rows.append(row)

        prev_frame = frame.copy()
        prev_count = count
        sampled += 1
        idx += 1

    cap.release()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract crowd-physics temporal features from video.")
    parser.add_argument("--video", required=True, help="Path to video file (.mp4/.mov/.avi/.mkv)")
    parser.add_argument("--out", default="ml/data/extracted_features.csv", help="Output CSV path")
    parser.add_argument("--area-m2", type=float, default=36.0, help="Estimated monitored floor area in m^2")
    parser.add_argument("--max-frames", type=int, default=200, help="Maximum sampled frames to extract")
    args = parser.parse_args()

    video_path = Path(args.video)
    rows = extract_from_video(video_path, args.area_m2, args.max_frames)
    if not rows:
        raise SystemExit("No frames extracted from video.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["video_id", "time_seconds"] + TEMPORAL_FEATURES + ["event_within_horizon"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Extracted {len(rows)} feature windows to {out_path.resolve()}")
    print("Populate `event_within_horizon` (0/1) with verified horizon annotations before training.")


if __name__ == "__main__":
    main()

