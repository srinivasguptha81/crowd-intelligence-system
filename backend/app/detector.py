"""Thread-safe YOLOv8 person detector wrapper."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from .config import settings

_lock = threading.Lock()
_model: Any = None
_model_error: str | None = None


def get_detector() -> Any:
    """Lazy-load and cache the Ultralytics YOLO model in a thread-safe manner."""
    global _model, _model_error
    if not settings.enable_yolo:
        return None
    if _model is not None:
        return _model
    if _model_error is not None:
        return None
    with _lock:
        if _model is not None:
            return _model
        if _model_error is not None:
            return None
        try:
            from ultralytics import YOLO

            model_path = settings.yolo_model
            resolved = settings.resolve_path(model_path, settings.resolve_path("backend"))
            load_target = str(resolved) if resolved.exists() else model_path
            _model = YOLO(load_target)
            return _model
        except Exception as exc:
            _model_error = str(exc)
            return None


def detector_status_info() -> dict[str, Any]:
    """Return current YOLO detector availability info."""
    det = get_detector()
    return {
        "enabled": settings.enable_yolo,
        "loaded": det is not None,
        "model": settings.yolo_model,
        "error": _model_error,
    }


def detect_people(frame: np.ndarray) -> tuple[list[list[int]], str]:
    """Run person detection (COCO class 0) on an RGB/BGR frame."""
    detector = get_detector()
    if detector is None:
        return [], "unavailable"
    with _lock:
        results = detector.predict(
            frame,
            imgsz=640,
            conf=settings.yolo_conf,
            classes=[0],
            verbose=False,
        )
    boxes: list[list[int]] = []
    if results and results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int).tolist()
    return boxes, "yolo"

