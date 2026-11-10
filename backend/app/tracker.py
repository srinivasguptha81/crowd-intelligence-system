"""Lightweight Kalman-velocity + Hungarian Multi-Object Tracker using SciPy."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


def _box_iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return float(inter / (area_a + area_b - inter))


@dataclass
class TrackState:
    """Internal state for a single tracked person."""

    track_id: int
    bbox: list[int]
    cx: float
    cy: float
    w: float
    h: float
    vx: float = 0.0
    vy: float = 0.0
    prev_speed: float = 0.0
    hits: int = 1
    misses: int = 0
    history: deque[tuple[int, int]] = field(default_factory=lambda: deque(maxlen=14))
    aspect_history: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    fallen: bool = False

    def predict(self, dt: float) -> tuple[float, float, list[int]]:
        pcx = self.cx + self.vx * dt
        pcy = self.cy + self.vy * dt
        hw, hh = self.w / 2.0, self.h / 2.0
        pbbox = [int(pcx - hw), int(pcy - hh), int(pcx + hw), int(pcy + hh)]
        return pcx, pcy, pbbox

    def update(self, bbox: list[int], dt: float) -> None:
        x1, y1, x2, y2 = bbox
        ncx = (x1 + x2) / 2.0
        ncy = (y1 + y2) / 2.0
        nw = max(4.0, float(x2 - x1))
        nh = max(4.0, float(y2 - y1))

        safe_dt = max(0.03, dt)
        inst_vx = (ncx - self.cx) / safe_dt
        inst_vy = (ncy - self.cy) / safe_dt

        # Smooth velocity estimate (alpha-beta style filter)
        alpha = 0.55
        self.prev_speed = math.hypot(self.vx, self.vy)
        self.vx = alpha * inst_vx + (1.0 - alpha) * self.vx
        self.vy = alpha * inst_vy + (1.0 - alpha) * self.vy

        self.cx = ncx
        self.cy = ncy
        self.w = 0.7 * nw + 0.3 * self.w
        self.h = 0.7 * nh + 0.3 * self.h
        self.bbox = [int(x1), int(y1), int(x2), int(y2)]
        self.hits += 1
        self.misses = 0
        self.history.append((int(round(ncx)), int(round(ncy))))

        ar = nw / max(nh, 1.0)
        self.aspect_history.append(ar)
        # Detect posture collapse / fallen individual when upright history flips horizontal
        if len(self.aspect_history) >= 3:
            prior_median = float(np.median(list(self.aspect_history)[:-1]))
            self.fallen = (prior_median < 0.78 and ar > 1.15) or (ar > 1.38 and nh < 80)
        else:
            self.fallen = ar > 1.45

    @property
    def speed_px_s(self) -> float:
        return math.hypot(self.vx, self.vy)

    @property
    def norm_speed(self) -> float:
        """Speed expressed in bounding-box heights per second (perspective-invariant)."""
        return self.speed_px_s / max(self.h, 20.0)

    @property
    def accel_px_s2(self) -> float:
        return self.speed_px_s - self.prev_speed

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "bbox": self.bbox,
            "centroid": [round(self.cx, 1), round(self.cy, 1)],
            "vx": round(self.vx, 2),
            "vy": round(self.vy, 2),
            "speed_px_s": round(self.speed_px_s, 2),
            "norm_speed": round(self.norm_speed, 3),
            "heading_rad": round(math.atan2(self.vy, self.vx), 3),
            "hits": self.hits,
            "fallen": self.fallen,
            "trail": list(self.history),
        }


class CrowdTracker:
    """Multi-object tracker associating YOLO person boxes across frames via Hungarian matching."""

    def __init__(self, max_misses: int = 4, max_match_dist_ratio: float = 0.16) -> None:
        self.max_misses = max_misses
        self.max_match_dist_ratio = max_match_dist_ratio
        self._next_id = 1
        self._tracks: dict[int, TrackState] = {}

    def reset(self) -> None:
        self._next_id = 1
        self._tracks.clear()

    def update(
        self,
        boxes: list[list[int]],
        frame_shape: tuple[int, ...],
        dt: float = 0.2,
    ) -> dict[str, Any]:
        """Match detections to active tracks and return active track summaries & dynamics."""
        h, w = frame_shape[:2]
        frame_diag = max(100.0, math.hypot(w, h))
        max_dist = frame_diag * self.max_match_dist_ratio

        track_ids = list(self._tracks.keys())
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()

        if track_ids and boxes:
            cost = np.full((len(track_ids), len(boxes)), 1e5, dtype=float)
            for ti, tid in enumerate(track_ids):
                tr = self._tracks[tid]
                pcx, pcy, pbbox = tr.predict(dt)
                for di, det in enumerate(boxes):
                    dcx = (det[0] + det[2]) / 2.0
                    dcy = (det[1] + det[3]) / 2.0
                    dist = math.hypot(dcx - pcx, dcy - pcy)
                    iou = _box_iou(pbbox, det)
                    if iou >= 0.12:
                        cost[ti, di] = 0.55 * (1.0 - iou) + 0.45 * min(1.0, dist / max_dist)
                    elif dist <= max_dist:
                        cost[ti, di] = 0.65 + 0.35 * (dist / max_dist)

            row_idx, col_idx = linear_sum_assignment(cost)
            for r, c in zip(row_idx, col_idx):
                if cost[r, c] <= 1.0:
                    tid = track_ids[r]
                    self._tracks[tid].update(boxes[c], dt)
                    matched_tracks.add(tid)
                    matched_dets.add(c)

        # Increment misses for unmatched tracks & prune expired tracks
        for tid in track_ids:
            if tid not in matched_tracks:
                self._tracks[tid].misses += 1
                if self._tracks[tid].misses > self.max_misses:
                    del self._tracks[tid]

        # Initialize new tracks for unmatched detections
        for di, det in enumerate(boxes):
            if di not in matched_dets:
                x1, y1, x2, y2 = det
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                bw = max(4.0, float(x2 - x1))
                bh = max(4.0, float(y2 - y1))
                tid = self._next_id
                self._next_id += 1
                st = TrackState(
                    track_id=tid,
                    bbox=[int(x1), int(y1), int(x2), int(y2)],
                    cx=cx,
                    cy=cy,
                    w=bw,
                    h=bh,
                )
                st.history.append((int(round(cx)), int(round(cy))))
                st.aspect_history.append(bw / max(bh, 1.0))
                self._tracks[tid] = st

        active = [tr for tr in self._tracks.values() if tr.misses == 0]
        speeds = [tr.norm_speed for tr in active if tr.hits >= 2]
        accels = [abs(tr.accel_px_s2) for tr in active if tr.hits >= 2]
        fallen = [tr for tr in active if tr.fallen]

        # Direction coherence across tracked people (mean resultant vector length R in [0, 1])
        moving = [tr for tr in active if tr.hits >= 2 and tr.norm_speed > 0.12]
        if len(moving) >= 2:
            cos_sum = sum(math.cos(math.atan2(tr.vy, tr.vx)) for tr in moving)
            sin_sum = sum(math.sin(math.atan2(tr.vy, tr.vx)) for tr in moving)
            coherence = math.hypot(cos_sum, sin_sum) / len(moving)
        else:
            coherence = 1.0

        return {
            "active_count": len(active),
            "tracks": [tr.to_dict() for tr in active],
            "mean_norm_speed": round(float(np.mean(speeds)), 3) if speeds else 0.0,
            "speed_variance": round(float(np.var(speeds)), 4) if len(speeds) >= 2 else 0.0,
            "mean_accel": round(float(np.mean(accels)), 2) if accels else 0.0,
            "direction_coherence": round(coherence, 3),
            "fallen_count": len(fallen),
            "fallen_track_ids": [tr.track_id for tr in fallen],
        }

