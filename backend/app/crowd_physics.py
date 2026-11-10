"""Physics-based crowd behavior analysis, panic detection, and temporal stampede-risk forecasting."""

from __future__ import annotations

import base64
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import settings


@dataclass
class PerspectiveConfig:
    """Per-camera ground-plane calibration for physical crowd density (people/m^2)."""

    monitored_area_m2: float = 36.0
    row_weights: tuple[float, float, float] = (1.35, 1.0, 0.85)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PerspectiveConfig":
        if not data:
            return cls(
                monitored_area_m2=settings.default_monitored_area_m2,
                row_weights=settings.parsed_perspective_weights(),
            )
        area = max(4.0, min(2000.0, float(data.get("monitored_area_m2", settings.default_monitored_area_m2))))
        raw_w = data.get("row_weights") or list(settings.parsed_perspective_weights())
        if isinstance(raw_w, (list, tuple)) and len(raw_w) == 3:
            weights = (
                max(0.3, min(4.0, float(raw_w[0]))),
                max(0.3, min(4.0, float(raw_w[1]))),
                max(0.3, min(4.0, float(raw_w[2]))),
            )
        else:
            weights = settings.parsed_perspective_weights()
        return cls(monitored_area_m2=area, row_weights=weights)

    def to_dict(self) -> dict[str, Any]:
        return {
            "monitored_area_m2": self.monitored_area_m2,
            "row_weights": list(self.row_weights),
        }


def fruin_los(density_p_m2: float) -> str:
    """Classify physical pedestrian density into Fruin's Level of Service (A-F)."""
    if density_p_m2 <= 0.31:
        return "A"
    if density_p_m2 <= 0.43:
        return "B"
    if density_p_m2 <= 0.72:
        return "C"
    if density_p_m2 <= 1.08:
        return "D"
    if density_p_m2 <= 2.17:
        return "E"
    return "F"


def compute_flow_dynamics(
    previous_frame: np.ndarray | None,
    current_frame: np.ndarray,
    rows: int = 3,
    cols: int = 3,
) -> dict[str, Any]:
    """
    Compute Farneback dense optical flow with full vector orientation analysis:
    - Mean & max magnitude
    - 8-bin Shannon flow entropy H(theta) in [0, 1]
    - Opposing counter-flow collision index in [0, 1]
    - Spatial divergence (outward panic dispersion vs inward compression)
    - Velocity variance (for Helbing's crowd pressure)
    - Zone-level flow averages & vector grid for HUD overlay
    """
    flow_grid = [[0.0 for _ in range(cols)] for _ in range(rows)]
    empty_vectors: list[tuple[int, int, int, int]] = []

    if previous_frame is None:
        return {
            "mean_mag": 0.0,
            "max_mag": 0.0,
            "flow_entropy": 0.0,
            "counter_flow": 0.0,
            "positive_divergence": 0.0,
            "negative_divergence": 0.0,
            "velocity_variance": 0.0,
            "dominant_angle_deg": 0.0,
            "flow_grid": flow_grid,
            "vector_samples": empty_vectors,
        }

    prev_gray = cv2.resize(cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY), (320, 180))
    curr_gray = cv2.resize(cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY), (320, 180))
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    u = flow[..., 0]
    v = flow[..., 1]
    mag, ang = cv2.cartToPolar(u, v, angleInDegrees=False)

    mean_mag = float(np.mean(mag))
    max_mag = float(np.percentile(mag, 97))
    h, w = mag.shape

    for r in range(rows):
        for c in range(cols):
            y0, y1 = int(r * h / rows), int((r + 1) * h / rows)
            x0, x1 = int(c * w / cols), int((c + 1) * w / cols)
            flow_grid[r][c] = float(np.mean(mag[y0:y1, x0:x1]))

    active_mask = mag > 0.45
    active_count = int(np.sum(active_mask))
    min_active_pixels = max(25, int(0.012 * h * w))

    if active_count >= min_active_pixels:
        active_ang = ang[active_mask]
        hist, _ = np.histogram(active_ang, bins=8, range=(0.0, 2.0 * math.pi))
        probs = hist.astype(float) / max(1, int(hist.sum()))
        nonzero = probs[probs > 0]
        entropy = float(-np.sum(nonzero * np.log2(nonzero)) / math.log2(8))
        entropy = max(0.0, min(1.0, entropy))

        # Opposing orientation bin overlap: pairs (0,4), (1,5), (2,6), (3,7)
        counter_flow = float(sum(2.0 * min(probs[i], probs[i + 4]) for i in range(4)))
        counter_flow = max(0.0, min(1.0, counter_flow))

        mean_u = float(np.mean(u[active_mask]))
        mean_v = float(np.mean(v[active_mask]))
        dom_angle = math.degrees(math.atan2(mean_v, mean_u))
        vel_var = float(np.var(u[active_mask]) + np.var(v[active_mask]))
    else:
        entropy = 0.0
        counter_flow = 0.0
        dom_angle = 0.0
        vel_var = float(np.var(u) + np.var(v))

    # Spatial divergence on smoothed field to distinguish radial panic scattering vs compression
    u_smooth = cv2.GaussianBlur(u, (9, 9), 0)
    v_smooth = cv2.GaussianBlur(v, (9, 9), 0)
    du_dx = np.gradient(u_smooth, axis=1)
    dv_dy = np.gradient(v_smooth, axis=0)
    div = du_dx + dv_dy
    pos_div = float(np.percentile(np.maximum(div, 0.0), 96))
    neg_div = float(abs(np.percentile(np.minimum(div, 0.0), 4)))

    # Coarse vector grid for rendering optical flow arrows onto full-res frame
    fh, fw = current_frame.shape[:2]
    sx = fw / float(w)
    sy = fh / float(h)
    vector_samples: list[tuple[int, int, int, int]] = []
    step_y, step_x = 18, 24
    for gy in range(9, h - 4, step_y):
        for gx in range(12, w - 4, step_x):
            m_val = float(mag[gy, gx])
            if m_val >= 0.7:
                vx_val = float(u_smooth[gy, gx])
                vy_val = float(v_smooth[gy, gx])
                x0 = int(gx * sx)
                y0 = int(gy * sy)
                scale = min(34.0, m_val * 5.0) / max(m_val, 1e-4)
                x1 = int(x0 + vx_val * scale)
                y1 = int(y0 + vy_val * scale)
                vector_samples.append((x0, y0, x1, y1))

    return {
        "mean_mag": round(mean_mag, 3),
        "max_mag": round(max_mag, 3),
        "flow_entropy": round(entropy, 3),
        "counter_flow": round(counter_flow, 3),
        "positive_divergence": round(pos_div, 4),
        "negative_divergence": round(neg_div, 4),
        "velocity_variance": round(vel_var, 3),
        "dominant_angle_deg": round(dom_angle, 1),
        "flow_grid": flow_grid,
        "vector_samples": vector_samples,
    }


def compute_zone_metrics(
    frame: np.ndarray,
    boxes: list[list[int]],
    flow_dynamics: dict[str, Any],
    perspective: PerspectiveConfig,
    rows: int = 3,
    cols: int = 3,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compute calibrated 3x3 spatial zone metrics including physical density (people/m^2) and Fruin LOS."""
    height, width = frame.shape[:2]
    counts = [[0 for _ in range(cols)] for _ in range(rows)]
    for x1, y1, x2, y2 in boxes:
        cx = max(0, min(width - 1, int((x1 + x2) / 2)))
        cy = max(0, min(height - 1, int((y1 + y2) / 2)))
        c = min(cols - 1, int(cx * cols / max(width, 1)))
        r = min(rows - 1, int(cy * rows / max(height, 1)))
        counts[r][c] += 1

    flow_grid = flow_dynamics.get("flow_grid") or [[0.0 for _ in range(cols)] for _ in range(rows)]
    zone_area_m2 = max(0.5, perspective.monitored_area_m2 / float(rows * cols))
    row_weights = perspective.row_weights

    max_count = max((n for line in counts for n in line), default=0)
    result: list[dict[str, Any]] = []
    peak_density = 0.0
    weighted_total_people = 0.0

    for r in range(rows):
        rw = row_weights[min(r, len(row_weights) - 1)]
        for c in range(cols):
            count = counts[r][c]
            motion = float(flow_grid[r][c])
            eff_count = count * rw
            weighted_total_people += eff_count
            density_p_m2 = eff_count / zone_area_m2
            peak_density = max(peak_density, density_p_m2)
            los = fruin_los(density_p_m2)

            if density_p_m2 >= 2.15 or count >= 8 or (density_p_m2 >= 1.25 and motion >= 2.0):
                level = "HIGH"
            elif density_p_m2 >= 0.90 or count >= 4 or (count >= 2 and motion >= 1.3):
                level = "ELEVATED"
            else:
                level = "NORMAL"

            result.append(
                {
                    "zone_id": f"R{r+1}C{c+1}",
                    "row": r,
                    "col": c,
                    "people_detected": count,
                    "effective_people": round(eff_count, 2),
                    "density_p_m2": round(density_p_m2, 3),
                    "fruin_los": los,
                    "relative_density": round(count / max(1, max_count), 3),
                    "motion_magnitude": round(motion, 3),
                    "motion_level": "HIGH" if motion >= 2.8 else "MEDIUM" if motion >= 1.1 else "LOW",
                    "level": level,
                }
            )

    overall_density = weighted_total_people / max(1.0, perspective.monitored_area_m2)
    spatial_summary = {
        "overall_density_p_m2": round(overall_density, 3),
        "peak_zone_density_p_m2": round(peak_density, 3),
        "overall_los": fruin_los(overall_density),
        "peak_los": fruin_los(peak_density),
        "monitored_area_m2": perspective.monitored_area_m2,
    }
    return result, spatial_summary


def detect_panic_signals(
    count: int,
    count_delta: int,
    spatial_summary: dict[str, Any],
    flow_dyn: dict[str, Any],
    track_dyn: dict[str, Any],
    kinetic_history: deque[float],
) -> dict[str, Any]:
    """
    Evaluate dedicated crowd panic and crush signatures:
    1. Radial scattering / dispersion panic (positive flow divergence + speed spike)
    2. Chaotic turbulent panic (high Shannon flow entropy + counter-flow collisions)
    3. Crowd crush / deadlock stagnation (critical density + velocity collapse + compression)
    4. Posture collapse / fallen individual anomaly inside crowd
    """
    peak_density = float(spatial_summary.get("peak_zone_density_p_m2", 0.0))
    overall_density = float(spatial_summary.get("overall_density_p_m2", 0.0))
    mean_mag = float(flow_dyn.get("mean_mag", 0.0))
    entropy = float(flow_dyn.get("flow_entropy", 0.0))
    counter_flow = float(flow_dyn.get("counter_flow", 0.0))
    pos_div = float(flow_dyn.get("positive_divergence", 0.0))
    neg_div = float(flow_dyn.get("negative_divergence", 0.0))
    vel_var = float(flow_dyn.get("velocity_variance", 0.0))

    tracker_speed = float(track_dyn.get("mean_norm_speed", 0.0))
    speed_var = float(track_dyn.get("speed_variance", 0.0))
    coherence = float(track_dyn.get("direction_coherence", 1.0))
    fallen_count = int(track_dyn.get("fallen_count", 0))

    # Crowd Pressure P = peak_density * combined velocity variance (Helbing's turbulence precursor)
    combined_var = vel_var + (speed_var * 3.0)
    crowd_pressure = peak_density * combined_var

    # Kinetic spike z-score against rolling history
    kinetic_energy = 0.5 * max(0.15, overall_density) * ((mean_mag + tracker_speed * 2.0) ** 2)
    if len(kinetic_history) >= 5:
        hist_arr = np.asarray(kinetic_history, dtype=float)
        hist_mean = float(np.mean(hist_arr))
        hist_std = max(0.35, float(np.std(hist_arr)))
        kinetic_z = max(0.0, (kinetic_energy - hist_mean) / hist_std)
    else:
        kinetic_z = 0.0

    # 1. Radial dispersion / scattering panic
    div_norm = min(1.0, pos_div / 0.45)
    speed_norm = min(1.0, max(mean_mag / 5.5, tracker_speed / 2.2))
    disorder_norm = 1.0 - coherence
    dispersion_sig = min(1.0, 0.45 * div_norm + 0.35 * speed_norm + 0.20 * disorder_norm)
    if mean_mag < 0.8 and tracker_speed < 0.4:
        dispersion_sig *= 0.25

    # 2. Chaotic turbulence & counter-flow conflict
    chaos_sig = min(
        1.0,
        (0.55 * entropy + 0.45 * counter_flow) * min(1.0, max(mean_mag / 1.8, tracker_speed / 0.7)),
    )

    # 3. Hyper-compression / crowd crush stagnation (high density + movement freeze/compression)
    dense_factor = min(1.0, max(0.0, (peak_density - 1.0) / 1.5))
    freeze_factor = max(0.0, 1.0 - (mean_mag / 1.1)) if peak_density >= 1.1 else 0.0
    comp_factor = min(1.0, neg_div / 0.35)
    surge_pos = min(1.0, max(0, count_delta) / 6.0)
    crush_stagnation_sig = min(1.0, dense_factor * (0.55 * freeze_factor + 0.25 * comp_factor + 0.20 * surge_pos))

    # 4. Fallen individual inside crowd
    fall_sig = min(1.0, fallen_count * (0.65 if count >= 3 or peak_density >= 0.6 else 0.35))

    # 5. Kinetic spike anomaly
    spike_sig = min(1.0, max(0.0, (kinetic_z - 1.2) / 3.0))

    raw_panic = max(
        0.55 * dispersion_sig + 0.30 * chaos_sig + 0.15 * spike_sig,
        0.75 * crush_stagnation_sig + 0.25 * fall_sig,
        0.85 * fall_sig if (fallen_count > 0 and peak_density >= 1.0) else 0.0,
    )
    panic_score = int(round(min(100.0, max(0.0, raw_panic * 100.0))))

    if panic_score >= 75:
        panic_level = "CRITICAL"
    elif panic_score >= 50:
        panic_level = "WARNING"
    elif panic_score >= 28:
        panic_level = "WATCH"
    else:
        panic_level = "NONE"

    # Determine dominant behavioral category
    reasons: list[str] = []
    if crush_stagnation_sig >= 0.48:
        panic_type = "CRUSH_STAGNATION"
        reasons.append(f"High density ({peak_density:.2f} p/m^2) with low flow velocity indicating compression/deadlock risk")
    elif fall_sig >= 0.50:
        panic_type = "FALL_TRAMPLING_RISK"
        reasons.append(f"{fallen_count} potential fallen person(s) detected inside crowd zone")
    elif dispersion_sig >= 0.48 and pos_div >= 0.16:
        panic_type = "RADIAL_DISPERSION"
        reasons.append("Strong outward radial dispersion and sudden acceleration detected")
    elif counter_flow >= 0.45 and mean_mag >= 1.1:
        panic_type = "COUNTER_FLOW_CONFLICT"
        reasons.append("Opposing crowd streams colliding with high directional conflict")
    elif chaos_sig >= 0.45:
        panic_type = "TURBULENT_PANIC"
        reasons.append("High flow entropy and erratic multi-directional movement")
    else:
        panic_type = "NORMAL_FLOW"

    if spike_sig >= 0.45:
        reasons.append(f"Sudden kinetic energy surge (+{kinetic_z:.1f} sigma above rolling baseline)")
    if crowd_pressure >= 1.8:
        reasons.append(f"Elevated Helbing crowd pressure ({crowd_pressure:.2f})")
    if not reasons:
        reasons.append("Crowd motion and directional coherence remain within baseline thresholds")

    return {
        "panic_score": panic_score,
        "panic_level": panic_level,
        "panic_type": panic_type,
        "crowd_pressure": round(crowd_pressure, 3),
        "kinetic_energy": round(kinetic_energy, 3),
        "kinetic_z_score": round(kinetic_z, 2),
        "signals": {
            "dispersion_signal": round(dispersion_sig, 3),
            "chaos_signal": round(chaos_sig, 3),
            "crush_stagnation_signal": round(crush_stagnation_sig, 3),
            "fall_anomaly_signal": round(fall_sig, 3),
            "kinetic_spike_signal": round(spike_sig, 3),
        },
        "reasons": reasons,
    }


# Canonical feature list for upgraded temporal ML training & inference
TEMPORAL_FEATURES = [
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
]


class TemporalRiskEngine:
    """
    Combines spatial crowd physics, panic indicators, and sliding-window temporal forecasting
    (supports both upgraded multi-feature ML models and legacy 3-feature models, with physics
    spatio-temporal trend forecasting fallback when no trained model file exists).
    """

    def __init__(self, window_size: int = 24) -> None:
        self.window_size = window_size
        self.history: deque[dict[str, float]] = deque(maxlen=window_size)
        self.kinetic_history: deque[float] = deque(maxlen=window_size)
        self._model_bundle: Any = None
        self._model_checked = False

    def reset(self) -> None:
        self.history.clear()
        self.kinetic_history.clear()

    def _get_model(self) -> Any:
        if self._model_checked:
            return self._model_bundle
        self._model_checked = True
        path = settings.resolve_path(settings.model_path)
        if path.exists():
            try:
                import joblib

                self._model_bundle = joblib.load(path)
            except Exception:
                self._model_bundle = None
        return self._model_bundle

    def evaluate(
        self,
        timestamp_s: float,
        count: int,
        count_delta: int,
        spatial_summary: dict[str, Any],
        flow_dyn: dict[str, Any],
        track_dyn: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Evaluate current panic indicators, current risk score, and +30s temporal prediction."""
        panic = detect_panic_signals(
            count=count,
            count_delta=count_delta,
            spatial_summary=spatial_summary,
            flow_dyn=flow_dyn,
            track_dyn=track_dyn,
            kinetic_history=self.kinetic_history,
        )
        self.kinetic_history.append(panic["kinetic_energy"])

        peak_density = float(spatial_summary["peak_zone_density_p_m2"])
        density_sig = min(1.0, max(count / 24.0, peak_density / 2.5))
        motion_sig = min(1.0, max(flow_dyn["mean_mag"] / 7.0, track_dyn["mean_norm_speed"] / 2.2))
        surge_sig = min(1.0, max(0, count_delta) / 8.0)
        pressure_norm = min(1.0, panic["crowd_pressure"] / 4.0)
        panic_norm = panic["panic_score"] / 100.0

        # Current composite physical risk score (0-100)
        base_physics_risk = (
            0.34 * density_sig
            + 0.24 * panic_norm
            + 0.18 * pressure_norm
            + 0.14 * motion_sig
            + 0.10 * surge_sig
        )
        current_score = int(round(min(100.0, max(0.0, base_physics_risk * 100.0))))

        self.history.append(
            {
                "t": float(timestamp_s),
                "risk": float(current_score),
                "density": float(peak_density),
                "pressure": float(panic["crowd_pressure"]),
                "panic": float(panic["panic_score"]),
            }
        )

        # Compute temporal derivatives across sliding window
        density_slope = 0.0
        pressure_slope = 0.0
        risk_slope_per_sec = 0.0
        if len(self.history) >= 3:
            ts = np.array([h["t"] for h in self.history], dtype=float)
            dt_span = float(ts[-1] - ts[0])
            if dt_span > 0.2:
                t_rel = ts - ts[0]
                risk_slope_per_sec = float(np.polyfit(t_rel, [h["risk"] for h in self.history], 1)[0])
                density_slope = float(np.polyfit(t_rel, [h["density"] for h in self.history], 1)[0])
                pressure_slope = float(np.polyfit(t_rel, [h["pressure"] for h in self.history], 1)[0])

        panic_roll_mean = float(np.mean([h["panic"] for h in self.history])) / 100.0
        pressure_roll_max = float(np.max([h["pressure"] for h in self.history]))

        feature_map = {
            "density_signal": round(density_sig, 4),
            "motion_signal": round(motion_sig, 4),
            "surge_signal": round(surge_sig, 4),
            "peak_density_p_m2": round(peak_density, 4),
            "flow_entropy": float(flow_dyn["flow_entropy"]),
            "counter_flow": float(flow_dyn["counter_flow"]),
            "positive_divergence": float(flow_dyn["positive_divergence"]),
            "negative_divergence": float(flow_dyn["negative_divergence"]),
            "crowd_pressure": float(panic["crowd_pressure"]),
            "crush_stagnation_signal": float(panic["signals"]["crush_stagnation_signal"]),
            "panic_score_norm": round(panic_norm, 4),
            "density_slope": round(density_slope, 4),
            "pressure_slope": round(pressure_slope, 4),
            "panic_rolling_mean": round(panic_roll_mean, 4),
            "pressure_rolling_max": round(pressure_roll_max, 4),
        }

        # Predict +30s risk horizon using trained model if present, else physics temporal extrapolation
        bundle = self._get_model()
        model_used = False
        probability: float | None = None
        predicted_30s = current_score
        model_label = "physics_temporal_forecaster"

        if bundle is not None:
            try:
                clf = bundle["model"] if isinstance(bundle, dict) and "model" in bundle else bundle
                feat_cols = (
                    bundle.get("features")
                    if isinstance(bundle, dict)
                    else list(getattr(clf, "feature_names_in_", []))
                )
                if not feat_cols:
                    feat_cols = ["density_signal", "motion_signal", "surge_signal"]
                x_vec = np.array([[float(feature_map.get(c, 0.0)) for c in feat_cols]], dtype=float)
                if hasattr(clf, "predict_proba"):
                    probability = float(clf.predict_proba(x_vec)[0][-1])
                    predicted_30s = int(round(min(100.0, max(0.0, probability * 100.0))))
                    model_used = True
                    model_label = "temporal_ml_model"
                else:
                    raw_pred = float(clf.predict(x_vec)[0])
                    predicted_30s = int(round(min(100.0, max(0.0, raw_pred))))
                    probability = predicted_30s / 100.0
                    model_used = True
                    model_label = "temporal_ml_model"
            except Exception:
                model_used = False

        if not model_used:
            # Physics-based temporal extrapolation over 30-second forecast horizon
            # Dampen noisy spikes over long horizons using bounded slope extrapolation
            clamped_slope = max(-2.0, min(2.5, risk_slope_per_sec))
            pressure_boost = max(0.0, min(20.0, pressure_slope * 14.0 + density_slope * 18.0))
            projected = current_score + clamped_slope * 18.0 + pressure_boost
            # Smoothly blend with rolling baseline so single transient frames don't overshoot
            ewma_risk = float(np.mean([h["risk"] for h in list(self.history)[-6:]]))
            predicted_30s = int(round(min(100.0, max(0.0, 0.65 * projected + 0.35 * ewma_risk))))

        # Determine escalation trend and Estimated Time to Critical Threshold (ETCT >= 75)
        delta_forecast = predicted_30s - current_score
        if risk_slope_per_sec >= 0.9 or delta_forecast >= 14:
            trend = "RAPID_ESCALATION"
        elif risk_slope_per_sec >= 0.35 or delta_forecast >= 6:
            trend = "RISING"
        elif risk_slope_per_sec <= -0.45 or delta_forecast <= -8:
            trend = "DEESCALATING"
        else:
            trend = "STABLE"

        etct_seconds: int | None = None
        if current_score >= 75:
            etct_seconds = 0
        elif risk_slope_per_sec > 0.25:
            est = (75.0 - current_score) / risk_slope_per_sec
            if 0 < est <= 180:
                etct_seconds = int(round(est))

        level = (
            "CRITICAL"
            if current_score >= 78
            else "HIGH"
            if current_score >= 60
            else "ELEVATED"
            if current_score >= 35
            else "NORMAL"
        )
        predicted_level = (
            "CRITICAL"
            if predicted_30s >= 78
            else "HIGH"
            if predicted_30s >= 60
            else "ELEVATED"
            if predicted_30s >= 35
            else "NORMAL"
        )

        reasons = list(panic["reasons"])
        if trend == "RAPID_ESCALATION":
            reasons.append(f"Rapid risk escalation (+{delta_forecast} pts projected over 30s horizon)")
        elif trend == "RISING":
            reasons.append("Crowd risk trending upward over trailing temporal window")

        risk_payload = {
            "score": current_score,
            "level": level,
            "predicted_score_30s": predicted_30s,
            "predicted_level_30s": predicted_level,
            "prediction_trend": trend,
            "estimated_seconds_to_critical": etct_seconds,
            "risk_slope_per_sec": round(risk_slope_per_sec, 3),
            "signals": {
                "density_signal": round(density_sig, 3),
                "motion_signal": round(motion_sig, 3),
                "surge_signal": round(surge_sig, 3),
                "pressure_signal": round(pressure_norm, 3),
                "panic_signal": round(panic_norm, 3),
            },
            "feature_vector": feature_map,
            "reasons": reasons,
            "model_type": model_label,
            "probability": round(probability, 4) if probability is not None else None,
            "is_validated_probability": False,
        }
        return risk_payload, panic


def _level_color_bgr(level: str) -> tuple[int, int, int]:
    if level == "CRITICAL":
        return (40, 35, 245)
    if level == "HIGH":
        return (45, 55, 235)
    if level == "ELEVATED":
        return (0, 190, 255)
    return (60, 195, 95)


def draw_annotations(
    frame: np.ndarray,
    zones: list[dict[str, Any]],
    track_dyn: dict[str, Any],
    flow_dyn: dict[str, Any],
    spatial_summary: dict[str, Any],
    risk: dict[str, Any],
    panic: dict[str, Any],
    source_tag: str = "CrowdGuard AI",
) -> tuple[np.ndarray, str]:
    """Render calibrated zones, flow vectors, person tracks/trails, and telemetry HUD onto frame."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]
    zone_map = {(z["row"], z["col"]): z for z in zones}
    rows = max(1, max((z["row"] for z in zones), default=2) + 1)
    cols = max(1, max((z["col"] for z in zones), default=2) + 1)

    # 1. Zone grid overlay with calibrated physical density
    overlay = annotated.copy()
    for r in range(rows):
        for c in range(cols):
            z = zone_map.get((r, c), {})
            level = z.get("level", "NORMAL")
            color = _level_color_bgr(level)
            x1, x2 = int(c * w / cols), int((c + 1) * w / cols)
            y1, y2 = int(r * h / rows), int((r + 1) * h / rows)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    annotated = cv2.addWeighted(overlay, 0.10, annotated, 0.90, 0)

    for r in range(rows):
        for c in range(cols):
            z = zone_map.get((r, c), {})
            level = z.get("level", "NORMAL")
            color = _level_color_bgr(level)
            x1, x2 = int(c * w / cols), int((c + 1) * w / cols)
            y1, y2 = int(r * h / rows), int((r + 1) * h / rows)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)
            label = f"R{r+1}C{c+1}: {z.get('people_detected', 0)}p | {z.get('density_p_m2', 0):.1f}/m2 | LOS {z.get('fruin_los', 'A')}"
            cv2.putText(
                annotated,
                label,
                (x1 + 6, min(y2 - 8, y1 + 52 if r == 0 else y1 + 22)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )

    # 2. Optical flow direction vector arrows
    for x0, y0, x1, y1 in flow_dyn.get("vector_samples", []):
        cv2.arrowedLine(annotated, (x0, y0), (x1, y1), (220, 225, 110), 1, tipLength=0.35)

    # 3. Tracked person bounding boxes, Track IDs, velocity trails, and fall alerts
    for tr in track_dyn.get("tracks", []):
        bx1, by1, bx2, by2 = tr["bbox"]
        fallen = bool(tr.get("fallen"))
        cx = max(0, min(w - 1, int((bx1 + bx2) / 2)))
        cy = max(0, min(h - 1, int((by1 + by2) / 2)))
        r_idx = min(rows - 1, max(0, int(cy * rows / max(h, 1))))
        c_idx = min(cols - 1, max(0, int(cx * cols / max(w, 1))))
        z_level = zone_map.get((r_idx, c_idx), {}).get("level", "NORMAL")
        color = (220, 40, 255) if fallen else _level_color_bgr(z_level)

        trail = tr.get("trail") or []
        for idx in range(1, len(trail)):
            cv2.line(annotated, tuple(trail[idx - 1]), tuple(trail[idx]), color, 2)

        cv2.rectangle(annotated, (bx1, by1), (bx2, by2), color, 3 if fallen else 2)
        tag = f"FALLEN #{tr['track_id']}" if fallen else f"ID#{tr['track_id']}"
        cv2.putText(
            annotated,
            tag,
            (bx1, max(44, by1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            color,
            1,
            cv2.LINE_AA,
        )

    # 4. Top Telemetry HUD Banner
    cv2.rectangle(annotated, (0, 0), (w, 36), (10, 16, 28), -1)
    hud_left = (
        f"{source_tag} | People:{track_dyn.get('active_count', 0)} | "
        f"Peak:{spatial_summary.get('peak_zone_density_p_m2', 0):.2f}p/m2(LOS {spatial_summary.get('peak_los', 'A')}) | "
        f"Entropy:{flow_dyn.get('flow_entropy', 0):.2f} | "
        f"Panic:{panic.get('panic_score', 0)} ({panic.get('panic_type', 'NORMAL_FLOW')}) | "
        f"Risk:{risk.get('score', 0)} -> +30s:{risk.get('predicted_score_30s', 0)}"
    )
    cv2.putText(
        annotated,
        hud_left,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        (235, 245, 255),
        1,
        cv2.LINE_AA,
    )

    ok, buffer = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
    b64 = base64.b64encode(buffer).decode("ascii") if ok else ""
    return annotated, b64

