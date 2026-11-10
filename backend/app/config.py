"""Configuration settings for CrowdGuard AI backend."""

from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env."""

    app_name: str = "CrowdGuard AI"
    app_version: str = "1.0.0"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    cors_origin_regex: str = r"https://.*\.vercel\.app"
    max_upload_mb: int = 150
    analysis_max_frames: int = 120
    yolo_model: str = "yolov8n.pt"
    yolo_conf: float = 0.30
    enable_yolo: bool = True
    rtsp_enabled: bool = True
    allow_private_rtsp: bool = True
    rtsp_allowed_hosts: str = ""
    auth_enabled: bool = False
    api_auth_token: str = ""
    model_path: str = "models/crowd_risk_model.joblib"
    database_path: str = "data/crowdguard.db"
    processed_dir: str = "processed"
    default_monitored_area_m2: float = 48.0
    default_perspective_weights: str = "1.8,1.2,0.8"
    live_target_fps: float = 6.0
    incident_cooldown_seconds: float = 15.0

    model_config = SettingsConfigDict(
        env_file=str(BACKEND_ROOT / ".env"),
        extra="ignore",
        case_sensitive=False,
    )

    def resolve_path(self, raw: str, base: Path | None = None) -> Path:
        """Resolve a configured path relative to project root (or given base)."""
        p = Path(raw)
        if p.is_absolute():
            return p
        return (base or PROJECT_ROOT) / p

    def parsed_perspective_weights(self) -> tuple[float, float, float]:
        """Parse top, mid, bot perspective density scaling factors."""
        parts = [x.strip() for x in self.default_perspective_weights.split(",") if x.strip()]
        if len(parts) == 3:
            try:
                vals = tuple(max(0.2, min(5.0, float(v))) for v in parts)
                return (vals[0], vals[1], vals[2])
            except ValueError:
                pass
        return (1.8, 1.2, 0.8)


settings = Settings()

