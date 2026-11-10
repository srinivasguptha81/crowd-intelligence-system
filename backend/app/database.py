"""Thread-safe SQLite persistence for incidents, video analyses, and venue camera profiles."""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .config import settings


class Database:
    """Lightweight SQLite store for CrowdGuard AI persistent state."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._path = db_path or settings.resolve_path(settings.database_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self._path), check_same_thread=False, timeout=15.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS incidents (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    camera_id TEXT,
                    source TEXT NOT NULL,
                    category TEXT NOT NULL,
                    level TEXT NOT NULL,
                    score INTEGER NOT NULL,
                    panic_score INTEGER DEFAULT 0,
                    predicted_score INTEGER DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'OPEN',
                    message TEXT NOT NULL,
                    details_json TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_incidents_created
                ON incidents (created_at DESC);

                CREATE TABLE IF NOT EXISTS analyses (
                    analysis_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress REAL DEFAULT 0.0,
                    video_path TEXT,
                    media_type TEXT DEFAULT 'video/mp4',
                    error TEXT,
                    result_json TEXT
                );

                CREATE TABLE IF NOT EXISTS cameras (
                    camera_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    venue_zone TEXT DEFAULT 'Main Concourse',
                    masked_url TEXT NOT NULL,
                    calibration_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    # --- Incidents ---

    def add_incident(self, item: dict[str, Any]) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO incidents (
                    id, created_at, camera_id, source, category, level,
                    score, panic_score, predicted_score, status, message, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item["id"],
                    item.get("created_at") or datetime.now(timezone.utc).isoformat(),
                    item.get("camera_id"),
                    item.get("source", "unknown"),
                    item.get("category", "CROWD_RISK"),
                    item.get("level", "ELEVATED"),
                    int(item.get("score", 0)),
                    int(item.get("panic_score", 0)),
                    int(item.get("predicted_score", 0)),
                    item.get("status", "OPEN"),
                    item.get("message", ""),
                    json.dumps(item.get("details") or {}),
                ),
            )

    def list_incidents(self, limit: int = 100, status: str | None = None) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM incidents WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                    (status.upper(), limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM incidents ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        results: list[dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            raw_details = d.pop("details_json", None)
            try:
                d["details"] = json.loads(raw_details) if raw_details else {}
            except Exception:
                d["details"] = {}
            results.append(d)
        return results

    def update_incident_status(self, incident_id: str, new_status: str) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "UPDATE incidents SET status = ? WHERE id = ?",
                (new_status.upper(), incident_id),
            )
            return cur.rowcount > 0

    def clear_incidents(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM incidents")

    # --- Analyses ---

    def save_analysis_job(
        self,
        analysis_id: str,
        filename: str,
        status: str = "PROCESSING",
        progress: float = 0.0,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO analyses (
                    analysis_id, created_at, filename, status, progress
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    analysis_id,
                    datetime.now(timezone.utc).isoformat(),
                    filename,
                    status,
                    float(progress),
                ),
            )

    def update_analysis_progress(self, analysis_id: str, progress: float) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE analyses SET progress = ? WHERE analysis_id = ?",
                (float(progress), analysis_id),
            )

    def complete_analysis(
        self,
        analysis_id: str,
        video_path: str,
        media_type: str,
        result: dict[str, Any],
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO analyses (
                    analysis_id, created_at, filename, status, progress, video_path, media_type, result_json
                ) VALUES (?, ?, ?, 'COMPLETED', 1.0, ?, ?, ?)
                ON CONFLICT(analysis_id) DO UPDATE SET
                    status = 'COMPLETED',
                    progress = 1.0,
                    video_path = excluded.video_path,
                    media_type = excluded.media_type,
                    result_json = excluded.result_json
                """,
                (
                    analysis_id,
                    result.get("created_at") or datetime.now(timezone.utc).isoformat(),
                    result.get("filename", "video"),
                    video_path,
                    media_type,
                    json.dumps(result),
                ),
            )

    def fail_analysis(self, analysis_id: str, error_msg: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE analyses SET status = 'FAILED', error = ? WHERE analysis_id = ?",
                (error_msg, analysis_id),
            )

    def get_analysis(self, analysis_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM analyses WHERE analysis_id = ?",
                (analysis_id,),
            ).fetchone()
        if not row:
            return None
        d = dict(row)
        raw_res = d.pop("result_json", None)
        d["result"] = json.loads(raw_res) if raw_res else None
        return d

    def list_analyses(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT analysis_id, created_at, filename, status, progress, error FROM analyses ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    # --- Cameras ---

    def upsert_camera(
        self,
        camera_id: str,
        name: str,
        venue_zone: str,
        masked_url: str,
        calibration: dict[str, Any],
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cameras (
                    camera_id, name, venue_zone, masked_url, calibration_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    camera_id,
                    name,
                    venue_zone,
                    masked_url,
                    json.dumps(calibration),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def delete_camera(self, camera_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM cameras WHERE camera_id = ?", (camera_id,))

    def list_cameras(self) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT * FROM cameras ORDER BY created_at ASC").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["calibration"] = json.loads(d.pop("calibration_json"))
            out.append(d)
        return out


db = Database()
