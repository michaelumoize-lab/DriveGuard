"""
DriveGuard — Thread-Safe SQLite WAL Local Persistence Store for Standalone Python AI.

Provides zero-dependency local persistence (`backend/data/driveguard_local.db`) so
the Python AI engine records driving sessions, idempotent safety alert transitions
(keyed by unique `event_id`), and a sync outbox completely offline without requiring
Next.js, Better Auth, LiveKit, or Neon PostgreSQL.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.app.core.config import settings

logger = logging.getLogger("driveguard.services.local_db")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _generate_local_session_code() -> str:
    token = uuid.uuid4().hex[:8].upper()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"DRV-PY-{ts}-{token}"


class LocalDB:
    """
    Thread-safe SQLite (WAL mode) persistence store for standalone Python monitoring.
    """

    def __init__(self, db_path: Optional[Path | str] = None) -> None:
        resolved = Path(db_path) if db_path is not None else settings.LOCAL_DB_PATH
        self.db_path: Path = resolved
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            timeout=10.0,
        )
        self._conn.row_factory = sqlite3.Row
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL;")
            cur.execute("PRAGMA synchronous=NORMAL;")
            cur.execute("PRAGMA foreign_keys=ON;")
            cur.execute("PRAGMA busy_timeout=5000;")

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS local_sessions (
                    session_id TEXT PRIMARY KEY,
                    session_code TEXT UNIQUE NOT NULL,
                    driver_id TEXT NOT NULL,
                    engine_type TEXT NOT NULL DEFAULT 'PYTHON',
                    camera_source TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'ACTIVE',
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    duration_seconds INTEGER NOT NULL DEFAULT 0,
                    total_frames INTEGER NOT NULL DEFAULT 0,
                    normal_frames_count INTEGER NOT NULL DEFAULT 0,
                    average_fps REAL NOT NULL DEFAULT 0.0,
                    average_latency_ms REAL NOT NULL DEFAULT 0.0,
                    average_perclos_pct REAL NOT NULL DEFAULT 0.0,
                    max_perclos_pct REAL NOT NULL DEFAULT 0.0,
                    total_blinks INTEGER NOT NULL DEFAULT 0,
                    average_blink_rate REAL NOT NULL DEFAULT 0.0,
                    warning_alerts_count INTEGER NOT NULL DEFAULT 0,
                    critical_alerts_count INTEGER NOT NULL DEFAULT 0,
                    mirror_suppressions INTEGER NOT NULL DEFAULT 0,
                    profile_uuid TEXT,
                    profile_version INTEGER,
                    sync_status TEXT NOT NULL DEFAULT 'LOCAL_ONLY'
                );
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS local_alerts (
                    event_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES local_sessions(session_id) ON DELETE CASCADE,
                    session_code TEXT NOT NULL,
                    driver_id TEXT NOT NULL,
                    engine_type TEXT NOT NULL DEFAULT 'PYTHON',
                    severity TEXT NOT NULL,
                    previous_state TEXT,
                    new_state TEXT NOT NULL,
                    risk_score REAL NOT NULL DEFAULT 0.0,
                    ear_value REAL,
                    perclos_pct REAL,
                    pose_zone TEXT,
                    reasons_json TEXT NOT NULL DEFAULT '[]',
                    timestamp_ms INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    sync_status TEXT NOT NULL DEFAULT 'LOCAL_ONLY'
                );
                """
            )

            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_local_alerts_session_ts
                ON local_alerts (session_id, timestamp_ms DESC);
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS local_sync_outbox (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity_type TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'PENDING',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(entity_type, entity_id)
                );
                """
            )

            self._conn.commit()

    def create_session(
        self,
        driver_id: str,
        camera_source: str,
        engine_type: str = "PYTHON",
        session_id: Optional[str] = None,
        session_code: Optional[str] = None,
        profile_uuid: Optional[str] = None,
        profile_version: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Create and persist a new local driving session and enqueue SESSION_START in outbox."""
        sid = session_id or str(uuid.uuid4())
        scode = session_code or _generate_local_session_code()
        now = _utc_now_iso()

        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO local_sessions (
                    session_id, session_code, driver_id, engine_type, camera_source,
                    status, started_at, profile_uuid, profile_version, sync_status
                ) VALUES (?, ?, ?, ?, ?, 'ACTIVE', ?, ?, ?, 'LOCAL_ONLY')
                """,
                (
                    sid,
                    scode,
                    driver_id,
                    engine_type,
                    str(camera_source),
                    now,
                    profile_uuid,
                    profile_version,
                ),
            )
            outbox_payload = json.dumps(
                {
                    "clientSessionId": sid,
                    "sessionCode": scode,
                    "driverId": driver_id,
                    "engineType": engine_type,
                    "cameraSource": str(camera_source),
                    "startedAt": now,
                }
            )
            cur.execute(
                """
                INSERT OR IGNORE INTO local_sync_outbox (
                    entity_type, entity_id, payload_json, status, created_at, updated_at
                ) VALUES ('SESSION_START', ?, ?, 'PENDING', ?, ?)
                """,
                (sid, outbox_payload, now, now),
            )
            self._conn.commit()
            return self.get_session(sid) or {}

    def record_alert(
        self,
        *,
        event_id: str,
        session_id: str,
        session_code: str,
        driver_id: str,
        severity: str,
        new_state: str,
        timestamp_ms: int,
        previous_state: Optional[str] = None,
        risk_score: float = 0.0,
        ear_value: Optional[float] = None,
        perclos_pct: Optional[float] = None,
        pose_zone: Optional[str] = None,
        reasons: Optional[List[str]] = None,
        engine_type: str = "PYTHON",
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Idempotently persist an alert transition keyed by unique `event_id`.
        Duplicate `event_id` calls are ignored without duplicating rows or inflating
        session warning/critical counters.
        Returns `(inserted: bool, alert_record: dict)`.
        """
        now = _utc_now_iso()
        sev_upper = severity.upper()
        reasons_list = list(reasons) if reasons else []
        reasons_json = json.dumps(reasons_list)

        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT OR IGNORE INTO local_alerts (
                    event_id, session_id, session_code, driver_id, engine_type,
                    severity, previous_state, new_state, risk_score,
                    ear_value, perclos_pct, pose_zone, reasons_json,
                    timestamp_ms, created_at, sync_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'LOCAL_ONLY')
                """,
                (
                    event_id,
                    session_id,
                    session_code,
                    driver_id,
                    engine_type,
                    sev_upper,
                    previous_state,
                    new_state,
                    float(risk_score),
                    float(ear_value) if ear_value is not None else None,
                    float(perclos_pct) if perclos_pct is not None else None,
                    pose_zone,
                    reasons_json,
                    int(timestamp_ms),
                    now,
                ),
            )
            inserted = cur.rowcount > 0

            if inserted:
                if sev_upper == "CRITICAL":
                    cur.execute(
                        "UPDATE local_sessions SET critical_alerts_count = critical_alerts_count + 1 WHERE session_id = ?",
                        (session_id,),
                    )
                elif sev_upper == "WARNING":
                    cur.execute(
                        "UPDATE local_sessions SET warning_alerts_count = warning_alerts_count + 1 WHERE session_id = ?",
                        (session_id,),
                    )

                outbox_payload = json.dumps(
                    {
                        "eventId": event_id,
                        "clientSessionId": session_id,
                        "sessionCode": session_code,
                        "driverId": driver_id,
                        "engineType": engine_type,
                        "severity": sev_upper,
                        "newState": new_state,
                        "previousState": previous_state,
                        "riskScore": float(risk_score),
                        "earValue": ear_value,
                        "perclosPct": perclos_pct,
                        "poseZone": pose_zone,
                        "reasons": reasons_list,
                        "timestampMs": int(timestamp_ms),
                    }
                )
                cur.execute(
                    """
                    INSERT OR IGNORE INTO local_sync_outbox (
                        entity_type, entity_id, payload_json, status, created_at, updated_at
                    ) VALUES ('ALERT', ?, ?, 'PENDING', ?, ?)
                    """,
                    (event_id, outbox_payload, now, now),
                )

            self._conn.commit()
            row = cur.execute(
                "SELECT * FROM local_alerts WHERE event_id = ?", (event_id,)
            ).fetchone()
            return inserted, self._row_to_alert(row) if row else {}

    def end_session(
        self,
        session_id: str,
        *,
        duration_seconds: int = 0,
        total_frames: int = 0,
        normal_frames_count: int = 0,
        average_fps: float = 0.0,
        average_latency_ms: float = 0.0,
        average_perclos_pct: float = 0.0,
        max_perclos_pct: float = 0.0,
        total_blinks: int = 0,
        average_blink_rate: float = 0.0,
        mirror_suppressions: int = 0,
        status_value: str = "COMPLETED",
    ) -> Optional[Dict[str, Any]]:
        """Idempotently finalize an active session and enqueue SESSION_END in outbox."""
        now = _utc_now_iso()
        with self._lock:
            cur = self._conn.cursor()
            existing = cur.execute(
                "SELECT * FROM local_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if existing is None:
                return None

            if existing["status"] == "COMPLETED":
                return dict(existing)

            cur.execute(
                """
                UPDATE local_sessions
                SET status = ?,
                    ended_at = ?,
                    duration_seconds = ?,
                    total_frames = ?,
                    normal_frames_count = ?,
                    average_fps = ?,
                    average_latency_ms = ?,
                    average_perclos_pct = ?,
                    max_perclos_pct = ?,
                    total_blinks = ?,
                    average_blink_rate = ?,
                    mirror_suppressions = ?
                WHERE session_id = ?
                """,
                (
                    status_value,
                    now,
                    int(max(0, duration_seconds)),
                    int(max(0, total_frames)),
                    int(max(0, normal_frames_count)),
                    float(round(average_fps, 2)),
                    float(round(average_latency_ms, 2)),
                    float(round(average_perclos_pct, 2)),
                    float(round(max_perclos_pct, 2)),
                    int(max(0, total_blinks)),
                    float(round(average_blink_rate, 2)),
                    int(max(0, mirror_suppressions)),
                    session_id,
                ),
            )

            updated = cur.execute(
                "SELECT * FROM local_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            updated_dict = dict(updated) if updated else {}

            outbox_payload = json.dumps(
                {
                    "clientSessionId": session_id,
                    "sessionCode": updated_dict.get("session_code"),
                    "finalizeToken": f"fin_{updated_dict.get('session_code')}_{session_id[:8]}",
                    "durationSeconds": updated_dict.get("duration_seconds", 0),
                    "totalFrames": updated_dict.get("total_frames", 0),
                    "normalFramesCount": updated_dict.get("normal_frames_count", 0),
                    "averageFps": updated_dict.get("average_fps", 0.0),
                    "averageLatencyMs": updated_dict.get("average_latency_ms", 0.0),
                    "averagePerclosPct": updated_dict.get("average_perclos_pct", 0.0),
                    "maxPerclosPct": updated_dict.get("max_perclos_pct", 0.0),
                    "totalBlinks": updated_dict.get("total_blinks", 0),
                    "averageBlinkRate": updated_dict.get("average_blink_rate", 0.0),
                    "warningAlertsCount": updated_dict.get("warning_alerts_count", 0),
                    "criticalAlertsCount": updated_dict.get("critical_alerts_count", 0),
                    "mirrorSuppressions": updated_dict.get("mirror_suppressions", 0),
                    "endedAt": now,
                }
            )
            cur.execute(
                """
                INSERT OR IGNORE INTO local_sync_outbox (
                    entity_type, entity_id, payload_json, status, created_at, updated_at
                ) VALUES ('SESSION_END', ?, ?, 'PENDING', ?, ?)
                """,
                (session_id, outbox_payload, now, now),
            )
            self._conn.commit()
            return updated_dict

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM local_sessions WHERE session_id = ? OR session_code = ?",
                (session_id, session_id),
            ).fetchone()
            return dict(row) if row else None

    def list_sessions(
        self,
        driver_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        safe_limit = max(1, min(500, int(limit)))
        with self._lock:
            if driver_id:
                rows = self._conn.execute(
                    """
                    SELECT * FROM local_sessions
                    WHERE driver_id = ?
                    ORDER BY started_at DESC
                    LIMIT ?
                    """,
                    (driver_id, safe_limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """
                    SELECT * FROM local_sessions
                    ORDER BY started_at DESC
                    LIMIT ?
                    """,
                    (safe_limit,),
                ).fetchall()
            return [dict(r) for r in rows]

    def list_session_alerts(
        self,
        session_id: str,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        safe_limit = max(1, min(1000, int(limit)))
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM local_alerts
                WHERE session_id = ? OR session_code = ?
                ORDER BY timestamp_ms ASC
                LIMIT ?
                """,
                (session_id, session_id, safe_limit),
            ).fetchall()
            return [self._row_to_alert(r) for r in rows]

    def list_pending_outbox(self, limit: int = 100) -> List[Dict[str, Any]]:
        safe_limit = max(1, min(500, int(limit)))
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM local_sync_outbox
                WHERE status = 'PENDING'
                ORDER BY id ASC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()
            results: List[Dict[str, Any]] = []
            for r in rows:
                item = dict(r)
                try:
                    item["payload"] = json.loads(item.pop("payload_json", "{}"))
                except json.JSONDecodeError:
                    item["payload"] = {}
                results.append(item)
            return results

    def mark_outbox_synced(self, entity_type: str, entity_id: str) -> bool:
        now = _utc_now_iso()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                UPDATE local_sync_outbox
                SET status = 'SYNCED', updated_at = ?
                WHERE entity_type = ? AND entity_id = ?
                """,
                (now, entity_type, entity_id),
            )
            updated = cur.rowcount > 0
            if entity_type == "SESSION_END":
                cur.execute(
                    "UPDATE local_sessions SET sync_status = 'SYNCED' WHERE session_id = ?",
                    (entity_id,),
                )
            elif entity_type == "ALERT":
                cur.execute(
                    "UPDATE local_alerts SET sync_status = 'SYNCED' WHERE event_id = ?",
                    (entity_id,),
                )
            self._conn.commit()
            return updated

    @staticmethod
    def _row_to_alert(row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        raw_reasons = d.pop("reasons_json", "[]")
        try:
            d["reasons"] = json.loads(raw_reasons)
        except (TypeError, json.JSONDecodeError):
            d["reasons"] = []
        return d

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass


local_db = LocalDB()
