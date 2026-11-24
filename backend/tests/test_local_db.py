"""
DriveGuard — Phase 4A Tests for SQLite LocalDB & Versioned Python ProfileStore.

Verifies:
- SQLite WAL sessions & idempotent alerts persist cleanly across LocalDB restarts
- Duplicate event_id does not duplicate rows or inflate warning/critical alert counters
- Outbox preserves causal order (SESSION_START -> ALERT -> SESSION_END) and sync state updates
- ProfileStore computes deterministic content_hash, increments profile_version, archives
  superseded JSON files into history/, and reconciles PYTHON profiles while rejecting BROWSER
"""

from __future__ import annotations

from pathlib import Path
import pytest

from backend.app.schemas.profile import DriverProfileDTO, ZoneBaselineDTO
from backend.app.services.local_db import LocalDB
from backend.app.services.profile_store import ProfileStore, compute_profile_content_hash


def _make_python_profile(
    driver_id: str = "driver_local_test",
    center_ear: float = 0.31,
    version: int = 1,
    calibrated_at: str = "2026-11-24T12:00:00Z",
    engine_type: str = "PYTHON",
) -> DriverProfileDTO:
    return DriverProfileDTO(
        driver_id=driver_id,
        engine_type=engine_type,
        profile_version=version,
        is_calibrated=True,
        calibration_mode="FOUR_ZONE",
        calibrated_at=calibrated_at,
        center_ear_median=center_ear,
        zone_baselines={
            "CENTER": ZoneBaselineDTO(
                zone="CENTER",
                ear_median=center_ear,
                ear_mad_scaled=0.015,
                dynamic_closure_threshold=0.20,
                pitch_median_deg=0.0,
                yaw_median_deg=0.0,
                sample_count=40,
            )
        },
    )


def test_local_db_persists_across_restart_and_enforces_alert_idempotency(tmp_path: Path) -> None:
    db_file = tmp_path / "driveguard_local.db"
    db = LocalDB(db_path=db_file)

    session = db.create_session(
        driver_id="driver_sqlite",
        camera_source="0",
        engine_type="PYTHON",
        profile_uuid="uuid-1",
        profile_version=2,
    )
    sid = session["session_id"]
    scode = session["session_code"]
    assert scode.startswith("DRV-PY-")
    assert session["status"] == "ACTIVE"

    inserted1, a1 = db.record_alert(
        event_id="evt_idemp_001",
        session_id=sid,
        session_code=scode,
        driver_id="driver_sqlite",
        severity="CRITICAL",
        previous_state="WARNING",
        new_state="CRITICAL",
        risk_score=0.92,
        ear_value=0.14,
        perclos_pct=27.5,
        pose_zone="CENTER",
        reasons=["SUSTAINED_EYE_CLOSURE"],
        timestamp_ms=1732400000000,
    )
    assert inserted1 is True
    assert a1["event_id"] == "evt_idemp_001"
    assert a1["reasons"] == ["SUSTAINED_EYE_CLOSURE"]

    # Duplicate event_id MUST NOT duplicate rows or inflate critical_alerts_count
    inserted2, a2 = db.record_alert(
        event_id="evt_idemp_001",
        session_id=sid,
        session_code=scode,
        driver_id="driver_sqlite",
        severity="CRITICAL",
        previous_state="WARNING",
        new_state="CRITICAL",
        risk_score=0.92,
        ear_value=0.14,
        perclos_pct=27.5,
        pose_zone="CENTER",
        reasons=["SUSTAINED_EYE_CLOSURE"],
        timestamp_ms=1732400000000,
    )
    assert inserted2 is False
    assert a2["event_id"] == "evt_idemp_001"

    ended = db.end_session(
        sid,
        duration_seconds=35,
        total_frames=520,
        normal_frames_count=490,
        average_fps=14.8,
        average_latency_ms=11.2,
        average_perclos_pct=9.4,
        max_perclos_pct=27.5,
        total_blinks=11,
        average_blink_rate=18.9,
        mirror_suppressions=2,
    )
    assert ended is not None
    assert ended["status"] == "COMPLETED"
    assert ended["critical_alerts_count"] == 1
    assert ended["warning_alerts_count"] == 0
    db.close()

    # Reopen fresh LocalDB instance against same SQLite file to prove crash/restart durability
    db_reopened = LocalDB(db_path=db_file)
    try:
        sessions = db_reopened.list_sessions(driver_id="driver_sqlite")
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == sid
        assert sessions[0]["duration_seconds"] == 35
        assert sessions[0]["critical_alerts_count"] == 1

        alerts = db_reopened.list_session_alerts(sid)
        assert len(alerts) == 1
        assert alerts[0]["event_id"] == "evt_idemp_001"

        outbox = db_reopened.list_pending_outbox()
        assert [o["entity_type"] for o in outbox] == ["SESSION_START", "ALERT", "SESSION_END"]

        assert db_reopened.mark_outbox_synced("ALERT", "evt_idemp_001") is True
        assert db_reopened.mark_outbox_synced("SESSION_END", sid) is True
        reloaded_session = db_reopened.get_session(sid)
        assert reloaded_session is not None
        assert reloaded_session["sync_status"] == "SYNCED"
    finally:
        db_reopened.close()


def test_profile_store_versioning_archival_and_reconciliation(tmp_path: Path) -> None:
    store = ProfileStore(storage_dir=tmp_path / "profiles")

    p1 = _make_python_profile("driver_ver", center_ear=0.31, version=1, calibrated_at="2026-11-24T10:00:00Z")
    saved1 = store.save_profile(p1)
    assert saved1.engine_type == "PYTHON"
    assert saved1.profile_uuid is not None
    assert saved1.content_hash == compute_profile_content_hash(p1)
    assert saved1.profile_version == 1

    # Save modified profile -> version auto-increments and previous JSON archived in history/
    p2 = _make_python_profile("driver_ver", center_ear=0.29, version=1, calibrated_at="2026-11-24T11:00:00Z")
    saved2 = store.save_profile(p2)
    assert saved2.profile_version == 2
    assert saved2.content_hash != saved1.content_hash
    history_files = list((tmp_path / "profiles" / "history").glob("driver_ver_v1_*.json"))
    assert len(history_files) == 1

    # Reject non-PYTHON reconciliation strictly
    browser_profile = _make_python_profile("driver_ver", center_ear=0.33, version=5, engine_type="BROWSER")
    with pytest.raises(ValueError, match="non-PYTHON"):
        store.reconcile_profile("driver_ver", browser_profile)

    # Reconcile newer remote PYTHON profile -> adopts remote and archives superseded local
    remote_newer = _make_python_profile("driver_ver", center_ear=0.33, version=4, calibrated_at="2026-11-24T13:00:00Z")
    rec = store.reconcile_profile("driver_ver", remote_newer)
    assert rec["action"] == "CONFLICT_RESOLVED_REMOTE_NEWER"
    assert rec["archived_previous"] is True
    assert rec["profile"].center_ear_median == pytest.approx(0.33)
    assert rec["profile"].sync_state == "SYNCED"

    # Reconciling identical profile again -> UNCHANGED
    rec_same = store.reconcile_profile("driver_ver", remote_newer)
    assert rec_same["action"] == "UNCHANGED"
