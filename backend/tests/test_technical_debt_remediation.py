"""
Technical Debt Remediation Verification Suite (Phases 9.1, 9.2, 9.3).

Verifies:
- Phase 9.1: Single canonical PoseZone across head_pose, geometry, telemetry schema, and pipeline;
  no redundant `_map_pose_zone_to_dto` translator; invalid poses strictly resolve to UNKNOWN (never CENTER).
- Phase 9.2: Production runtime (`backend.app.main`, `SessionAIEngine`, `LocalMonitoringService`,
  `DrowsinessEngine`, `AlertStateManager`) does NOT import legacy research modules from
  `backend.app.ai.legacy` or the 6 legacy compatibility shims.
- Phase 9.3: Desktop audio (`sounddevice`) is explicitly optional with graceful `NullAudioBackend`
  headless fallback, clear status reporting, and `/health/readiness` remains READY without it.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.ai import pipeline as pipeline_module
from backend.app.ai.geometry import PoseZone as LegacyGeometryPoseZone
from backend.app.alerts.audio_warning_engine import (
    AudioWarningEngine,
    NullAudioBackend,
    check_sounddevice_availability,
)
from backend.app.features.head_pose import (
    PoseZone as CanonicalPoseZone,
    classify_pose_zone,
    estimate_head_pose,
)
from backend.app.main import app
from backend.app.schemas.telemetry import PoseZoneEnum


REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Phase 9.1 — Unified PoseZone
# ---------------------------------------------------------------------------

def test_phase_9_1_single_canonical_pose_zone_enum():
    assert PoseZoneEnum is CanonicalPoseZone
    assert LegacyGeometryPoseZone is CanonicalPoseZone
    assert not hasattr(pipeline_module, "_map_pose_zone_to_dto"), (
        "Redundant _map_pose_zone_to_dto translator should be removed after PoseZone unification"
    )


def test_phase_9_1_invalid_pose_never_becomes_center():
    assert classify_pose_zone(pitch_deg=float("nan"), yaw_deg=0.0) == CanonicalPoseZone.UNKNOWN
    assert classify_pose_zone(pitch_deg=0.0, yaw_deg=float("inf")) == CanonicalPoseZone.UNKNOWN

    # Empty / missing landmarks must return UNKNOWN, never CENTER
    res_empty = estimate_head_pose(landmarks=[], frame_width=640, frame_height=480)
    assert res_empty.zone == CanonicalPoseZone.UNKNOWN
    assert res_empty.is_valid is False


# ---------------------------------------------------------------------------
# Phase 9.2 — Production vs Legacy Research Module Boundary
# ---------------------------------------------------------------------------

LEGACY_MODULE_NAMES = {
    "backend.app.ai.geometry",
    "backend.app.ai.landmarker",
    "backend.app.ai.cascade_classifier",
    "backend.app.ai.blink_kinematics",
    "backend.app.ai.gaze_attention",
    "backend.app.ai.risk_engine",
}

PRODUCTION_FILES = [
    REPO_ROOT / "backend" / "app" / "main.py",
    REPO_ROOT / "backend" / "app" / "ai" / "pipeline.py",
    REPO_ROOT / "backend" / "app" / "ai" / "pdbp_calibration.py",
    REPO_ROOT / "backend" / "app" / "ai" / "profile_adapter.py",
    REPO_ROOT / "backend" / "app" / "services" / "local_monitoring_service.py",
    REPO_ROOT / "backend" / "app" / "services" / "session_manager.py",
    REPO_ROOT / "backend" / "app" / "services" / "livekit_worker.py",
    REPO_ROOT / "backend" / "app" / "api" / "ws_telemetry.py",
    REPO_ROOT / "backend" / "app" / "api" / "routes" / "sessions.py",
    REPO_ROOT / "backend" / "app" / "api" / "routes" / "livekit_token.py",
    REPO_ROOT / "backend" / "app" / "api" / "routes" / "calibration.py",
    REPO_ROOT / "backend" / "app" / "api" / "routes" / "profiles.py",
]


def test_phase_9_2_production_modules_have_zero_ast_imports_of_legacy_research():
    violations = []
    for path in PRODUCTION_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module in LEGACY_MODULE_NAMES or node.module.startswith("backend.app.ai.legacy"):
                    violations.append(f"{path.name}:{node.lineno} imports {node.module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in LEGACY_MODULE_NAMES or alias.name.startswith("backend.app.ai.legacy"):
                        violations.append(f"{path.name}:{node.lineno} imports {alias.name}")
    assert not violations, f"Production modules still import legacy research code: {violations}"


def test_phase_9_2_fresh_process_startup_loads_zero_legacy_modules():
    code = (
        "import sys; import backend.app.main; "
        "from backend.app.ai.pipeline import SessionAIEngine; "
        "eng = SessionAIEngine(session_id='boundary_test', driver_id='d1'); "
        "eng.close(); "
        "bad = [m for m in sys.modules if m.startswith('backend.app.ai.legacy') or m in {"
        "'backend.app.ai.geometry','backend.app.ai.landmarker','backend.app.ai.cascade_classifier',"
        "'backend.app.ai.blink_kinematics','backend.app.ai.gaze_attention','backend.app.ai.risk_engine'}]; "
        "assert not bad, f'Legacy modules loaded: {bad}'; print('OK')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"Subprocess failed:\nSTDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
    assert "OK" in proc.stdout


# ---------------------------------------------------------------------------
# Phase 9.3 — Optional Desktop Audio Headless Fallback
# ---------------------------------------------------------------------------

def test_phase_9_3_audio_engine_falls_back_gracefully_when_sounddevice_missing():
    with patch(
        "backend.app.alerts.audio_warning_engine.check_sounddevice_availability",
        return_value=(False, "Optional desktop audio dependency 'sounddevice' is not installed."),
    ):
        engine = AudioWarningEngine()
        try:
            assert isinstance(engine.backend, NullAudioBackend)
            assert engine.audio_available is False
            assert engine.audio_unavailable_reason is not None
            assert "sounddevice" in engine.audio_unavailable_reason
            status = engine.get_audio_status()
            assert status["available"] is False
            assert status["backend"] == "NullAudioBackend"
        finally:
            engine.shutdown()


def test_phase_9_3_readiness_stays_ready_without_desktop_sounddevice():
    with patch(
        "backend.app.alerts.audio_warning_engine.check_sounddevice_availability",
        return_value=(False, "sounddevice not installed"),
    ):
        client = TestClient(app)
        res = client.get("/health/readiness")
        assert res.status_code == 200
        body = res.json()
        assert body["ready"] is True
        assert body["components"]["desktop_audio"]["optional"] is True
        assert body["components"]["desktop_audio"]["available"] is False
        assert body["components"]["desktop_audio"]["web_client_audio"] == "BROWSER_WEB_AUDIO_API"
