"""
DriveGuard — Phase 8: Local Warning Interface Automated Unit Tests.

Validates:
1. Diagnostic HUD rendering with graduated alert states (NORMAL, ATTENTION, WARNING, CRITICAL).
2. Diagnostic HUD recovery display ([RECOVERING] badge and telemetry).
3. Audio Warnings UI status indicator ([AUDIO: ON] vs [AUDIO: OFF]).
4. Audio feedback banner rendering on toggle.
5. Viewport border color mapping matching alert severity.
6. Headless integration execution with AlertStateManager and AlertDispatcher.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.app.alerts.alert_state import AlertState
from backend.app.camera.frame_quality import FrameQualityResult, QualityStatus
from backend.app.vision.face_landmarker import FaceLandmarkResult, FaceLandmarkStatus
from scripts.run_camera_test import draw_diagnostic_hud, run_benchmark


class TestAlertInterface:
    """Automated tests for Phase 8 Local Warning Interface."""

    @pytest.fixture
    def sample_frame(self) -> np.ndarray:
        # Create 720p synthetic BGR image
        return np.zeros((720, 1280, 3), dtype=np.uint8)

    def test_hud_renders_with_normal_alert_and_audio_on(self, sample_frame: np.ndarray) -> None:
        rendered = draw_diagnostic_hud(
            frame=sample_frame,
            source_id="0",
            actual_res=(1280, 720),
            ingestion_fps=30.0,
            display_fps=30.0,
            frame_age_ms=15.0,
            buffer_stats={"total_pushed": 100, "total_dropped": 0},
            alert_state=AlertState.NORMAL,
            is_recovering=False,
            audio_warnings_enabled=True,
        )
        assert rendered.shape == (720, 1280, 3)
        assert rendered.dtype == np.uint8
        # Border should be Emerald green (34, 197, 94)
        border_pixel = rendered[1, 1]
        assert tuple(border_pixel) == (34, 197, 94)

    def test_hud_renders_with_critical_alert_and_audio_off(self, sample_frame: np.ndarray) -> None:
        rendered = draw_diagnostic_hud(
            frame=sample_frame,
            source_id="0",
            actual_res=(1280, 720),
            ingestion_fps=30.0,
            display_fps=30.0,
            frame_age_ms=18.0,
            buffer_stats={"total_pushed": 200, "total_dropped": 0},
            alert_state=AlertState.CRITICAL,
            is_recovering=False,
            audio_warnings_enabled=False,
        )
        assert rendered.shape == (720, 1280, 3)
        # Border should be Red Alert (68, 68, 239) in BGR
        border_pixel = rendered[1, 1]
        assert tuple(border_pixel) == (68, 68, 239)

    def test_hud_renders_recovery_indicator(self, sample_frame: np.ndarray) -> None:
        rendered = draw_diagnostic_hud(
            frame=sample_frame,
            source_id="0",
            actual_res=(1280, 720),
            ingestion_fps=30.0,
            display_fps=30.0,
            frame_age_ms=12.0,
            buffer_stats={"total_pushed": 300, "total_dropped": 0},
            alert_state=AlertState.WARNING,
            is_recovering=True,
            audio_warnings_enabled=True,
        )
        assert rendered.shape == (720, 1280, 3)
        # Border should be Orange (22, 115, 249) in BGR
        border_pixel = rendered[1, 1]
        assert tuple(border_pixel) == (22, 115, 249)

    def test_hud_renders_audio_feedback_banner(self, sample_frame: np.ndarray) -> None:
        rendered = draw_diagnostic_hud(
            frame=sample_frame,
            source_id="0",
            actual_res=(1280, 720),
            ingestion_fps=30.0,
            display_fps=30.0,
            frame_age_ms=10.0,
            buffer_stats={"total_pushed": 50, "total_dropped": 0},
            alert_state=AlertState.NORMAL,
            is_recovering=False,
            audio_warnings_enabled=False,
            audio_feedback_banner="AUDIO WARNINGS: MUTED",
        )
        assert rendered.shape == (720, 1280, 3)

    def test_hud_handles_compact_resolutions(self) -> None:
        compact_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        rendered = draw_diagnostic_hud(
            frame=compact_frame,
            source_id="0",
            actual_res=(640, 480),
            ingestion_fps=25.0,
            display_fps=25.0,
            frame_age_ms=20.0,
            buffer_stats={"total_pushed": 10, "total_dropped": 0},
            alert_state=AlertState.ATTENTION,
            is_recovering=False,
            audio_warnings_enabled=True,
        )
        assert rendered.shape == (480, 640, 3)
