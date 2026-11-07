"""
Session Manager Service — In-Memory Registry of Active Driver AI Sessions.
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional

from backend.app.ai.cascade_classifier import ExperimentMode
from backend.app.ai.landmarker import ensure_face_landmarker_model
from backend.app.ai.pipeline import SessionAIEngine
from backend.app.services.profile_store import profile_store


class SessionManager:
    """Thread-safe registry of active driving monitoring sessions with per-session landmarker isolation."""

    def __init__(self) -> None:
        self._sessions: Dict[str, SessionAIEngine] = {}
        self._lock = asyncio.Lock()

    def preload_shared_models(self) -> None:
        """Verify and download MediaPipe model asset bundle if needed during application lifespan startup."""
        ensure_face_landmarker_model()

    async def get_or_create_session(
        self,
        session_id: str,
        driver_id: str = "driver_default",
        experiment_mode: Optional[str] = None,
    ) -> SessionAIEngine:
        """Retrieve an existing session or instantiate a new SessionAIEngine with its own isolated landmarker."""
        async with self._lock:
            if session_id in self._sessions:
                return self._sessions[session_id]

            # Parse experiment mode
            exp_mode = ExperimentMode.C_CONFIDENCE_GATED_CASCADE
            if experiment_mode:
                for em in ExperimentMode:
                    if em.value == experiment_mode or em.name == experiment_mode:
                        exp_mode = em
                        break

            # Check if this driver has a pre-calibrated profile on disk
            stored_profile_dto = profile_store.get_profile(driver_id)
            engine = SessionAIEngine(
                session_id=session_id,
                driver_id=driver_id,
                experiment_mode=exp_mode,
            )
            if stored_profile_dto is not None:
                engine.load_profile_dto(stored_profile_dto)

            self._sessions[session_id] = engine
            return engine

    def get_session(self, session_id: str) -> Optional[SessionAIEngine]:
        """Fetch active session by session_id without creating a new one."""
        return self._sessions.get(session_id)

    async def terminate_session(self, session_id: str) -> Optional[Dict[str, object]]:
        """Remove an active session, release its landmarker resources, and return final summary diagnostics."""
        async with self._lock:
            engine = self._sessions.pop(session_id, None)
            if engine is not None:
                # If profile was calibrated, auto-persist to profile store
                if engine.profile.calibration_mode != "UNCALIBRATED_DEFAULT":
                    profile_store.save_profile(engine.export_profile_dto())
                summary = engine.get_session_summary()
                engine.close()
                return summary
            return None

    def list_active_sessions(self) -> List[Dict[str, object]]:
        """List active session metadata."""
        summaries = []
        for s_id, engine in self._sessions.items():
            summaries.append(
                {
                    "session_id": s_id,
                    "driver_id": engine.driver_id,
                    "frame_count": engine.frame_index,
                    "is_calibrating": engine.is_calibrating,
                    "is_calibrated": engine.profile.calibration_mode != "UNCALIBRATED_DEFAULT",
                    "experiment_mode": engine.experiment_mode.value,
                    "fps": round(engine.fps_ema, 1),
                }
            )
        return summaries


session_manager = SessionManager()
