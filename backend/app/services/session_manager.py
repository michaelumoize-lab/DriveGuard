"""
Session Manager Service — In-Memory Registry of Active Driver AI Sessions.
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional

from backend.app.ai.cascade_classifier import ExperimentMode
from backend.app.ai.landmarker import ensure_face_landmarker_model
from backend.app.ai.pipeline import SessionAIEngine
from backend.app.schemas.profile import DriverProfileDTO
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
        profile_dto: Optional[DriverProfileDTO] = None,
        owner_id: Optional[str] = None,
    ) -> SessionAIEngine:
        """Retrieve an existing session or instantiate a new SessionAIEngine with its own isolated landmarker."""
        effective_owner = owner_id or driver_id
        async with self._lock:
            if session_id in self._sessions:
                existing = self._sessions[session_id]
                existing_owner = getattr(existing, "owner_id", existing.driver_id)
                if owner_id and existing_owner != owner_id:
                    raise PermissionError(
                        f"Session '{session_id}' belongs to user '{existing_owner}', not '{owner_id}'."
                    )
                return existing

            # Parse experiment mode
            exp_mode = ExperimentMode.C_CONFIDENCE_GATED_CASCADE
            if experiment_mode:
                for em in ExperimentMode:
                    if em.value == experiment_mode or em.name == experiment_mode:
                        exp_mode = em
                        break

            # Resolve profile: prioritize explicitly provided profile payload, fallback to local store
            target_profile_dto = profile_dto or profile_store.get_profile(driver_id)
            engine: Optional[SessionAIEngine] = None
            registered = False
            try:
                construction_task = asyncio.create_task(
                    asyncio.to_thread(
                        SessionAIEngine,
                        session_id=session_id,
                        driver_id=driver_id,
                        experiment_mode=exp_mode,
                        owner_id=effective_owner,
                    )
                )
                try:
                    engine = await construction_task
                except BaseException:
                    if construction_task.done() and not construction_task.cancelled():
                        try:
                            engine = construction_task.result()
                        except Exception:
                            engine = None
                    raise

                if target_profile_dto is not None:
                    engine.load_profile_dto(target_profile_dto)
                    # Cache in local store for resilience across reconnects
                    try:
                        profile_store.save_profile(target_profile_dto)
                    except Exception:
                        pass

                self._sessions[session_id] = engine
                registered = True
                return engine
            finally:
                if engine is not None and not registered:
                    try:
                        await asyncio.shield(asyncio.to_thread(engine.close))
                    except Exception:
                        pass

    def get_session(self, session_id: str) -> Optional[SessionAIEngine]:
        """Fetch active session by session_id without creating a new one."""
        return self._sessions.get(session_id)

    async def terminate_session(self, session_id: str) -> Optional[Dict[str, object]]:
        """Remove an active session, release its landmarker resources, and return final summary diagnostics."""
        async with self._lock:
            engine = self._sessions.pop(session_id, None)
            if engine is not None:
                try:
                    # If profile was calibrated, auto-persist to profile store
                    if engine.profile.calibration_mode != "UNCALIBRATED_DEFAULT":
                        profile_store.save_profile(engine.export_profile_dto())
                    summary = engine.get_session_summary()
                    return summary
                finally:
                    await asyncio.to_thread(engine.close)
            return None

    def list_active_sessions(self) -> List[Dict[str, object]]:
        """List active session metadata."""
        summaries = []
        for s_id, engine in self._sessions.items():
            summaries.append(
                {
                    "session_id": s_id,
                    "driver_id": engine.driver_id,
                    "owner_id": getattr(engine, "owner_id", engine.driver_id),
                    "frame_count": engine.frame_index,
                    "is_calibrating": engine.is_calibrating,
                    "is_calibrated": engine.profile.calibration_mode != "UNCALIBRATED_DEFAULT",
                    "experiment_mode": engine.experiment_mode.value,
                    "fps": round(engine.fps_ema, 1),
                }
            )
        return summaries


session_manager = SessionManager()
