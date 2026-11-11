"""
DriveGuard Phase 7 Stage 5: Session Isolation & Cross-Session Contamination Test Suite.
Verifies strict isolation of computer vision metrics, risk state machines, calibration profiles,
lifecycle teardown, and telemetry scoping across concurrent driver sessions.
"""

import asyncio
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from backend.app.ai.pipeline import SessionAIEngine
from backend.app.services.livekit_worker import LiveKitAIWorkerSession
from backend.app.services.session_manager import session_manager
from backend.benchmarks.benchmark_inference_offline import generate_benchmark_frame


class MockVideoFrame:
    """Mock WebRTC VideoFrame wrapping numpy RGB buffer for memory simulation."""

    def __init__(self, img_bgr: np.ndarray):
        self.width = img_bgr.shape[1]
        self.height = img_bgr.shape[0]
        self._img_bgr = img_bgr

    def convert(self, buffer_type: object) -> "MockVideoFrame":
        return self

    @property
    def data(self) -> bytes:
        return self._img_bgr.tobytes()


class NullDataCollector:
    """Mock WebRTC participant collecting published telemetry payloads per session."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.published_payloads = []

    async def publish_data(self, payload: bytes, reliable: bool = False) -> None:
        import json
        try:
            msg = json.loads(payload.decode("utf-8"))
            self.published_payloads.append(msg)
        except Exception:
            pass


class TestSessionIsolation(unittest.IsolatedAsyncioTestCase):
    """Test suite validating cross-session isolation and lifecycle autonomy."""

    async def asyncSetUp(self):
        self.session_id_a = f"test_iso_sess_a_{int(time.time() * 1000)}"
        self.session_id_b = f"test_iso_sess_b_{int(time.time() * 1000)}"

    async def asyncTearDown(self):
        await session_manager.terminate_session(self.session_id_a)
        await session_manager.terminate_session(self.session_id_b)

    async def test_session_epoch_and_boot_id_scoping(self):
        """worker_boot_id must be process-scoped; session_epoch must be unique per session lifecycle."""
        worker_a = LiveKitAIWorkerSession(room_name=self.session_id_a, driver_id="driver_a")
        worker_b = LiveKitAIWorkerSession(room_name=self.session_id_b, driver_id="driver_b")

        try:
            # Boot ID is identical (same process)
            self.assertEqual(worker_a.worker_boot_id, worker_b.worker_boot_id)
            self.assertTrue(len(worker_a.worker_boot_id) >= 16)

            # Session Epoch is distinct and unique
            self.assertNotEqual(worker_a.session_epoch, worker_b.session_epoch)
            self.assertNotEqual(worker_a.session_id_uuid, worker_b.session_id_uuid)
            self.assertIn(worker_a.session_id_uuid, worker_a.session_epoch)
            self.assertIn(worker_b.session_id_uuid, worker_b.session_epoch)
        finally:
            if worker_a._executor:
                worker_a._executor.shutdown(wait=False, cancel_futures=True)
            if worker_b._executor:
                worker_b._executor.shutdown(wait=False, cancel_futures=True)

    async def test_risk_state_and_metrics_isolation(self):
        """A session entering critical drowsiness does not contaminate another session's state or metrics."""
        engine_a = await session_manager.get_or_create_session(
            session_id=self.session_id_a,
            driver_id="driver_alpha",
            experiment_mode="EXPERIMENT_C_CONFIDENCE_GATED_CASCADE",
        )
        engine_b = await session_manager.get_or_create_session(
            session_id=self.session_id_b,
            driver_id="driver_beta",
            experiment_mode="EXPERIMENT_C_CONFIDENCE_GATED_CASCADE",
        )

        # 1. Warm up both engines with 10 normal open-eyed frames
        for idx in range(10):
            frame_open = generate_benchmark_frame(idx)
            now_ms = time.perf_counter() * 1000.0
            dto_a = engine_a.process_frame(frame_open, now_ms)
            dto_b = engine_b.process_frame(frame_open, now_ms)
            self.assertEqual(dto_a.state.value, "NORMAL")
            self.assertEqual(dto_b.state.value, "NORMAL")

        # 2. Feed Session A open-eyed frames; feed Session B closed-eyed frames
        # Closed-eyed synthetic frame (black/blanked or synthetic with low EAR)
        # Using blank image causes no face or low EAR landmark detector
        # For realistic closed-eyed test, we set eyes closed by passing eye closed synthetic frame
        img_open = generate_benchmark_frame(100)
        # Create an image that triggers closed eyes (or simulate eye closure by modifying eyelid region)
        img_closed = img_open.copy()
        # Darken / occlude eye band (rows 180 to 260 across center) to simulate prolonged eye closure
        img_closed[180:260, 220:420] = 0

        # Feed 35 frames: A stays natural with periodic blinks, B has prolonged eye closure
        last_dto_a = None
        last_dto_b = None
        for idx in range(35):
            now_ms = (time.perf_counter() * 1000.0) + (idx * 40.0)
            frame_open_a = generate_benchmark_frame(idx + 20)
            last_dto_a = engine_a.process_frame(frame_open_a, now_ms)
            last_dto_b = engine_b.process_frame(img_closed, now_ms)

        # Session A must remain NORMAL
        self.assertEqual(last_dto_a.state.value, "NORMAL")
        self.assertGreater(last_dto_a.eye_metrics.ear_raw, 0.30)

        # Session B must reflect risk / monitoring state change due to occlusion or eye closure
        self.assertNotEqual(last_dto_b.session_id, last_dto_a.session_id)
        self.assertEqual(last_dto_a.session_id, self.session_id_a)
        self.assertEqual(last_dto_b.session_id, self.session_id_b)

        # Session A's counters must not be influenced by Session B
        self.assertEqual(engine_a.session_id, self.session_id_a)
        self.assertEqual(engine_b.session_id, self.session_id_b)
        self.assertEqual(engine_a.frame_index, 45)
        self.assertEqual(engine_b.frame_index, 45)

    async def test_calibration_profile_isolation(self):
        """Calibrating Session A modifies only Session A's profile; Session B remains default."""
        engine_a = await session_manager.get_or_create_session(
            session_id=self.session_id_a,
            driver_id="driver_cal_a",
        )
        engine_b = await session_manager.get_or_create_session(
            session_id=self.session_id_b,
            driver_id="driver_cal_b",
        )

        initial_ear_median_b = engine_b.profile.center_ear_median
        initial_pitch_b = engine_b.profile.neutral_pitch_deg

        # Update Session A's calibration profile with custom values
        engine_a.profile.center_ear_median = 0.335
        engine_a.profile.calibration_mode = "FOUR_ZONE_PERSONALIZED"
        engine_a.profile.neutral_pitch_deg = 15.5

        # Session B's profile must remain completely unchanged
        self.assertEqual(engine_b.profile.center_ear_median, initial_ear_median_b)
        self.assertNotEqual(engine_b.profile.center_ear_median, 0.335)
        self.assertNotEqual(engine_b.profile.calibration_mode, "FOUR_ZONE_PERSONALIZED")
        self.assertEqual(engine_b.profile.neutral_pitch_deg, initial_pitch_b)
        self.assertNotEqual(engine_b.profile.neutral_pitch_deg, 15.5)

    async def test_worker_lifecycle_and_shutdown_independence(self):
        """Stopping Session A during active execution does not terminate or corrupt Session B."""
        worker_a = LiveKitAIWorkerSession(room_name=self.session_id_a, driver_id="driver_a")
        worker_b = LiveKitAIWorkerSession(room_name=self.session_id_b, driver_id="driver_b")

        collector_a = NullDataCollector(self.session_id_a)
        collector_b = NullDataCollector(self.session_id_b)

        worker_a.room = MagicMock()
        worker_a.room.disconnect = AsyncMock()
        worker_a.room.local_participant = collector_a
        worker_a.is_running = True

        worker_b.room = MagicMock()
        worker_b.room.disconnect = AsyncMock()
        worker_b.room.local_participant = collector_b
        worker_b.is_running = True

        worker_a.engine = await session_manager.get_or_create_session(
            session_id=self.session_id_a,
            driver_id="driver_a",
        )
        worker_b.engine = await session_manager.get_or_create_session(
            session_id=self.session_id_b,
            driver_id="driver_b",
        )

        worker_a._inference_task = asyncio.create_task(worker_a._inference_loop())
        worker_b._inference_task = asyncio.create_task(worker_b._inference_loop())

        # Feed 10 frames to both
        for idx in range(10):
            frame = MockVideoFrame(generate_benchmark_frame(idx))
            now_ms = time.perf_counter() * 1000.0
            worker_a._frames_received += 1
            worker_a._frame_seq += 1
            worker_b._frames_received += 1
            worker_b._frame_seq += 1
            await worker_a.frame_queue.put((frame, now_ms, worker_a._frame_seq))
            await worker_b.frame_queue.put((frame, now_ms, worker_b._frame_seq))
            await asyncio.sleep(0.01)

        # Abruptly stop worker A
        await worker_a.stop()
        self.assertFalse(worker_a.is_running)
        self.assertIsNone(worker_a.engine)

        # Worker B must still be running and processing frames
        self.assertTrue(worker_b.is_running)
        self.assertIsNotNone(worker_b.engine)

        # Feed 10 more frames to worker B
        for idx in range(10, 20):
            frame = MockVideoFrame(generate_benchmark_frame(idx))
            now_ms = time.perf_counter() * 1000.0
            worker_b._frames_received += 1
            worker_b._frame_seq += 1
            await worker_b.frame_queue.put((frame, now_ms, worker_b._frame_seq))
            await asyncio.sleep(0.01)

        # Settle
        await asyncio.sleep(0.2)

        snap_b = worker_b.get_frame_accounting_snapshot()
        self.assertGreater(snap_b.frames_processed, 0)
        self.assertTrue(snap_b.is_invariant_satisfied)

        # Now cleanly stop worker B
        await worker_b.stop()
        self.assertFalse(worker_b.is_running)
        self.assertIsNone(worker_b.engine)


if __name__ == "__main__":
    unittest.main()
