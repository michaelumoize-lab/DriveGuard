"""
Phase 3 Verification Tests: MediaPipe VIDEO Mode, Temporal Landmark Tracking & Session Isolation
1. VIDEO-mode initialization.
2. Successful sequential frame processing.
3. Strictly increasing timestamps per instance.
4. Duplicate timestamp handling without crash or corruption.
5. Out-of-order timestamp handling without crash.
6. Two simultaneous independent sessions with isolated tracking.
7. Concurrent multi-session thread execution without collision.
8. Session termination, engine closing, and landmarker resource cleanup.
9. Reconnection lifecycle and new session creation.
10. Frame drops and large temporal gap handling.
11. Regression testing for EAR, MAR, head pose, and cascade outputs.
"""

import asyncio
import threading
import time
import unittest
import numpy as np
import cv2

from backend.app.ai.landmarker import DriveGuardFaceLandmarker
from backend.app.ai.pipeline import SessionAIEngine
from backend.app.services.session_manager import SessionManager, session_manager


class TestMediaPipeVideoPhase3(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        # Create a synthetic 640x480 test image (BGR)
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a synthetic face/oval with eye-like features so landmarker pipeline handles it cleanly
        cv2.ellipse(self.frame, (320, 240), (120, 160), 0, 0, 360, (200, 200, 200), -1)
        cv2.circle(self.frame, (280, 210), 15, (50, 50, 50), -1)
        cv2.circle(self.frame, (360, 210), 15, (50, 50, 50), -1)

    def test_01_video_mode_initialization(self):
        """Verify that DriveGuardFaceLandmarker initializes with running_mode='VIDEO' by default."""
        lm = DriveGuardFaceLandmarker(running_mode="VIDEO")
        self.assertEqual(lm.running_mode, "VIDEO")
        diag = lm.get_diagnostics()
        self.assertEqual(diag["running_mode"], "VIDEO")
        self.assertIsNone(diag["last_timestamp_ms"])
        self.assertEqual(diag["duplicate_or_unordered_count"], 0)
        self.assertFalse(diag["is_closed"])
        lm.close()
        self.assertTrue(lm.get_diagnostics()["is_closed"])

    def test_02_sequential_frame_processing_monotonic_timestamps(self):
        """Verify sequential frames with strictly increasing timestamps process smoothly."""
        lm = DriveGuardFaceLandmarker(running_mode="VIDEO")
        timestamps = [1000.0, 1040.0, 1080.0, 1120.0]

        for ts in timestamps:
            features, lms = lm.process_frame(self.frame, timestamp_ms=ts)
            self.assertEqual(features.timestamp_ms, ts)
            diag = lm.get_diagnostics()
            self.assertEqual(diag["last_timestamp_ms"], int(ts))
            self.assertEqual(diag["duplicate_or_unordered_count"], 0)
            self.assertEqual(diag["inference_exceptions_count"], 0)

        lm.close()

    def test_03_duplicate_timestamp_handling(self):
        """Verify safe timestamp policy: duplicate timestamps must not crash MediaPipe and must increment monotonically."""
        lm = DriveGuardFaceLandmarker(running_mode="VIDEO")

        # 1. First frame at 2000ms
        lm.process_frame(self.frame, timestamp_ms=2000.0)
        self.assertEqual(lm.get_diagnostics()["last_timestamp_ms"], 2000)

        # 2. Duplicate frame also at 2000ms
        features, _ = lm.process_frame(self.frame, timestamp_ms=2000.0)
        diag = lm.get_diagnostics()

        # Must record diagnostic and enforce strict +1ms monotonicity
        self.assertEqual(diag["duplicate_or_unordered_count"], 1)
        self.assertEqual(diag["last_timestamp_ms"], 2001)
        self.assertEqual(diag["inference_exceptions_count"], 0)

        # 3. Subsequent valid frame at 2040ms
        lm.process_frame(self.frame, timestamp_ms=2040.0)
        self.assertEqual(lm.get_diagnostics()["last_timestamp_ms"], 2040)

        lm.close()

    def test_04_out_of_order_timestamp_handling(self):
        """Verify that older/out-of-order timestamps do not crash or violate monotonicity."""
        lm = DriveGuardFaceLandmarker(running_mode="VIDEO")

        # Frame at 3000ms
        lm.process_frame(self.frame, timestamp_ms=3000.0)
        self.assertEqual(lm.get_diagnostics()["last_timestamp_ms"], 3000)

        # Out-of-order frame at 2950ms (50ms in the past)
        features, _ = lm.process_frame(self.frame, timestamp_ms=2950.0)
        diag = lm.get_diagnostics()

        self.assertEqual(diag["duplicate_or_unordered_count"], 1)
        self.assertEqual(diag["last_timestamp_ms"], 3001)
        self.assertEqual(diag["inference_exceptions_count"], 0)

        lm.close()

    def test_05_two_simultaneous_independent_sessions(self):
        """Verify two independent sessions have isolated landmarkers and independent timestamp streams."""
        engine_a = SessionAIEngine(session_id="room_alpha", driver_id="driver_a")
        engine_b = SessionAIEngine(session_id="room_beta", driver_id="driver_b")

        # Verify distinct landmarker instances
        self.assertIsNot(engine_a.landmarker, engine_b.landmarker)

        # Interleave timestamps: Driver A is at t=1000..1080, Driver B is at t=100..180
        # If instances were shared, Driver B's lower timestamps would crash Driver A's stream!
        dto_a1 = engine_a.process_frame(self.frame, timestamp_ms=1000.0)
        dto_b1 = engine_b.process_frame(self.frame, timestamp_ms=100.0)

        dto_a2 = engine_a.process_frame(self.frame, timestamp_ms=1040.0)
        dto_b2 = engine_b.process_frame(self.frame, timestamp_ms=140.0)

        # Verify independent progress
        diag_a = engine_a.landmarker.get_diagnostics()
        diag_b = engine_b.landmarker.get_diagnostics()

        self.assertEqual(diag_a["last_timestamp_ms"], 1040)
        self.assertEqual(diag_b["last_timestamp_ms"], 140)
        self.assertEqual(diag_a["duplicate_or_unordered_count"], 0)
        self.assertEqual(diag_b["duplicate_or_unordered_count"], 0)

        engine_a.close()
        engine_b.close()

    def test_06_concurrent_session_execution_in_threads(self):
        """Verify multiple threads running separate driver engines simultaneously encounter zero race conditions."""
        engine_1 = SessionAIEngine(session_id="room_thread_1", driver_id="driver_1")
        engine_2 = SessionAIEngine(session_id="room_thread_2", driver_id="driver_2")

        errors = []

        def worker_loop(engine: SessionAIEngine, base_ts: float):
            try:
                for i in range(15):
                    ts = base_ts + i * 33.3
                    dto = engine.process_frame(self.frame, timestamp_ms=ts)
                    self.assertIsNotNone(dto)
                    time.sleep(0.005)
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=worker_loop, args=(engine_1, 5000.0))
        t2 = threading.Thread(target=worker_loop, args=(engine_2, 1000.0))

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertEqual(len(errors), 0, f"Thread errors encountered: {errors}")
        self.assertEqual(engine_1.frame_index, 15)
        self.assertEqual(engine_2.frame_index, 15)

        engine_1.close()
        engine_2.close()

    async def test_07_session_manager_lifecycle_and_landmarker_cleanup(self):
        """Verify that session_manager creates isolated instances and cleans them up on termination."""
        sm = SessionManager()
        s_id = "test_cleanup_room"
        engine = await sm.get_or_create_session(session_id=s_id, driver_id="cleanup_driver")

        self.assertIsNotNone(engine.landmarker)
        self.assertFalse(engine.landmarker.get_diagnostics()["is_closed"])

        # Terminate session
        summary = await sm.terminate_session(s_id)
        self.assertIsNotNone(summary)
        self.assertNotIn(s_id, sm._sessions)

        # Landmarker must be closed
        self.assertTrue(engine.landmarker.get_diagnostics()["is_closed"])

        # Calling process_frame on a closed engine returns clean unavailable telemetry
        closed_dto = engine.process_frame(self.frame, timestamp_ms=9000.0)
        self.assertFalse(closed_dto.face_detected)
        self.assertIn("SESSION_CLOSED", closed_dto.reason_codes)

    async def test_08_reconnection_and_new_session_isolation(self):
        """Verify that after disconnecting a session, a new session starts with a clean landmarker state."""
        sm = SessionManager()
        room = "reconnect_room"

        # 1. First session
        eng1 = await sm.get_or_create_session(room, "driver_orig")
        eng1.process_frame(self.frame, timestamp_ms=10000.0)
        self.assertEqual(eng1.landmarker.get_diagnostics()["last_timestamp_ms"], 10000)

        # 2. Terminate session
        await sm.terminate_session(room)
        self.assertTrue(eng1.landmarker.get_diagnostics()["is_closed"])

        # 3. Reconnect to same room name
        eng2 = await sm.get_or_create_session(room, "driver_reconnected")
        self.assertIsNot(eng1, eng2)
        self.assertFalse(eng2.landmarker.get_diagnostics()["is_closed"])

        # Can start timestamps from earlier time (e.g. 500ms) without timestamp conflict
        eng2.process_frame(self.frame, timestamp_ms=500.0)
        self.assertEqual(eng2.landmarker.get_diagnostics()["last_timestamp_ms"], 500)
        self.assertEqual(eng2.landmarker.get_diagnostics()["duplicate_or_unordered_count"], 0)

        await sm.terminate_session(room)

    def test_09_frame_drops_and_large_temporal_gaps(self):
        """Verify large time gaps (e.g. dropped frames) do not corrupt tracker or generate false blinks."""
        engine = SessionAIEngine(session_id="gap_test", driver_id="driver_gap")

        # Initial frames
        engine.process_frame(self.frame, timestamp_ms=1000.0)
        engine.process_frame(self.frame, timestamp_ms=1033.0)

        # Large 1.5-second frame drop gap
        dto_gap = engine.process_frame(self.frame, timestamp_ms=2533.0)

        self.assertIsNotNone(dto_gap)
        self.assertEqual(dto_gap.frame_index, 3)
        # Should not fabricate false blinks
        self.assertEqual(len(engine.kinematics.blink_history), 0)

        engine.close()

    def test_10_regression_pipeline_metrics(self):
        """Verify pipeline output format remains fully compatible with downstream telemetry DTOs."""
        engine = SessionAIEngine(session_id="regression_test", driver_id="driver_reg")
        dto = engine.process_frame(self.frame, timestamp_ms=1000.0)

        # Validate core fields
        self.assertEqual(dto.session_id, "regression_test")
        self.assertEqual(dto.driver_id, "driver_reg")
        self.assertIsInstance(dto.fps, float)
        self.assertIsInstance(dto.risk_score, float)
        self.assertIsNotNone(dto.head_pose)
        self.assertIsNotNone(dto.eye_metrics)
        self.assertIsNotNone(dto.cascade)
        self.assertIsNotNone(dto.comparison)

        # Summary diagnostics include landmarker diagnostics
        summary = engine.get_session_summary()
        self.assertIn("landmarker_diagnostics", summary)
        self.assertEqual(summary["landmarker_diagnostics"]["running_mode"], "VIDEO")

        engine.close()


if __name__ == "__main__":
    unittest.main()
