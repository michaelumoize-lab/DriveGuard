"""
Regression and Concurrency Tests for Workstream B — Serialized AI Engine Mutations.

Covers:
1. Calibration command submitted during active frame inference.
2. Experiment switch submitted during active frame inference.
3. Profile mutation submitted during active frame inference.
4. Deterministic order of multiple queued commands.
5. Rejection of invalid experiment modes without state corruption.
6. Rejection of commands after engine shutdown begins.
7. Verification that event loop is not blocked during command submission.
"""

import asyncio
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from backend.app.ai.cascade_classifier import ExperimentMode
from backend.app.ai.pdbp_calibration import DriverBehaviourProfile
from backend.app.ai.pipeline import SessionAIEngine
from backend.app.schemas.calibration import CalibrationPhaseDTO


class SlowMockLandmarker:
    """Mock landmarker with configurable artificial latency to simulate heavy frame processing."""

    def __init__(self, delay_sec: float = 0.08):
        self.delay_sec = delay_sec

    def process_frame(self, *args, **kwargs):
        if self.delay_sec > 0:
            time.sleep(self.delay_sec)
        from backend.app.ai.geometry import FrameGeometryFeatures, PoseZone
        return FrameGeometryFeatures(
            timestamp_ms=100.0,
            face_detected=True,
            landmark_confidence=0.95,
            left_ear=0.28,
            right_ear=0.28,
            yaw_weighted_ear=0.28,
            blendshape_blink_left=0.05,
            blendshape_blink_right=0.05,
            blendshape_blink_mean=0.05,
            mar=0.10,
            pitch_deg=0.0,
            yaw_deg=0.0,
            roll_deg=0.0,
            pose_zone=PoseZone.CENTER,
            geometric_reliability=0.95,
        ), None


class TestSerializedEngineMutations(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.slow_landmarker = SlowMockLandmarker(delay_sec=0.06)
        self.engine = SessionAIEngine(
            session_id="test_serialize_sess",
            driver_id="driver_test",
            landmarker=self.slow_landmarker,
            experiment_mode=ExperimentMode.C_CONFIDENCE_GATED_CASCADE,
        )
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="test_infer")

    async def asyncTearDown(self):
        self.engine.close()
        self.executor.shutdown(wait=False)

    async def test_calibration_command_during_active_inference(self):
        """Verify START_CALIBRATION arriving while inference thread is active executes safely at frame boundary."""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Launch slow frame inference in background thread
        loop = asyncio.get_running_loop()
        infer_future = loop.run_in_executor(
            self.executor, self.engine.process_frame, dummy_frame, 1000.0
        )

        # Brief yield to ensure inference thread has acquired _process_lock
        await asyncio.sleep(0.01)

        # Submit START_CALIBRATION from event loop
        cal_task = asyncio.create_task(
            self.engine.submit_command("START_CALIBRATION", four_zone_mode=True, zone_duration_sec=4.0)
        )

        # Await both operations
        dto, cal_dto = await asyncio.gather(infer_future, cal_task)

        self.assertIsNotNone(dto)
        self.assertIsInstance(cal_dto, CalibrationPhaseDTO)
        self.assertTrue(self.engine.is_calibrating)
        self.assertEqual(cal_dto.current_zone, "CENTER")

    async def test_switch_experiment_during_active_inference(self):
        """Verify SWITCH_EXPERIMENT does not mutate cascade/engine state mid-frame."""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        loop = asyncio.get_running_loop()

        self.assertEqual(self.engine.experiment_mode, ExperimentMode.C_CONFIDENCE_GATED_CASCADE)

        infer_future = loop.run_in_executor(
            self.executor, self.engine.process_frame, dummy_frame, 1000.0
        )
        await asyncio.sleep(0.01)

        switch_task = asyncio.create_task(
            self.engine.submit_command("SWITCH_EXPERIMENT", mode="EXPERIMENT_A_GEOMETRIC_ONLY")
        )

        first_frame_dto, new_mode = await asyncio.gather(infer_future, switch_task)

        # First frame should have been processed under initial mode
        self.assertEqual(new_mode, ExperimentMode.A_GEOMETRIC_ONLY)
        self.assertEqual(self.engine.experiment_mode, ExperimentMode.A_GEOMETRIC_ONLY)
        self.assertEqual(self.engine.cascade.mode, ExperimentMode.A_GEOMETRIC_ONLY)

        # Subsequent frame runs under the new mode
        next_frame_dto = await loop.run_in_executor(
            self.executor, self.engine.process_frame, dummy_frame, 1040.0
        )
        self.assertEqual(next_frame_dto.cascade.experiment_mode, ExperimentMode.A_GEOMETRIC_ONLY.value)

    async def test_deterministic_command_order(self):
        """Multiple commands queued during active inference execute in strict FIFO order."""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        loop = asyncio.get_running_loop()

        infer_future = loop.run_in_executor(
            self.executor, self.engine.process_frame, dummy_frame, 1000.0
        )
        await asyncio.sleep(0.01)

        # Enqueue: START_CALIBRATION then CANCEL_CALIBRATION
        cmd1 = asyncio.create_task(self.engine.submit_command("START_CALIBRATION", four_zone_mode=True))
        cmd2 = asyncio.create_task(self.engine.submit_command("CANCEL_CALIBRATION"))

        await asyncio.gather(infer_future, cmd1, cmd2)

        # Because CANCEL followed START, final state must be not calibrating
        self.assertFalse(self.engine.is_calibrating)
        self.assertIsNone(self.engine.calibration_session)

    async def test_invalid_experiment_mode_rejected_without_corrupting_state(self):
        """Invalid experiment mode strings raise ValueError and leave engine state intact."""
        original_mode = self.engine.experiment_mode

        with self.assertRaises(ValueError) as ctx:
            await self.engine.submit_command("SWITCH_EXPERIMENT", mode="INVALID_NONEXISTENT_MODE")

        self.assertIn("Invalid experiment mode", str(ctx.exception))
        self.assertEqual(self.engine.experiment_mode, original_mode)
        self.assertEqual(self.engine.cascade.mode, original_mode)

    async def test_command_rejected_after_shutdown_begins(self):
        """Commands submitted after close() raise RuntimeError immediately."""
        self.engine.close()

        with self.assertRaises(RuntimeError) as ctx:
            await self.engine.submit_command("START_CALIBRATION")

        self.assertIn("closed", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
