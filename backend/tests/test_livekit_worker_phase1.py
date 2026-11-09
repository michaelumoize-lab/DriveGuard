"""
Phase 1 Verification Tests:
1. Frame normalization & zero-resize bypass verification for 640x480.
2. Safe normalization fallback for unexpected resolutions (720p, 1080p, sub-VGA).
3. Session lifecycle cleanup & termination in session_manager.
"""

import asyncio
import unittest
import numpy as np
import cv2

from backend.app.services.livekit_worker import (
    LiveKitAIWorkerSession,
    livekit_worker_manager,
    normalize_frame_width,
)
from backend.app.services.session_manager import session_manager


class TestLiveKitWorkerPhase1(unittest.IsolatedAsyncioTestCase):

    def test_frame_normalization_dimensions(self):
        """Verify normalization logic: 640px bypass, downscaling >640, upscaling <320."""
        # 1. Target standard resolution (640x480)
        img_640 = np.zeros((480, 640, 3), dtype=np.uint8)
        norm_img = normalize_frame_width(img_640)
        self.assertEqual(norm_img.shape, (480, 640, 3))
        self.assertIs(norm_img, img_640, "640x480 frame should bypass resize completely (zero-copy)")

        # 2. Unexpected 720p resolution (1280x720)
        img_720p = np.zeros((720, 1280, 3), dtype=np.uint8)
        norm_720 = normalize_frame_width(img_720p)
        self.assertEqual(norm_720.shape, (360, 640, 3))

        # 3. Unexpected 1080p resolution (1920x1080)
        img_1080p = np.zeros((1080, 1920, 3), dtype=np.uint8)
        norm_1080 = normalize_frame_width(img_1080p)
        self.assertEqual(norm_1080.shape, (360, 640, 3))

        # 4. Unexpected sub-VGA resolution (240x180)
        img_sub = np.zeros((180, 240, 3), dtype=np.uint8)
        norm_sub = normalize_frame_width(img_sub)
        self.assertEqual(norm_sub.shape, (360, 480, 3))

    async def test_session_lifecycle_cleanup(self):
        """Verify that stopping a worker terminates and frees the session in session_manager."""
        test_room = "test_phase1_cleanup_room"
        test_driver = "test_phase1_driver"

        # 1. Ensure engine is registered in session_manager
        engine = await session_manager.get_or_create_session(
            session_id=test_room,
            driver_id=test_driver,
        )
        self.assertIsNotNone(engine)
        self.assertIn(test_room, session_manager._sessions)

        # 2. Instantiate a worker session
        worker = LiveKitAIWorkerSession(room_name=test_room, driver_id=test_driver)
        worker.engine = engine
        worker.is_running = True

        # Register worker in manager
        livekit_worker_manager._workers[test_room] = worker

        # 3. Trigger worker.stop()
        await worker.stop()

        # 4. Verify session_manager cleanly terminated the session
        self.assertNotIn(test_room, session_manager._sessions, "Engine must be removed from session_manager upon worker stop")
        self.assertFalse(worker.is_running)

        # Cleanup manager
        livekit_worker_manager._workers.pop(test_room, None)


if __name__ == "__main__":
    unittest.main()
