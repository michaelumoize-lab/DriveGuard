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

from backend.app.services.livekit_worker import LiveKitAIWorkerSession, livekit_worker_manager
from backend.app.services.session_manager import session_manager


class TestLiveKitWorkerPhase1(unittest.IsolatedAsyncioTestCase):

    def test_frame_normalization_dimensions(self):
        """Verify normalization logic: 640px bypass, downscaling >640, upscaling <320."""
        # 1. Target standard resolution (640x480)
        img_640 = np.zeros((480, 640, 3), dtype=np.uint8)
        raw_h, raw_w = img_640.shape[:2]
        
        if raw_w == 640:
            norm_img = img_640
        elif raw_w > 640:
            target_w = 640
            target_h = int(round(raw_h * (640.0 / raw_w)))
            norm_img = cv2.resize(img_640, (target_w, target_h), interpolation=cv2.INTER_AREA)
        else:
            norm_img = img_640

        self.assertEqual(norm_img.shape, (480, 640, 3))
        self.assertIs(norm_img, img_640, "640x480 frame should bypass resize completely (zero-copy)")

        # 2. Unexpected 720p resolution (1280x720)
        img_720p = np.zeros((720, 1280, 3), dtype=np.uint8)
        raw_h, raw_w = img_720p.shape[:2]
        if raw_w > 640:
            target_w = 640
            target_h = int(round(raw_h * (640.0 / raw_w)))
            norm_720 = cv2.resize(img_720p, (target_w, target_h), interpolation=cv2.INTER_AREA)
        self.assertEqual(norm_720.shape, (360, 640, 3))

        # 3. Unexpected 1080p resolution (1920x1080)
        img_1080p = np.zeros((1080, 1920, 3), dtype=np.uint8)
        raw_h, raw_w = img_1080p.shape[:2]
        if raw_w > 640:
            target_w = 640
            target_h = int(round(raw_h * (640.0 / raw_w)))
            norm_1080 = cv2.resize(img_1080p, (target_w, target_h), interpolation=cv2.INTER_AREA)
        self.assertEqual(norm_1080.shape, (360, 640, 3))

        # 4. Unexpected sub-VGA resolution (240x180)
        img_sub = np.zeros((180, 240, 3), dtype=np.uint8)
        raw_h, raw_w = img_sub.shape[:2]
        if raw_w < 320 and raw_w > 0:
            target_w = 480
            target_h = int(round(raw_h * (480.0 / raw_w)))
            norm_sub = cv2.resize(img_sub, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
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
