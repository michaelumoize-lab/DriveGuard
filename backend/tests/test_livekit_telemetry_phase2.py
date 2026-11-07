"""
Phase 2 Verification Tests: Real-Time Telemetry, Frame Synchronization & Latency Measurement
1. High-frequency telemetry serialization & timing breakdown (TimingBreakdownDTO, TelemetryPacketDTO).
2. Important safety event state transition detection & unique event generation (SafetyEventDTO).
3. Reliable delivery & Application-level ACK tracking (EVENT_ACK).
4. Non-blocking retry mechanism for unacknowledged safety events.
5. Client state synchronization on reconnection (SYNC_STATE / STATE_SYNC).
6. Latency ping probe handling (PING / PONG with queue depth).
"""

import asyncio
import json
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.app.schemas.telemetry import (
    TimingBreakdownDTO,
    SafetyEventDTO,
    EventAckDTO,
    TelemetryPacketDTO,
    FrameTelemetryDTO,
    AlertStateEnum,
    PoseZoneEnum,
    HeadPoseDTO,
    EyeMetricsDTO,
    CascadeStatusDTO,
    ConventionalFixedComparisonDTO,
)
from backend.app.services.livekit_worker import LiveKitAIWorkerSession


class TestLiveKitTelemetryPhase2(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.worker = LiveKitAIWorkerSession(room_name="test_phase2_room", driver_id="test_driver")
        self.worker.room = MagicMock()
        self.worker.room.local_participant = MagicMock()
        self.worker.room.local_participant.publish_data = AsyncMock()
        self.worker.engine = MagicMock()

    def _create_mock_telemetry(self, state: AlertStateEnum = AlertStateEnum.NORMAL, frame_idx: int = 1) -> FrameTelemetryDTO:
        return FrameTelemetryDTO(
            session_id="test_phase2_room",
            driver_id="test_driver",
            frame_index=frame_idx,
            timestamp_ms=time.time() * 1000.0,
            fps=24.5,
            face_detected=True,
            state=state,
            risk_score=0.15 if state == AlertStateEnum.NORMAL else 0.85,
            pose_zone=PoseZoneEnum.CENTER,
            reason_codes=["NORMAL_GAZE"] if state == AlertStateEnum.NORMAL else ["HIGH_PERCLOS"],
            head_pose=HeadPoseDTO(pitch_deg=0.0, yaw_deg=0.0, roll_deg=0.0),
            eye_metrics=EyeMetricsDTO(
                ear_raw=0.310,
                ear_left=0.305,
                ear_right=0.315,
                ear_normalized=1.0,
                ear_closure_threshold=0.22,
                is_eye_closed=False,
                fused_closure_prob=0.05,
                perclos_pct=4.2 if state == AlertStateEnum.NORMAL else 32.5,
            ),
            cascade=CascadeStatusDTO(
                experiment_mode="EXPERIMENT_C_CONFIDENCE_GATED_CASCADE",
                cnn_triggered=False,
                geom_open_prob=0.95,
                geom_reliability=0.9,
                arbitration_winner="GEOMETRY",
            ),
            comparison=ConventionalFixedComparisonDTO(
                fixed_cutoff_ear=0.22,
                fixed_alert_state="NORMAL",
                fixed_false_alarm=False,
                personalised_zone="CENTER",
                personalised_threshold=0.22,
                is_mirror_check_suppressed=False,
            ),
            inference_latency_ms=18.4,
            timing=TimingBreakdownDTO(
                queue_latency_ms=2.1,
                infer_duration_ms=18.4,
                server_total_ms=20.5,
            ),
        )

    def test_timing_breakdown_and_telemetry_packet_schema(self):
        """Verify TelemetryPacketDTO and TimingBreakdownDTO structure and serialization."""
        tel_dto = self._create_mock_telemetry(frame_idx=42)
        packet = TelemetryPacketDTO(
            v=2,
            type="TELEMETRY",
            session_id="test_phase2_room",
            seq=42,
            timestamp=time.time(),
            telemetry=tel_dto,
            echo_client_ts=123456.78,
        )
        packet_dict = packet.model_dump()

        self.assertEqual(packet_dict["v"], 2)
        self.assertEqual(packet_dict["type"], "TELEMETRY")
        self.assertEqual(packet_dict["seq"], 42)
        self.assertIsNotNone(packet_dict["telemetry"]["timing"])
        self.assertEqual(packet_dict["telemetry"]["timing"]["queue_latency_ms"], 2.1)
        self.assertEqual(packet_dict["telemetry"]["timing"]["infer_duration_ms"], 18.4)
        self.assertEqual(packet_dict["telemetry"]["timing"]["server_total_ms"], 20.5)

    def test_safety_event_dto_schema(self):
        """Verify SafetyEventDTO structure and serialization for state transitions."""
        event = SafetyEventDTO(
            v=2,
            type="SAFETY_EVENT",
            event_id="evt_abc12345",
            session_id="test_phase2_room",
            event_seq=1,
            event_type="STATE_TRANSITION",
            previous_state="NORMAL",
            new_state="CRITICAL",
            timestamp=time.time(),
            metrics={"risk_score": 0.88, "ear": 0.12, "perclos": 35.0},
        )
        event_dict = event.model_dump()

        self.assertEqual(event_dict["v"], 2)
        self.assertEqual(event_dict["type"], "SAFETY_EVENT")
        self.assertEqual(event_dict["event_id"], "evt_abc12345")
        self.assertEqual(event_dict["new_state"], "CRITICAL")
        self.assertEqual(event_dict["previous_state"], "NORMAL")
        self.assertEqual(event_dict["metrics"]["perclos"], 35.0)

    async def test_publish_reliable_safety_event_and_ack(self):
        """Verify reliable safety event dispatch, tracking in unacknowledged map, and ACK confirmation."""
        event_payload = {
            "v": 2,
            "type": "SAFETY_EVENT",
            "event_id": "evt_test_001",
            "session_id": "test_phase2_room",
            "event_seq": 1,
            "event_type": "STATE_TRANSITION",
            "previous_state": "NORMAL",
            "new_state": "CRITICAL",
            "timestamp": time.time(),
            "metrics": {"risk_score": 0.9},
        }

        # 1. Publish reliable event
        await self.worker._publish_reliable_safety_event(event_payload)

        # Worker should have published via reliable=True
        self.worker.room.local_participant.publish_data.assert_called_once()
        args, kwargs = self.worker.room.local_participant.publish_data.call_args
        self.assertTrue(kwargs.get("reliable"))

        # Event should be in unacknowledged map
        self.assertIn("evt_test_001", self.worker._unacknowledged_events)
        self.assertEqual(self.worker._unacknowledged_events["evt_test_001"]["retries"], 0)

        # 2. Simulate client EVENT_ACK
        mock_packet = MagicMock()
        mock_packet.data = json.dumps({
            "action": "EVENT_ACK",
            "event_id": "evt_test_001",
            "event_seq": 1,
            "client_timestamp": 5000.0,
        }).encode("utf-8")

        await self.worker._handle_client_control_message(mock_packet)

        # Event must be removed from unacknowledged map upon ACK
        self.assertNotIn("evt_test_001", self.worker._unacknowledged_events)

    async def test_retry_unacknowledged_event_aborts_when_acked(self):
        """Verify that the retry task aborts without duplicate sends when an ACK arrives promptly."""
        event_payload = {
            "v": 2,
            "type": "SAFETY_EVENT",
            "event_id": "evt_test_fast_ack",
            "session_id": "test_phase2_room",
            "event_seq": 2,
            "event_type": "STATE_TRANSITION",
            "previous_state": "NORMAL",
            "new_state": "WARNING",
            "timestamp": time.time(),
            "metrics": {},
        }
        self.worker.is_running = True
        self.worker._unacknowledged_events["evt_test_fast_ack"] = {
            "payload": event_payload,
            "retries": 0,
            "sent_at": time.perf_counter(),
        }

        # Simulate incoming ACK before retry sleep elapses
        ack_packet = MagicMock()
        ack_packet.data = json.dumps({
            "action": "EVENT_ACK",
            "event_id": "evt_test_fast_ack",
            "event_seq": 2,
        }).encode("utf-8")

        # Start retry in background
        retry_task = asyncio.create_task(self.worker._retry_unacknowledged_event("evt_test_fast_ack"))
        # Immediately deliver ACK
        await self.worker._handle_client_control_message(ack_packet)
        await retry_task

        # No retries should have been published since it was already acked
        self.worker.room.local_participant.publish_data.assert_not_called()
        self.assertNotIn("evt_test_fast_ack", self.worker._unacknowledged_events)

    async def test_sync_state_on_reconnection(self):
        """Verify SYNC_STATE returns authoritative current state and latest safety event."""
        self.worker._last_state = "CRITICAL"
        self.worker._latest_safety_event = {
            "event_id": "evt_crit_123",
            "new_state": "CRITICAL",
        }

        sync_packet = MagicMock()
        sync_packet.data = json.dumps({"action": "SYNC_STATE"}).encode("utf-8")

        await self.worker._handle_client_control_message(sync_packet)

        # Worker should have replied with STATE_SYNC reliably
        self.worker.room.local_participant.publish_data.assert_called_once()
        args, kwargs = self.worker.room.local_participant.publish_data.call_args
        self.assertTrue(kwargs.get("reliable"))
        sent_dict = json.loads(args[0].decode("utf-8"))

        self.assertEqual(sent_dict["type"], "STATE_SYNC")
        self.assertEqual(sent_dict["current_state"], "CRITICAL")
        self.assertEqual(sent_dict["latest_safety_event"]["event_id"], "evt_crit_123")

    async def test_ping_pong_latency_probe(self):
        """Verify PING action returns PONG echoing client_ts and queue depth."""
        ping_packet = MagicMock()
        ping_packet.data = json.dumps({
            "action": "PING",
            "client_ts": 12345.67,
            "ping_id": 7,
        }).encode("utf-8")

        await self.worker._handle_client_control_message(ping_packet)

        self.assertEqual(self.worker._last_client_ping_ts, 12345.67)
        self.worker.room.local_participant.publish_data.assert_called_once()
        args, kwargs = self.worker.room.local_participant.publish_data.call_args
        sent_dict = json.loads(args[0].decode("utf-8"))

        self.assertEqual(sent_dict["type"], "PONG")
        self.assertEqual(sent_dict["client_ts"], 12345.67)
        self.assertEqual(sent_dict["ping_id"], 7)
        self.assertIn("queue_depth", sent_dict)


if __name__ == "__main__":
    unittest.main()
