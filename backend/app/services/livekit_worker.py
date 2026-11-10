"""
DriveGuard LiveKit AI Worker Service.

Manages autonomous WebRTC room participants that:
  1. Join DriveGuard sessions on LiveKit Cloud SFU.
  2. Subscribe to remote camera video tracks via `rtc.VideoStream`.
  3. Enforce a bounded latest-frame buffer (maxsize=1) dropping stale frames to guarantee zero lag.
  4. Stream real-time AI telemetry (EAR, PERCLOS, PoseZone, Risk Score) over WebRTC Data Channel.
  5. Handle bidirectional control events (calibration phases, experiment mode switching).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from dataclasses import dataclass
import json
import logging
import time
import uuid
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from livekit import api, rtc

from backend.app.ai.cascade_classifier import ExperimentMode
from backend.app.ai.pipeline import SessionAIEngine
from backend.app.core.config import settings
from backend.app.services.profile_store import profile_store
from backend.app.services.session_manager import session_manager

logger = logging.getLogger("driveguard.livekit_worker")
logger.setLevel(logging.INFO)

WORKER_BOOT_ID = uuid.uuid4().hex


def generate_session_epoch() -> str:
    """Generate a collision-resistant session epoch formatted as <unix_timestamp>_<128bit_uuid4_hex>."""
    return f"{int(time.time())}_{uuid.uuid4().hex}"


@dataclass(frozen=True)
class FrameAccountingSnapshot:
    """Immutable snapshot of frame accounting counters on the event loop."""

    frames_received: int
    frames_dequeued: int
    frames_processed: int
    frames_replaced: int
    frames_stale_discarded: int
    frames_errored: int
    frames_aborted: int
    frames_queued: int
    frames_in_flight: int
    threads_executing: int = 0

    @property
    def total_accounted(self) -> int:
        return (
            self.frames_replaced
            + self.frames_stale_discarded
            + self.frames_errored
            + self.frames_aborted
            + self.frames_processed
            + self.frames_queued
            + self.frames_in_flight
        )

    @property
    def is_invariant_satisfied(self) -> bool:
        return self.frames_received == self.total_accounted

    @property
    def discrepancy(self) -> int:
        return self.frames_received - self.total_accounted


def normalize_frame_width(img_bgr: np.ndarray) -> np.ndarray:
    """Normalize input frame dimensions: bypass resize if 640px wide or 320<=w<640, downscale >640, upscale <320."""
    raw_h, raw_w = img_bgr.shape[:2]
    if raw_w == 640:
        return img_bgr
    elif raw_w > 640:
        target_w = 640
        target_h = int(round(raw_h * (640.0 / raw_w)))
        return cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
    elif raw_w < 320 and raw_w > 0:
        target_w = 480
        target_h = int(round(raw_h * (480.0 / raw_w)))
        return cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    else:
        return img_bgr


def _timed_process_frame(
    engine: SessionAIEngine,
    norm_img: np.ndarray,
    recv_mono_ms: float,
    t_submit: float,
) -> Tuple[Any, float, float]:
    """Execute engine.process_frame measuring executor wait and actual inference duration."""
    t_start = time.perf_counter()
    executor_wait_ms = max(0.0, (t_start - t_submit) * 1000.0)
    try:
        telemetry_dto = engine.process_frame(norm_img, recv_mono_ms, is_rgb=True)
    except TypeError:
        telemetry_dto = engine.process_frame(norm_img, recv_mono_ms)
    t_end = time.perf_counter()
    process_frame_ms = max(0.0, (t_end - t_start) * 1000.0)
    return telemetry_dto, executor_wait_ms, process_frame_ms


class LiveKitAIWorkerSession:
    """Manages an autonomous Python AI participant in a single LiveKit room."""

    def __init__(
        self,
        room_name: str,
        driver_id: str = "driver_default",
        experiment_mode: Optional[str] = None,
    ) -> None:
        self.room_name = room_name
        self.driver_id = driver_id
        self.experiment_mode = experiment_mode

        self.room = rtc.Room()
        self.engine: Optional[SessionAIEngine] = None
        self.is_running = False
        self._calibration_completion_sent = False
        self._shutdown_task: Optional[asyncio.Task] = None
        self._last_logged_dims: Optional[Tuple[int, int, int, int]] = None

        self.worker_boot_id: str = WORKER_BOOT_ID
        # Format: "<unix_timestamp>_<128bit_uuid4_hex>" ensuring timestamp sortability + 128-bit collision resistance
        self.session_id_uuid: str = uuid.uuid4().hex
        self.session_epoch: str = f"{int(time.time())}_{self.session_id_uuid}"

        # Phase 2: State Tracking & Reliable Safety Event Store
        self._last_state: str = "NORMAL"
        self._frame_seq: int = 0
        self._event_seq: int = 0
        self._unacknowledged_events: Dict[str, dict] = {}
        self._latest_safety_event: Optional[dict] = None
        self._last_client_ping_ts: Optional[float] = None
        self._last_publish_duration_ms: Optional[float] = None

        # Thread-Safe Frame Accounting Counters (owned strictly on asyncio event loop)
        self._frames_received: int = 0
        self._frames_dequeued: int = 0
        self._frames_processed: int = 0
        self._frames_replaced: int = 0
        self._frames_stale_discarded: int = 0
        self._frames_errored: int = 0
        self._frames_aborted: int = 0
        self._frames_in_flight: int = 0

        # Dedicated single-thread worker for CPU inference to decouple from event loop
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"dg_infer_{self.room_name[:8]}",
        )
        self._active_cfut: Optional[concurrent.futures.Future] = None
        self._threads_executing: int = 0
        self._coro_cancelled_for_active_frame: bool = False

        # Latest-frame queue with maxsize=1 (drops stale frames immediately)
        # Tuple contains: (frame, ingest_mono_ms, frame_seq_num)
        self.frame_queue: asyncio.Queue[Tuple[rtc.VideoFrame, float, int]] = asyncio.Queue(maxsize=1)

        # Worker lifecycle tasks
        self._ingest_task: Optional[asyncio.Task] = None
        self._inference_task: Optional[asyncio.Task] = None
        self._current_video_stream: Optional[rtc.VideoStream] = None
        self._stream_cleanup_lock: asyncio.Lock = asyncio.Lock()
        self._stream_close_task: Optional[asyncio.Task] = None
        self._stream_close_tasks: Dict[int, asyncio.Task] = {}
        self._closed_stream_ids: set[int] = set()
        self._stream_cleanup_error_count: int = 0
        self._last_stream_cleanup_error: Optional[str] = None
        self._is_stopped: bool = False
        self._stop_lock: asyncio.Lock = asyncio.Lock()

    @property
    def stream_cleanup_error_count(self) -> int:
        """Total count of exceptions encountered during VideoStream aclose() native teardown."""
        return self._stream_cleanup_error_count

    @property
    def last_stream_cleanup_error(self) -> Optional[str]:
        """Description of the most recent VideoStream cleanup exception, or None."""
        return self._last_stream_cleanup_error

    def get_frame_accounting_snapshot(self) -> FrameAccountingSnapshot:
        """Return an immutable snapshot of all frame lifecycle counters on the event loop."""
        return FrameAccountingSnapshot(
            frames_received=self._frames_received,
            frames_dequeued=self._frames_dequeued,
            frames_processed=self._frames_processed,
            frames_replaced=self._frames_replaced,
            frames_stale_discarded=self._frames_stale_discarded,
            frames_errored=self._frames_errored,
            frames_aborted=self._frames_aborted,
            frames_queued=self.frame_queue.qsize(),
            frames_in_flight=self._frames_in_flight,
            threads_executing=self._threads_executing,
        )

    def _on_inference_thread_done(self, fut: concurrent.futures.Future) -> None:
        """Thread-safe callback executed on the event loop when the worker thread finishes process_frame."""
        self._threads_executing = max(0, self._threads_executing - 1)
        if fut is not self._active_cfut:
            return
        self._active_cfut = None
        if self._coro_cancelled_for_active_frame:
            # Coroutine was cancelled while OS thread was still executing; reconcile now that thread has terminated
            if self._frames_in_flight > 0:
                self._frames_in_flight -= 1
                self._frames_aborted += 1
            self._coro_cancelled_for_active_frame = False
            logger.info(
                f"[LiveKit Worker] Inference thread terminated after coroutine cancellation. "
                f"Reconciled frame in-flight -> aborted (in_flight={self._frames_in_flight}, aborted={self._frames_aborted})."
            )

    async def start(self) -> None:
        """Connect to LiveKit room and listen for video tracks and client control messages."""
        if self.is_running:
            return

        # 1. Initialize or acquire existing DriveGuard AI Engine
        self.engine = await session_manager.get_or_create_session(
            session_id=self.room_name,
            driver_id=self.driver_id,
            experiment_mode=self.experiment_mode,
        )

        # Auto-activate calibration mode if room is designated for calibration
        if self.room_name.startswith("cal_"):
            cal_dto = self.engine.start_calibration(four_zone_mode=False, zone_duration_sec=5.0)
            print(f"[LiveKit Worker] Auto-activated calibration for room '{self.room_name}' (Zone 1: {cal_dto.current_zone})")

        # 2. Mint LiveKit JWT for the AI Worker
        worker_identity = f"ai_worker_{self.room_name}"
        token = (
            api.AccessToken(settings.LIVEKIT_API_KEY, settings.LIVEKIT_API_SECRET)
            .with_identity(worker_identity)
            .with_name("DriveGuard AI Engine")
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=self.room_name,
                    can_subscribe=True,
                    can_publish_data=True,
                    can_publish=False,
                )
            )
            .to_jwt()
        )

        # 3. Setup event listeners
        self._register_event_handlers()

        # 4. Connect to LiveKit Cloud SFU
        logger.info(f"[LiveKit Worker] Connecting to room '{self.room_name}' at {settings.LIVEKIT_URL}...")
        await self.room.connect(settings.LIVEKIT_URL, token)
        self.is_running = True
        logger.info(f"[LiveKit Worker] Successfully joined room '{self.room_name}'.")

        # 5. Start background inference task
        self._inference_task = asyncio.create_task(self._inference_loop())

    async def _safe_close_video_stream(self, stream_to_close: Optional[rtc.VideoStream] = None) -> None:
        """
        Idempotent, race-safe teardown of rtc.VideoStream.
        Ensures underlying C++ FFI handle and queue task are cleanly disposed,
        protects against concurrent close from unsubscription and worker shutdown,
        preserves exactly one cleanup task per stream instance,
        records teardown failures in diagnostic state,
        and uses asyncio.shield to prevent cancellation of the caller from abandoning
        native stream resource disposal.
        """
        target_task: Optional[asyncio.Task] = None
        target_stream: Optional[rtc.VideoStream] = None

        async with self._stream_cleanup_lock:
            target_stream = stream_to_close or self._current_video_stream
            if target_stream is None:
                running_tasks = [t for t in self._stream_close_tasks.values() if not t.done()]
                if running_tasks:
                    try:
                        await asyncio.shield(asyncio.gather(*running_tasks, return_exceptions=True))
                    except asyncio.CancelledError:
                        raise
                return

            if target_stream is self._current_video_stream:
                self._current_video_stream = None

            stream_id = id(target_stream)
            existing_task = self._stream_close_tasks.get(stream_id)
            if existing_task is not None:
                target_task = existing_task
            elif stream_id in self._closed_stream_ids:
                return
            else:
                self._closed_stream_ids.add(stream_id)

                async def _do_close(st: rtc.VideoStream, s_id: int) -> None:
                    try:
                        if hasattr(st, "aclose"):
                            await st.aclose()
                    except asyncio.CancelledError:
                        raise
                    except Exception as err:
                        self._stream_cleanup_error_count += 1
                        self._last_stream_cleanup_error = f"{type(err).__name__}: {err}"
                        logger.error(
                            f"[LiveKit Worker] VideoStream aclose() error on stream {s_id}: {err}",
                            exc_info=True,
                        )
                        raise

                target_task = asyncio.create_task(_do_close(target_stream, stream_id))
                self._stream_close_tasks[stream_id] = target_task
                self._stream_close_task = target_task

        if target_task and not target_task.done():
            try:
                await asyncio.shield(target_task)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    f"[LiveKit Worker] Stream cleanup failed for stream {id(target_stream) if target_stream else 'unknown'}: {exc}"
                )
                raise

    async def _handle_track_subscribed(
        self,
        track: rtc.Track,
        participant: rtc.RemoteParticipant,
    ) -> None:
        """Handle video track subscription with clean teardown of any previous stream."""
        logger.info(
            f"[LiveKit Worker] Video track subscribed from '{participant.identity}' (SID: {track.sid})."
        )
        if self._ingest_task and not self._ingest_task.done():
            self._ingest_task.cancel()
            try:
                await self._ingest_task
            except asyncio.CancelledError:
                pass

        try:
            await self._safe_close_video_stream()
        except Exception as exc:
            logger.warning(f"[LiveKit Worker] Warning during previous stream close on new track subscription: {exc}")
        if not self.is_running:
            return

        self._current_video_stream = rtc.VideoStream(track)
        self._ingest_task = asyncio.create_task(self._ingest_video_stream(self._current_video_stream))

    def _register_event_handlers(self) -> None:
        """Attach room callbacks for video track subscription and data channel communication."""

        @self.room.on("track_subscribed")
        def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ) -> None:
            if track.kind == rtc.TrackKind.KIND_VIDEO:
                asyncio.create_task(self._handle_track_subscribed(track, participant))

        @self.room.on("track_unsubscribed")
        def on_track_unsubscribed(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ) -> None:
            if track.kind == rtc.TrackKind.KIND_VIDEO:
                logger.info(f"[LiveKit Worker] Video track unsubscribed from '{participant.identity}'.")
                if self._ingest_task and not self._ingest_task.done():
                    self._ingest_task.cancel()
                asyncio.create_task(self._safe_close_video_stream())
                asyncio.create_task(self._transition_monitoring_unavailable("VIDEO_TRACK_UNSUBSCRIBED"))

        @self.room.on("data_received")
        def on_data_received(data_packet: rtc.DataPacket) -> None:
            asyncio.create_task(self._handle_client_control_message(data_packet))

        @self.room.on("participant_connected")
        def on_participant_connected(participant: rtc.RemoteParticipant) -> None:
            logger.info(f"[LiveKit Worker] Participant '{participant.identity}' joined room '{self.room_name}'.")
            if self._shutdown_task and not self._shutdown_task.done():
                self._shutdown_task.cancel()
                self._shutdown_task = None
                logger.info(f"[LiveKit Worker] Cancelled pending shutdown for room '{self.room_name}' (participant reconnected).")

            # Dispatch immediate state sync to reconnected participant
            asyncio.create_task(self._send_control_ack({
                "v": 2,
                "type": "STATE_SYNC",
                "session_id": self.room_name,
                "current_state": self._last_state,
                "latest_safety_event": self._latest_safety_event,
                "server_time": time.time(),
            }))

        @self.room.on("participant_disconnected")
        def on_participant_disconnected(participant: rtc.RemoteParticipant) -> None:
            logger.info(f"[LiveKit Worker] Participant '{participant.identity}' left room '{self.room_name}'.")
            # If no remote participants left, schedule a graceful worker shutdown
            if len(self.room.remote_participants) == 0:
                logger.info(f"[LiveKit Worker] No participants remain in '{self.room_name}'. Auto-closing worker in 10s...")
                if self._shutdown_task and not self._shutdown_task.done():
                    self._shutdown_task.cancel()
                self._shutdown_task = asyncio.create_task(self._auto_shutdown_if_empty(delay_sec=10.0))

    async def _auto_shutdown_if_empty(self, delay_sec: float = 10.0) -> None:
        """Shut down the worker and persist session if room remains empty after grace period."""
        try:
            await asyncio.sleep(delay_sec)
            if self.is_running and len(self.room.remote_participants) == 0:
                logger.info(f"[LiveKit Worker] Room '{self.room_name}' is still empty after {delay_sec}s. Terminating worker.")
                await livekit_worker_manager.stop_worker(self.room_name)
        except asyncio.CancelledError:
            pass

    async def _ingest_video_stream(self, stream: rtc.VideoStream) -> None:
        """Asynchronously receive raw WebRTC video frames and feed the latest-frame buffer."""
        try:
            async for event in stream:
                if not self.is_running:
                    break

                frame: rtc.VideoFrame = event.frame
                ingest_mono_ms = time.perf_counter() * 1000.0
                self._frames_received += 1
                self._frame_seq += 1
                current_frame_seq = self._frame_seq

                # Zero-Lag Drop: If queue has a pending frame, discard it immediately
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                        self._frames_replaced += 1
                    except asyncio.QueueEmpty:
                        pass

                await self.frame_queue.put((frame, ingest_mono_ms, current_frame_seq))
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error(f"[LiveKit Worker] Ingest error on room '{self.room_name}': {exc}", exc_info=True)
        finally:
            await self._safe_close_video_stream(stream)

    async def _transition_monitoring_unavailable(self, reason: str) -> None:
        """Immediately transition session state to MONITORING_UNAVAILABLE and emit reliable safety event."""
        if self._last_state == "MONITORING_UNAVAILABLE":
            return
        previous_state = self._last_state
        self._last_state = "MONITORING_UNAVAILABLE"
        event_id = f"evt_{uuid.uuid4().hex[:8]}"
        self._event_seq += 1
        safety_event_payload = {
            "v": 2,
            "type": "SAFETY_EVENT",
            "event_id": event_id,
            "session_id": self.room_name,
            "event_seq": str(self._event_seq),
            "frame_seq": str(self._frame_seq),
            "worker_boot_id": self.worker_boot_id,
            "session_epoch": self.session_epoch,
            "event_type": "STATE_TRANSITION",
            "previous_state": previous_state,
            "new_state": "MONITORING_UNAVAILABLE",
            "timestamp": time.time(),
            "server_wall_epoch_ms": round(time.time() * 1000.0, 1),
            "metrics": {
                "risk_score": 0.0,
                "ear": 0.0,
                "ear_raw": 0.0,
                "perclos": 0.0,
                "pose_zone": "OUT_OF_BOUNDS",
                "reasons": [reason],
            },
        }
        self._latest_safety_event = safety_event_payload
        await self._publish_reliable_safety_event(safety_event_payload)

    async def _inference_loop(self) -> None:
        """Continuous AI inference worker: processes latest frame and publishes telemetry."""
        while self.is_running:
            try:
                try:
                    frame, recv_mono_ms, frame_seq_num = await asyncio.wait_for(
                        self.frame_queue.get(), timeout=2.0
                    )
                except asyncio.TimeoutError:
                    if not self.is_running:
                        break
                    # Frame delivery stall watchdog: if frames were previously arriving and now stopped
                    if self._frames_received > 0 and self._last_state != "MONITORING_UNAVAILABLE":
                        await self._transition_monitoring_unavailable("FRAME_DELIVERY_TIMEOUT")
                    continue

                if not self.is_running:
                    self._frames_aborted += 1
                    break

                self._frames_dequeued += 1
                self._frames_in_flight = 1

                try:
                    t_dequeue_mono_ms = time.perf_counter() * 1000.0
                    frame_queue_wait_ms = max(0.0, t_dequeue_mono_ms - recv_mono_ms)

                    # 1. Convert WebRTC VideoFrame to packed RGB24 bytes
                    t_conv_start = time.perf_counter()
                    rgb_frame = frame.convert(rtc.VideoBufferType.RGB24)
                    t_conv_end = time.perf_counter()
                    frame_convert_ms = (t_conv_end - t_conv_start) * 1000.0

                    # 2. Map directly to NumPy array without disk/JPEG encoding overhead
                    t_prep_start = time.perf_counter()
                    img_rgb = np.frombuffer(rgb_frame.data, dtype=np.uint8).reshape(
                        (rgb_frame.height, rgb_frame.width, 3)
                    )

                    raw_h, raw_w = img_rgb.shape[:2]

                    # 3. Frame dimension normalization (EXP-1A: RGB Pass-Through):
                    # Retains native RGB24 data directly, bypassing redundant RGB->BGR->RGB roundtrips.
                    # np.ascontiguousarray enforces C-contiguity for memory safety and thread dispatch.
                    norm_img = np.ascontiguousarray(normalize_frame_width(img_rgb))
                    norm_h, norm_w = norm_img.shape[:2]
                    t_prep_end = time.perf_counter()
                    frame_preprocess_ms = (t_prep_end - t_prep_start) * 1000.0

                    # Log dimension changes or first frame
                    current_dims = (raw_w, raw_h, norm_w, norm_h)
                    if self._last_logged_dims != current_dims:
                        self._last_logged_dims = current_dims
                        print(
                            f"[LiveKit Worker] Ingest dimensions: raw={raw_w}x{raw_h} -> "
                            f"normalized={norm_w}x{norm_h} (resize_bypassed={raw_w == norm_w})"
                        )

                    # 4. Execute CPU-bound AI pipeline in thread pool to prevent blocking event loop
                    t_infer_start = time.perf_counter()
                    t_infer_start_ms = t_infer_start * 1000.0
                    queue_latency_ms = max(0.0, t_infer_start_ms - recv_mono_ms)

                    loop = asyncio.get_running_loop()
                    self._threads_executing = 1
                    self._coro_cancelled_for_active_frame = False
                    self._active_cfut = self._executor.submit(
                        _timed_process_frame, self.engine, norm_img, recv_mono_ms, t_infer_start
                    )

                    def _safe_dispatch_done(f: concurrent.futures.Future) -> None:
                        if not loop.is_closed():
                            try:
                                loop.call_soon_threadsafe(self._on_inference_thread_done, f)
                            except RuntimeError:
                                pass

                    self._active_cfut.add_done_callback(_safe_dispatch_done)
                    afut = asyncio.wrap_future(self._active_cfut)
                    telemetry_dto, executor_wait_ms, process_frame_ms = await afut
                    t_infer_end = time.perf_counter()
                    t_infer_end_ms = t_infer_end * 1000.0
                    infer_duration_ms = max(0.0, t_infer_end_ms - t_infer_start_ms)
                    server_total_ms = max(0.0, t_infer_end_ms - recv_mono_ms)
                    server_publish_mono_ms = time.perf_counter() * 1000.0
                    server_wall_epoch_ms = time.time() * 1000.0

                    from backend.app.schemas.telemetry import TimingBreakdownDTO
                    telemetry_dto.timing = TimingBreakdownDTO(
                        queue_latency_ms=round(queue_latency_ms, 2),
                        infer_duration_ms=round(infer_duration_ms, 2),
                        server_total_ms=round(server_total_ms, 2),
                        server_ingest_mono_ms=round(recv_mono_ms, 2),
                        server_publish_mono_ms=round(server_publish_mono_ms, 2),
                        server_wall_epoch_ms=round(server_wall_epoch_ms, 1),
                        frame_received_mono_ms=round(recv_mono_ms, 2),
                        frame_queue_wait_ms=round(frame_queue_wait_ms, 2),
                        frame_convert_ms=round(frame_convert_ms, 2),
                        frame_preprocess_ms=round(frame_preprocess_ms, 2),
                        executor_wait_ms=round(executor_wait_ms, 2),
                        process_frame_ms=round(process_frame_ms, 2),
                        publish_ms=round(self._last_publish_duration_ms, 2) if self._last_publish_duration_ms is not None else None,
                        frame_age_ms=None,
                    )
                    telemetry_dto.frame_seq = str(frame_seq_num)
                    telemetry_dto.worker_boot_id = self.worker_boot_id
                    telemetry_dto.session_epoch = self.session_epoch

                    # 5. Important Safety Event Detection (Reliable Delivery Trigger)
                    current_state_str = telemetry_dto.state.value
                    if current_state_str != self._last_state:
                        import uuid
                        event_id = f"evt_{uuid.uuid4().hex[:8]}"
                        self._event_seq += 1
                        safety_event_payload = {
                            "v": 2,
                            "type": "SAFETY_EVENT",
                            "event_id": event_id,
                            "session_id": self.room_name,
                            "event_seq": str(self._event_seq),
                            "frame_seq": str(frame_seq_num),
                            "worker_boot_id": self.worker_boot_id,
                            "session_epoch": self.session_epoch,
                            "event_type": "STATE_TRANSITION",
                            "previous_state": self._last_state,
                            "new_state": current_state_str,
                            "timestamp": time.time(),
                            "server_wall_epoch_ms": round(server_wall_epoch_ms, 1),
                            "metrics": {
                                "risk_score": telemetry_dto.risk_score,
                                "ear": telemetry_dto.eye_metrics.ear_normalized,
                                "ear_raw": telemetry_dto.eye_metrics.ear_raw,
                                "perclos": telemetry_dto.eye_metrics.perclos_pct,
                                "pose_zone": telemetry_dto.pose_zone.value,
                                "reasons": telemetry_dto.reason_codes,
                            },
                        }
                        self._last_state = current_state_str
                        self._latest_safety_event = safety_event_payload
                        asyncio.create_task(self._publish_reliable_safety_event(safety_event_payload))

                    # 6. Periodic terminal logging (every ~15 frames)
                    if telemetry_dto.frame_index % 15 == 0 or telemetry_dto.frame_index == 1:
                        print(
                            f"[LiveKit AI] Frame #{telemetry_dto.frame_index} (seq={frame_seq_num}): "
                            f"State={telemetry_dto.state.value} | "
                            f"EAR={telemetry_dto.eye_metrics.ear_raw:.3f} | "
                            f"Zone={telemetry_dto.pose_zone.value} | "
                            f"Queue={queue_latency_ms:.1f}ms | "
                            f"Infer={infer_duration_ms:.1f}ms | "
                            f"ServerTotal={server_total_ms:.1f}ms"
                        )

                    # 7. Construct and publish telemetry payload over WebRTC Data Channel
                    cal_status = self.engine.get_calibration_status()
                    cal_data = None
                    if cal_status is not None:
                        cal_data = cal_status.model_dump()
                        if cal_status.is_completed and not self._calibration_completion_sent:
                            self._calibration_completion_sent = True
                            if cal_status.rejection_reason:
                                print(
                                    f"[LiveKit AI] >>> Calibration REJECTED for driver '{self.driver_id}': "
                                    f"{cal_status.rejection_reason}. Sending CALIBRATION_REJECTED event."
                                )
                                asyncio.create_task(
                                    self._send_control_ack({
                                        "type": "CALIBRATION_REJECTED",
                                        "calibration": cal_data,
                                        "rejection_reason": cal_status.rejection_reason,
                                    })
                                )
                            else:
                                profile_dto = self.engine.export_profile_dto()
                                profile_store.save_profile(profile_dto)
                                print(
                                    f"[LiveKit AI] >>> Calibration complete for driver '{self.driver_id}'! "
                                    f"Center EAR={profile_dto.center_ear_median:.3f}. Sending CALIBRATION_COMPLETED event."
                                )
                                asyncio.create_task(
                                    self._send_control_ack({
                                        "type": "CALIBRATION_COMPLETED",
                                        "calibration": cal_data,
                                        "profile": profile_dto.model_dump(),
                                    })
                                )

                    response_payload = self._build_telemetry_payload(
                        telemetry_dto,
                        frame_seq_num=frame_seq_num,
                        cal_data=cal_data,
                        server_total_ms=server_total_ms,
                    )

                    self._frames_processed += 1
                    self._frames_in_flight = 0
                    self._active_cfut = None

                    # Publish lossy/unreliable for telemetry (fastest WebRTC UDP delivery)
                    try:
                        t_pub_start = time.perf_counter()
                        payload_bytes = json.dumps(response_payload).encode("utf-8")
                        await self.room.local_participant.publish_data(payload_bytes, reliable=False)
                        self._last_publish_duration_ms = (time.perf_counter() - t_pub_start) * 1000.0
                    except Exception as pub_exc:
                        logger.warning(f"[LiveKit Worker] Non-fatal telemetry publication failure: {pub_exc}")

                except asyncio.CancelledError:
                    self._coro_cancelled_for_active_frame = True
                    if self._active_cfut and not self._active_cfut.done():
                        # The OS worker thread is STILL physically running in the background.
                        # Retain frames_in_flight=1 so get_frame_accounting_snapshot()
                        # never falsely reports 0 in-flight while the thread is active.
                        # _on_inference_thread_done will reconcile when the thread terminates.
                        logger.info(
                            f"[LiveKit Worker] Coroutine cancelled while inference thread is still executing. "
                            f"Retaining frames_in_flight=1 until thread completion."
                        )
                    else:
                        if self._frames_in_flight > 0:
                            self._frames_aborted += self._frames_in_flight
                            self._frames_in_flight = 0
                        self._active_cfut = None
                    raise
                except Exception as exc:
                    self._frames_errored += 1
                    self._frames_in_flight = 0
                    self._active_cfut = None
                    logger.error(f"[LiveKit Worker] Inference execution error: {exc}", exc_info=True)
                    if self._last_state != "MONITORING_UNAVAILABLE":
                        asyncio.create_task(
                            self._transition_monitoring_unavailable(f"INFERENCE_ERROR: {type(exc).__name__}")
                        )
                finally:
                    if not (self._coro_cancelled_for_active_frame and self._active_cfut and not self._active_cfut.done()):
                        self._frames_in_flight = 0

            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._frames_errored += 1
                logger.error(f"[LiveKit Worker] Inference loop error: {exc}", exc_info=True)
                await asyncio.sleep(0.01)

    def _consume_echo_ts(self) -> Optional[float]:
        """Atomically read and clear the latest client ping timestamp."""
        echo_ts = self._last_client_ping_ts
        self._last_client_ping_ts = None
        return echo_ts

    def _build_telemetry_payload(
        self,
        telemetry_dto: Any,
        frame_seq_num: int = 0,
        cal_data: Optional[Dict[str, Any]] = None,
        server_total_ms: float = 0.0,
    ) -> Dict[str, Any]:
        """Build telemetry message payload and clear last client ping timestamp."""
        echo_ts = self._consume_echo_ts()
        self._event_seq += 1

        resolved_frame_seq = str(frame_seq_num or getattr(telemetry_dto, "frame_index", 0))
        if hasattr(telemetry_dto, "event_seq"):
            telemetry_dto.event_seq = str(self._event_seq)

        return {
            "v": 2,
            "type": "TELEMETRY",
            "session_id": self.room_name,
            "seq": resolved_frame_seq,
            "frame_seq": resolved_frame_seq,
            "event_seq": str(self._event_seq),
            "worker_boot_id": self.worker_boot_id,
            "session_epoch": self.session_epoch,
            "timestamp": time.time(),
            "server_wall_epoch_ms": round(time.time() * 1000.0, 1),
            "telemetry": telemetry_dto.model_dump() if hasattr(telemetry_dto, "model_dump") else telemetry_dto,
            "calibration": cal_data,
            "echo_client_ts": echo_ts,
            "network_roundtrip_ms": round(server_total_ms, 1),
        }

    async def _handle_client_control_message(self, data_packet: rtc.DataPacket) -> None:
        """Handle control actions from frontend client over WebRTC Data Channel."""
        try:
            raw_text = data_packet.data.decode("utf-8")
            msg = json.loads(raw_text)
            action = msg.get("action")

            if not action:
                return

            print(f"[LiveKit Worker] >>> Control action received: '{action}'")
            logger.info(f"[LiveKit Worker] Control action received: '{action}'")

            if action in ("EVENT_ACK", "ACK_EVENT"):
                event_id = msg.get("event_id")
                if event_id:
                    acked = self._unacknowledged_events.pop(event_id, None)
                    if acked:
                        ack_rtt_ms = (time.perf_counter() - acked["sent_at"]) * 1000.0
                        logger.info(f"[LiveKit Worker] Safety event '{event_id}' ACK confirmed in {ack_rtt_ms:.1f}ms.")

            elif action == "SYNC_STATE":
                unacked = [entry["payload"] for entry in self._unacknowledged_events.values()]
                await self._send_control_ack({
                    "v": 2,
                    "type": "STATE_SYNC",
                    "session_id": self.room_name,
                    "current_state": self._last_state,
                    "latest_safety_event": self._latest_safety_event,
                    "unacknowledged_events": unacked,
                    "server_time": time.time(),
                })

            elif action == "PING":
                client_ts = msg.get("client_ts")
                ping_id = msg.get("ping_id")
                if client_ts is not None:
                    self._last_client_ping_ts = client_ts
                await self._send_control_ack({
                    "v": 2,
                    "type": "PONG",
                    "client_ts": client_ts,
                    "ping_id": ping_id,
                    "server_time": time.time(),
                    "queue_depth": self.frame_queue.qsize(),
                })

            elif not self.engine:
                logger.warning(f"[LiveKit Worker] Cannot execute '{action}' because engine is not initialized.")
                return

            elif action == "START_CALIBRATION":
                self._calibration_completion_sent = False
                four_zone = msg.get("four_zone", True)
                three_zone = msg.get("three_zone", False)
                duration = float(msg.get("zone_duration_sec", 5.0))
                include_blink = bool(msg.get("include_blink_baseline", False))
                blink_duration = float(msg.get("blink_baseline_duration_sec", 120.0))
                cal_dto = self.engine.start_calibration(
                    four_zone_mode=four_zone,
                    three_zone_mode=three_zone,
                    zone_duration_sec=duration,
                    include_blink_baseline=include_blink,
                    blink_baseline_duration_sec=blink_duration,
                )
                print(
                    f"[LiveKit Worker] Calibration started: zone={cal_dto.current_zone}, "
                    f"phase={cal_dto.phase_index}/{cal_dto.total_phases}, include_blink={include_blink}"
                )
                await self._send_control_ack({"type": "CALIBRATION_STARTED", "calibration": cal_dto.model_dump()})

            elif action == "RECALIBRATE_ZONE":
                self._calibration_completion_sent = False
                target_zone = msg.get("zone", "CENTER")
                duration = float(msg.get("zone_duration_sec", 5.0))
                blink_duration = float(msg.get("blink_baseline_duration_sec", 120.0))
                cal_dto = self.engine.start_calibration(
                    target_zone=target_zone,
                    zone_duration_sec=duration,
                    blink_baseline_duration_sec=blink_duration,
                )
                print(f"[LiveKit Worker] Individual zone recalibration started for zone '{target_zone}'")
                await self._send_control_ack({"type": "CALIBRATION_STARTED", "calibration": cal_dto.model_dump()})

            elif action == "CANCEL_CALIBRATION":
                self.engine.cancel_calibration()
                await self._send_control_ack({"type": "CALIBRATION_CANCELLED"})

            elif action == "SKIP_CALIBRATION_STEP":
                cal_dto = self.engine.skip_calibration_step()
                await self._send_control_ack({
                    "type": "CALIBRATION_STEP_SKIPPED",
                    "calibration": cal_dto.model_dump() if cal_dto else None,
                })

            elif action == "SWITCH_EXPERIMENT":
                mode_str = msg.get("mode")
                if mode_str:
                    new_mode = ExperimentMode(mode_str)
                    self.engine.set_experiment_mode(new_mode)
                    await self._send_control_ack({"type": "EXPERIMENT_SWITCHED", "mode": new_mode.value})

            elif action == "SAVE_PROFILE":
                profile_dto = self.engine.export_profile_dto()
                profile_store.save_profile(profile_dto)
                await self._send_control_ack({"type": "PROFILE_SAVED", "driver_id": profile_dto.driver_id})

        except Exception as exc:
            logger.error(f"[LiveKit Worker] Control message parsing error: {exc}", exc_info=True)

    async def _publish_reliable_safety_event(self, event_payload: dict) -> None:
        """Publish important safety event with reliable DataChannel delivery and track ACK."""
        event_id = event_payload["event_id"]
        self._unacknowledged_events[event_id] = {
            "payload": event_payload,
            "retries": 0,
            "sent_at": time.perf_counter(),
        }
        logger.info(
            f"[LiveKit Worker] Emitting reliable safety event '{event_id}' "
            f"({event_payload['previous_state']} -> {event_payload['new_state']})."
        )
        try:
            payload_bytes = json.dumps(event_payload).encode("utf-8")
            await self.room.local_participant.publish_data(payload_bytes, reliable=True)
        except Exception as exc:
            logger.error(f"[LiveKit Worker] Failed to publish initial safety event '{event_id}': {exc}")

        # Non-blocking retry task
        asyncio.create_task(self._retry_unacknowledged_event(event_id))

    async def _retry_unacknowledged_event(self, event_id: str) -> None:
        """Retry unacknowledged safety events with bounded backoff without blocking inference."""
        for attempt in range(1, 4):
            await asyncio.sleep(0.4 * attempt)
            if not self.is_running:
                break
            entry = self._unacknowledged_events.get(event_id)
            if not entry:
                # Confirmed ACK
                return
            entry["retries"] += 1
            logger.warning(
                f"[LiveKit Worker] Retrying unacknowledged event '{event_id}' (attempt {entry['retries']}/3)..."
            )
            try:
                payload_bytes = json.dumps(entry["payload"]).encode("utf-8")
                await self.room.local_participant.publish_data(payload_bytes, reliable=True)
            except Exception as exc:
                logger.warning(f"[LiveKit Worker] Retry {attempt} failed: {exc}")

        entry = self._unacknowledged_events.get(event_id)
        if entry and self.is_running:
            entry["exhausted"] = True
            logger.warning(
                f"[LiveKit Worker] Event '{event_id}' unacknowledged after 3 retries (client may be disconnected)."
            )

    async def _send_control_ack(self, payload: dict) -> None:
        """Send reliable ACK or state change confirmation back to client."""
        try:
            data = json.dumps(payload).encode("utf-8")
            await self.room.local_participant.publish_data(data, reliable=True)
        except Exception as exc:
            logger.warning(f"[LiveKit Worker] Failed to send control ACK: {exc}")

    async def stop(self) -> None:
        """Cleanly disconnect the worker from the LiveKit room, terminate tasks, and flush session summaries."""
        async with self._stop_lock:
            if self._is_stopped:
                return
            self._is_stopped = True
            if not self.is_running and not self.room:
                return
            self.is_running = False

        shutdown_task = self._shutdown_task
        self._shutdown_task = None
        if shutdown_task and not shutdown_task.done() and shutdown_task is not asyncio.current_task():
            shutdown_task.cancel()

        if self._ingest_task and not self._ingest_task.done():
            self._ingest_task.cancel()
            try:
                await self._ingest_task
            except asyncio.CancelledError:
                pass
            self._ingest_task = None

        try:
            await self._safe_close_video_stream()
        except Exception as exc:
            logger.warning(f"[LiveKit Worker] Stream cleanup error during worker stop(): {exc}")

        if self._inference_task and not self._inference_task.done():
            self._inference_task.cancel()
            try:
                await self._inference_task
            except asyncio.CancelledError:
                pass
            self._inference_task = None

        # Drain remaining frames from queue into aborted count
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
                self._frames_aborted += 1
            except asyncio.QueueEmpty:
                break

        if self._active_cfut and not self._active_cfut.done():
            self._coro_cancelled_for_active_frame = True
            logger.info(
                f"[LiveKit Worker] stop() called while inference thread active. "
                f"Preserving in-flight frame ({self._frames_in_flight}) until background thread terminates."
            )
        else:
            if self._frames_in_flight > 0:
                self._frames_aborted += self._frames_in_flight
                self._frames_in_flight = 0

        if self._executor:
            self._executor.shutdown(wait=False, cancel_futures=True)

        try:
            await self.room.disconnect()
        except Exception as exc:
            logger.warning(f"[LiveKit Worker] Disconnect warning for '{self.room_name}': {exc}")

        # Terminate active session in session_manager to release in-memory state and auto-persist profile
        try:
            summary = await session_manager.terminate_session(self.room_name)
            if self.engine is not None and not getattr(self.engine, "_is_closed", False):
                try:
                    self.engine.close()
                except Exception:
                    pass
            self.engine = None
            if summary:
                logger.info(
                    f"[LiveKit Worker] Session '{self.room_name}' terminated cleanly. "
                    f"Frames processed: {summary.get('total_frames_processed', 0)}"
                )
        except Exception as exc:
            logger.error(f"[LiveKit Worker] Error terminating session in session_manager: {exc}")

        logger.info(f"[LiveKit Worker] Worker cleanly disconnected and released for room '{self.room_name}'.")


class LiveKitWorkerManager:
    """Singleton manager tracking and dispatching worker sessions per room."""

    def __init__(self) -> None:
        self._workers: Dict[str, LiveKitAIWorkerSession] = {}
        self._lock = asyncio.Lock()

    async def ensure_worker_running(
        self,
        room_name: str,
        driver_id: str = "driver_default",
        experiment_mode: Optional[str] = None,
    ) -> LiveKitAIWorkerSession:
        """Ensure an active AI worker participant exists for the given room."""
        async with self._lock:
            existing = self._workers.get(room_name)
            if existing and existing.is_running:
                return existing

            worker = LiveKitAIWorkerSession(
                room_name=room_name,
                driver_id=driver_id,
                experiment_mode=experiment_mode,
            )
            # Start worker in background
            asyncio.create_task(worker.start())
            self._workers[room_name] = worker
            return worker

    async def stop_worker(self, room_name: str) -> bool:
        """Stop and remove a worker session."""
        async with self._lock:
            worker = self._workers.pop(room_name, None)
            if worker:
                await worker.stop()
                return True
            return False

    def list_active_workers(self) -> list[str]:
        """Return list of active room names with running workers."""
        return [k for k, v in self._workers.items() if v.is_running]


livekit_worker_manager = LiveKitWorkerManager()
