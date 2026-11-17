# DriveGuard — Phase-by-Phase Development Plan

**Project:** Personalized AI-Based Driver Drowsiness Detection and Early Warning System  
**Repository:** `michaelumoize-lab/DriveGuard`  
**Primary Direction:** Local-first AI application with an optional web dashboard  
**Development Target:** Windows laptop, initially using the HP EliteBook 840 G4 webcam, with phone-camera and Arducam support through interchangeable camera adapters.  
**Active Branch:** `core-engine`

---

## 1. Project Vision and Non-Negotiable Principles

DriveGuard monitors a driver's face and behaviour in real time, estimates signs of drowsiness, and issues timely warnings through a local interface and optional ESP32 peripherals.

Its key differentiator is **personalized calibration**: it learns a driver's normal facial measurements and behaviour, while preserving conservative safety limits.

### Architectural Principles

1. **Local-first:** Camera capture, AI inference, calibration, alert decisions, and core session storage run locally on the user's computer.
2. **No mandatory account:** Users must be able to start monitoring without registering for a website or logging in.
3. **Offline-capable core:** Internet access is not required for core detection once dependencies and models are installed.
4. **Dashboard is optional:** Retain the existing Next.js dashboard, but do not make it responsible for the critical camera-to-AI path.
5. **Camera independence:** Support the built-in webcam, a phone camera, and an external UVC USB camera through a common interface.
6. **Hardware independence:** Detection must continue if the ESP32 is disconnected.
7. **Incremental development:** Implement, test, and document each phase before moving on.
8. **Safety-aware design:** DriveGuard is a prototype driver-assistance and early-warning system, not a certified safety device or a substitute for sleep, safe driving, or vehicle safety systems.
9. **Privacy by default:** Do not upload or retain raw video by default. Make any recording explicit and user-controlled.
10. **Measurable performance:** Record capture FPS, inference FPS, frame age, processing latency, dropped frames, and detection confidence.

---

## 2. Target Architecture

```text
CAMERA SOURCES
  ├── Laptop built-in webcam (HP EliteBook 840 G4)
  ├── Phone camera (USB or local-network transport)
  └── External USB UVC camera (e.g. Arducam IMX462)
             │
             ▼
CAMERA ADAPTER / CAPTURE SERVICE
  - Source selection (DirectShow / MSMF)
  - Monotonic frame timestamps
  - Resolution and FPS configuration
  - Reconnect and error handling
  - Bounded latest-frame buffer (maxlen=1 drop-oldest)
             │
             ▼
VISION PIPELINE
  - Frame validation and preprocessing
  - Face detection / MediaPipe 478 Face Landmarker
  - Face tracking, quality checks, and face-loss grace FSM
             │
             ▼
FEATURE EXTRACTION
  - Eye Aspect Ratio (EAR) & eye closure states
  - Sub-frame blink kinematics & blink duration
  - Cumulative rolling PERCLOS
  - Mouth Aspect Ratio (MAR) & yawn events
  - 6-DoF head pose (yaw, pitch, roll) & nodding patterns
  - Visibility / confidence signals
             │
             ▼
PERSONALIZATION + TEMPORAL ENGINE
  - 4-Zone baseline calibration (Center, Left Mirror, Right Mirror, Dashboard)
  - Driver-specific normalization (robust median / MAD)
  - Temporal accumulation windows
  - Confidence-aware fusion & bounded adaptive thresholds
             │
             ▼
ALERT DECISION ENGINE
  - 4-State FSM: NORMAL / ATTENTION / DROWSINESS_WARNING / CRITICAL_WARNING / INSUFFICIENT_DATA
  - Hysteresis, persistence, and cooldown logic
  - Explainable event reason codes
             │
       ┌─────┴──────────────────┐
       │                        │
       ▼                        ▼
LOCAL USER INTERFACE       ESP32 ADAPTER
  - OpenCV live HUD          - Serial connection (USB COM)
  - Standalone desktop UI    - OLED status display
                             - Buzzer / RGB LED / Vibration
       │                        │
       ▼                        ▼
LOCAL DATABASE (SQLite)    HARDWARE FAIL-SAFE
  - Driver profiles          - Asynchronous worker
  - Calibration records      - Reconnection watchdog
  - Session analytics
       │
       ▼
OPTIONAL LOCAL API (FastAPI)
  - Localhost-bound HTTP & WebSocket telemetry
       │
       ▼
EXISTING WEB DASHBOARD (Next.js)
  - Visual telemetry & analytics viewer (optional consumer, NOT video streamer)
       │
       ▼
OPTIONAL CLOUD SERVICES (future, never required for core detection)
```

### Distinction: Local-First vs. Browser-Only
The primary camera capture and inference path runs inside the local Python engine. The browser dashboard acts strictly as an optional display and management interface consuming local telemetry via WebSocket/HTTP, never capturing or streaming raw camera frames over the network. If an optional local MJPEG preview stream is enabled for debugging or visual calibration, it must be bound strictly to the localhost loopback interface (`127.0.0.1`), require explicit local user activation, enforce access controls, and strictly prohibit transmitting raw frames outside the local machine boundary.

### Proposed Modular Directory Structure

```text
DriveGuard/
├── backend/
│   ├── app/
│   │   ├── camera/
│   │   │   ├── base.py                 # Abstract Camera Base Class
│   │   │   ├── opencv_camera.py        # OpenCV DirectShow / MSMF Adapter
│   │   │   ├── phone_camera.py         # Phone Camera Adapter (USB / IP)
│   │   │   ├── camera_manager.py       # Camera Device Discovery & Switching
│   │   │   ├── frame_buffer.py         # Thread-safe Bounded Latest-Frame Buffer
│   │   │   └── camera_config.py        # Resolution, FPS, Backend Settings
│   │   ├── vision/
│   │   │   ├── face_landmarker.py      # MediaPipe Face Landmarker Wrapper
│   │   │   ├── face_quality.py         # Lighting, Blur, Occlusion Checks
│   │   │   ├── eye_tracking.py         # Iris & Eye Contour Tracking
│   │   │   ├── mouth_tracking.py       # Lip Landmarks & Yawn Geometry
│   │   │   └── head_pose.py            # 6-DoF SolvePnP Pose & Zone Classifier
│   │   ├── features/
│   │   │   ├── ear.py                  # Left & Right Eye Aspect Ratio
│   │   │   ├── blink.py                # Kinematics, Duration, Reopening Rate
│   │   │   ├── perclos.py              # Rolling Time-Window PERCLOS
│   │   │   ├── mar.py                  # Mouth Aspect Ratio
│   │   │   └── feature_extractor.py    # Unified Timestamped Feature Aggregator
│   │   ├── intelligence/
│   │   │   ├── temporal_analyzer.py    # Time-Window Event Accumulator
│   │   │   ├── confidence.py           # Multi-Signal Reliability Scoring
│   │   │   ├── decision_engine.py      # Hysteresis & State Transition Rules
│   │   │   └── drowsiness_engine.py    # Multi-Indicator Drowsiness Arbiter
│   │   ├── personalization/
│   │   │   ├── calibration.py          # Guided 4-Zone Calibration Wizard
│   │   │   ├── driver_profile.py       # PDBP Schema & Serialization
│   │   │   └── adaptive_thresholds.py  # Bounded Safe Adaptive Thresholds
│   │   ├── alerts/
│   │   │   ├── alert_levels.py         # Alert Enums & Severity Mapping
│   │   │   └── alert_manager.py        # Sound, Visual, and Cooldown Dispatcher
│   │   ├── hardware/
│   │   │   ├── esp32_controller.py     # Threaded Serial Controller & Heartbeat
│   │   │   └── serial_protocol.py      # Binary/JSON Frame Protocol Definition
│   │   ├── storage/
│   │   │   ├── database.py             # SQLite Connection & Session Maker
│   │   │   ├── models.py               # SQLAlchemy ORM Models
│   │   │   └── repositories/           # Profile, Session, Alert Repositories
│   │   ├── monitoring/
│   │   │   ├── metrics.py              # Capture FPS, Inference FPS, Frame Latency
│   │   │   └── session_logger.py       # Local Run Structured Diagnostics
│   │   ├── api/                        # Localhost API for Dashboard Integration
│   │   │   ├── routes/
│   │   │   └── websocket.py
│   │   ├── config/
│   │   └── main.py                     # Local Application Entrypoint
│   ├── tests/
│   ├── scripts/
│   └── requirements.txt
├── frontend/                           # Preserved Next.js Dashboard
├── esp32/
│   ├── src/
│   └── README.md
├── docs/
└── README.md
```

---

## 3. Phase-by-Phase Roadmap

### [x] Phase 0 — Repository Audit, Scope and Baseline
- **Goal:** Understand the current project baseline without breaking existing functionality.
- **Tasks:**
  - [x] Inspect complete repository and current Git status.
  - [x] Identify existing frontend (`Next.js 16`, `LiveKit`), backend (`FastAPI`, `MediaPipe`), API routes, and database.
  - [x] Audit why the browser-streaming prototype had latency (browser frame encoding, WebRTC jitter buffer, network serialization).
  - [x] Create dedicated development branch `core-engine` branched from `main`.
  - [x] Preserve existing dashboard and ensure zero code deletion.
  - [x] Document target modular architecture.
- **Deliverables:** Architecture decision record (ADR) and Phase Plan document.
- **Acceptance Criteria:** Existing dashboard preserved; git branch `core-engine` active.

---

### [x] Phase 1 — Development Environment and Camera Capture Foundation
- **Goal:** Reliably capture live frames locally using the laptop webcam (HP EliteBook 840 G4).
- **Tasks:**
  - [x] Set up dedicated Python 3.11 virtual environment (`py -3.11 -m venv .venv311`).
  - [x] Install local runtime requirements (`opencv-python` with GUI support, `numpy`).
  - [x] Build camera configuration module `camera_config.py` (index, resolution, target FPS, backend `CAP_DSHOW` / `CAP_MSMF`).
  - [x] Implement `OpenCVCamera` adapter in `backend/app/camera/opencv_camera.py`:
    - Interface: `open()`, `read()`, `get_status()`, `set_property()`, `close()`.
    - Dedicated background capture thread continuously draining the OS frame queue.
    - Bounded 1-slot buffer (`deque(maxlen=1)`) to eliminate frame backlog and latency.
    - Monotonic capture timestamps on every frame.
    - Automatic camera reconnect logic.
  - [x] Build standalone camera benchmark runner (`scripts/run_camera_test.py`):
    - Opens HP webcam with DirectShow backend.
    - Displays OpenCV preview window with real-time HUD (Negotiated Resolution, Capture FPS, Consumer Processing FPS, Frame Age).
    - Measures sustained FPS over 30 seconds.
    - Handles clean exit on `ESC` or `q` without process hang.
- **Deliverables:** Minimal local camera module and FPS verification runner.
- **Acceptance Criteria:**
  - Camera opens, reads, and closes reliably without freezing.
  - Live preview runs smoothly at native webcam FPS (~30 FPS).
  - Frame age remains < 35 ms (zero buffering backlog).

---

### [x] Phase 2 — Phone Camera Integration
- **Goal:** Enable a mobile phone to act as an alternative camera without altering downstream vision modules.
- **Tasks:**
  - [x] Evaluate USB UVC transport vs. Local Network RTSP/HTTP MJPEG stream.
  - [x] Implement `PhoneCamera` adapter matching the common camera interface.
  - [x] Measure Wi-Fi jitter, frame arrival variance, and latency on local network.
  - [x] Add transport reconnection state machine for handling network drops.
  - [x] Provide clear setup guide for phone mounting, power, and stream URL configuration.
  - [x] Implement `create_camera()` factory in `camera_manager.py` supporting both webcams and phone URLs.
- **Deliverables:** `phone_camera.py` adapter, `camera_manager.py`, and `docs/PHONE_CAMERA_SETUP_GUIDE.md`.
- **Acceptance Criteria:** The vision pipeline consumes phone frames identically to the laptop webcam.

---

### [x] Phase 3 — Camera Abstraction, Frame Quality, and Performance
- **Goal:** Standardize frame packets and make video stream health fully observable.
- **Tasks:**
  - [x] Implement `FramePacket` dataclass (BGR frame, monotonic timestamp, sequence index, source identifier).
  - [x] Implement `CameraManager` providing dynamic camera discovery, selection, and hot-swapping.
  - [x] Add frame quality diagnostics: blur detection (Laplacian variance), low-light / underexposure check, and dropped frame counters (`FrameQualityAnalyzer`).
  - [x] Establish strict memory bounds and thread ownership to prevent leaks (`FrameBuffer` 1-slot latest-frame strategy).
- **Deliverables:** `camera_manager.py`, `frame_buffer.py`, `frame_quality.py`, and diagnostic metrics.
- **Acceptance Criteria:** Hot-swapping camera sources requires zero modifications to downstream AI code; optical frame health is observable.

---

### [x] Phase 4 — Face Detection and Facial Landmark Tracking
- **Goal:** Reliably locate the driver's face, contours, and 3D landmarks in real time.
- **Tasks:**
  - [x] Wrap MediaPipe Face Landmarker (`weights/face_landmarker.task`) with dedicated lifecycle initialization.
  - [x] Run inference in live stream mode with monotonic timestamps.
  - [x] Extract 478 3D landmarks (face mesh, eye contours, irises).
  - [x] Implement face presence validation, detection confidence assessment, and face-loss grace FSM.
  - [x] Render debug overlay with toggles for mesh wireframes and eye landmarks.
- **Deliverables:** `backend/app/vision/face_landmarker.py` and visual tracking preview.
- **Acceptance Criteria:**
  - Face loss triggers `INSUFFICIENT_DATA` rather than crashing or predicting false drowsiness.
  - Inference runs at >= 30 FPS on CPU/GPU (benchmarked at 25.2 ms / ~40 FPS on CPU).

---

### [x] Phase 5 — Feature Extraction
- **Goal:** Convert raw facial landmarks into timestamped, explainable physiological measurements.
- **Tasks:**
  - [x] **EAR:** Independent left and right Eye Aspect Ratio calculation using Euclidean 3D distances with yaw perspective weighting and pitch compensation.
  - [x] **Blink Kinematics:** Temporal blink event detector recording start timestamp, duration (ms), sub-frame linear crossing interpolation, closing velocity, reopening velocity, and RCR.
  - [x] **PERCLOS:** Cumulative percentage of eye closure over a sliding 60-second window (excluding lost-face intervals) with observation coverage tracking.
  - [x] **MAR & Yawning:** Mouth Aspect Ratio calculation using 3 inner-lip pairs and sustained yawn detector (rejecting normal conversation speech < 1.5s).
  - [x] **Head Pose (6-DoF):** SolvePnP projection using canonical 3D facial points estimating Pitch (nodding), Yaw (mirror glances), Roll, and classifying cockpit gaze zones (CENTER, LEFT_MIRROR, RIGHT_MIRROR, DASHBOARD, OUT_OF_BOUNDS).
  - [x] **Unified Feature Extractor:** Aggregate all measurements into a timestamped, immutable `DriverFeatures` record.
  - [x] **Diagnostic HUD Integration:** Display real-time EAR, PERCLOS, MAR, Pose zone, and blink counts on the diagnostic HUD in `scripts/run_camera_test.py`.
- **Deliverables:** `backend/app/features/` module (`ear.py`, `head_pose.py`, `mar.py`, `blink_detector.py`, `perclos.py`, `feature_extractor.py`, `__init__.py`) and comprehensive test suite `backend/tests/test_features.py`.
- **Acceptance Criteria:**
  - All feature modules have unit tests verifying mathematical correctness against synthetic landmark fixtures (18/18 tests passing, 74/74 across repository).
  - Live pipeline benchmarks at native 30 FPS ingestion with zero memory leaks.

---

### [x] Phase 6 — Temporal Analysis and Baseline Drowsiness Engine
- **Goal:** Fuse multiple temporal features into robust, explainable drowsiness assessments.
- **Tasks:**
  - [x] Implement rolling temporal accumulators (`TemporalAnalyzer` tracking 10s, 30s, 60s windows).
  - [x] Define multi-indicator decision policy:
    - Primary urgent trigger: Sustained eye closure (eyes closed >= 1.5s immediately triggers `CRITICAL_WARNING`).
    - Urgent nodding microsleep trigger: Head pitch slump (< -15°) during eye closure (>= 800ms) triggers `CRITICAL_WARNING`.
    - Severe PERCLOS crisis trigger: Cumulative PERCLOS >= 30% triggers `CRITICAL_WARNING`.
    - Cumulative fatigue triggers: Elevated PERCLOS (>= 15%), sluggish/prolonged blinks (mean duration >= 350ms or 3+ prolonged blinks), frequent yawns (2+ in 60s coupled with fatigue indicators).
    - Early warnings: Extended closure (500ms - 1500ms), authentic yawn, mild PERCLOS (8% - 15%).
  - [x] Implement 5 output states: `NORMAL`, `ATTENTION`, `DROWSINESS_WARNING`, `CRITICAL_WARNING`, `INSUFFICIENT_DATA`.
  - [x] Add explainable reason codes attached to every decision (`ExplainableReason` with underlying metric values and safety thresholds).
  - [x] Automotive context awareness: Suppress false distraction alarms during transient mirror glances (<= 3.0s) and instrument cluster glances (<= 2.0s).
  - [x] Asymmetric hysteresis and recovery cooldown: Fast escalation for safety; cooldown required for de-escalation (2.0s from critical, 1.5s from warning, 1.0s from attention) to eliminate alarm chattering.
  - [x] Diagnostic HUD integration: Real-time Drowsiness State badge, severity, explainable reason code, and dynamic colored status viewport border in `scripts/run_camera_test.py`.
- **Deliverables:** `backend/app/intelligence/` package (`reason_codes.py`, `confidence.py`, `temporal_analyzer.py`, `decision_engine.py`, `drowsiness_engine.py`, `__init__.py`) and test suite `backend/tests/test_drowsiness_engine.py`.
- **Acceptance Criteria:** Zero single-frame false alarms; decisions require temporal persistence; 15/15 unit tests passing (149/149 across entire repository).

---

### [ ] Phase 7 — Personalized Driver Calibration
- **Goal:** Learn individual facial baselines and normal driving postures while enforcing conservative safety bounds.
- **Tasks:**
  - [ ] Implement interactive 4-zone calibration protocol:
    1. Looking straight ahead (center baseline EAR, neutral pitch/yaw).
    2. Looking at left mirror (yaw left baseline).
    3. Looking at right mirror (yaw right baseline).
    4. Looking at dashboard / instruments (pitch down baseline).
  - [ ] Compute robust statistical baselines: Median and Median Absolute Deviation (MAD) for EAR and blink duration.
  - [ ] Implement bounded adaptive thresholds:
    - Lower and upper safety clamps: Personalization can adapt threshold sensitivity, but *never* allow sustained eye closure > 1.5s to go unflagged.
  - [ ] Implement calibration quality score (stability, lighting, landmark confidence).
  - [ ] Persist driver profile to local disk.
- **Deliverables:** `personalization/` module and calibration wizard CLI/GUI.
- **Acceptance Criteria:** Looking at mirrors does not trigger false drowsiness alerts; high-risk events always trigger critical warnings regardless of calibration.

---

### [ ] Phase 8 — Alert State Machine and Local Warning Interface
- **Goal:** Provide clear, immediate visual and auditory feedback with hysteresis and cooldown.
- **Tasks:**
  - [ ] Implement state machine with formal transition guards and hysteresis timers (prevent rapid state flipping).
  - [ ] Build local audio chime dispatcher (synthesized tones for Attention, Warning, Critical).
  - [ ] Build comprehensive local OpenCV HUD:
    - Live video feed with subtle status border (Green = Normal, Amber = Attention, Red = Warning/Critical).
    - Telemetry bar: EAR, PERCLOS gauge, Blink counter, Head pose angles.
    - System health: Capture FPS, Inference FPS, Camera status, Driver Profile name.
- **Deliverables:** `alerts/` manager and local monitoring HUD.
- **Acceptance Criteria:** Responsive UI with zero audio stutter or video lag during alert triggers.

---

### [ ] Phase 9 — ESP32 Hardware Integration
- **Goal:** Transmit real-time warning commands to an external ESP32 over USB-Serial without blocking AI processing.
- **Tasks:**
  - [ ] Define lightweight, versioned binary or delimited JSON serial protocol.
  - [ ] Build asynchronous serial worker thread with automatic port discovery, baud rate negotiation, and auto-reconnect watchdog.
  - [ ] Map alert levels to ESP32 peripheral behaviors:
    - Normal: Green LED, OLED displays driver status.
    - Attention: Yellow LED, single short pulse.
    - Warning: Red flashing LED, rapid buzzer pulse, OLED alert icon.
    - Critical: Continuous red LED, intense buzzer pattern, vibration motor pulse.
  - [ ] Implement fail-safe design: AI pipeline continues uninterrupted if serial cable is pulled.
- **Deliverables:** `hardware/esp32_controller.py` and ESP32 C++ firmware in `/esp32/src`.
- **Acceptance Criteria:** Bench-tested serial alerts trigger with < 20 ms latency; disconnecting ESP32 does not crash the Python engine.

---

### [ ] Phase 10 — Local Database, Driver Profiles, and Session Analytics
- **Goal:** Store driver profiles, calibration baselines, and historical sessions locally without cloud dependencies.
- **Tasks:**
  - [ ] Implement SQLite schema using SQLAlchemy:
    - `DriverProfile`, `CalibrationSession`, `MonitoringSession`, `AlertEvent`, `SessionMetricsSummary`.
  - [ ] Asynchronous write queue to ensure database transactions never block the real-time vision loop.
  - [ ] Build JSON/CSV export utility for offline session analytics and driver reports.
- **Deliverables:** `storage/` module with SQLite repositories and export scripts.
- **Acceptance Criteria:** Profiles and sessions persist across reboots with zero external network access.

---

### [ ] Phase 11 — Local API and Optional Dashboard Integration
- **Goal:** Reconnect the existing Next.js dashboard as an optional viewer consuming telemetry from the local engine.
- **Tasks:**
  - [ ] Build lightweight local FastAPI endpoints bound strictly to `127.0.0.1`:
    - `GET /api/status`: System health, camera status, active profile.
    - `POST /api/monitor/start`, `POST /api/monitor/stop`.
    - `WS /ws/telemetry`: High-frequency metrics stream (EAR, PERCLOS, Alert State, FPS) at 10-30 Hz.
    - Optional local MJPEG preview endpoint (`GET /api/stream/preview`) for in-browser visual monitoring, strictly bound to `127.0.0.1`, requiring explicit user consent/activation, local access controls, and strict prohibition of transmitting raw frames beyond the local machine boundary.
  - [ ] Update frontend dashboard to read from local API/WebSocket instead of running LiveKit video streaming.
  - [ ] Keep core Python engine completely functional whether the browser dashboard is open or closed.
- **Deliverables:** `api/` module and updated Next.js API client hooks.
- **Acceptance Criteria:** Closing the browser window has zero impact on local drowsiness detection or ESP32 alerts.

---

### [ ] Phase 12 — Optional Cloud Sync, Backup, and Updates
- **Goal:** Provide cloud synchronization for profiles and analytics as an opt-in convenience.
- **Tasks:**
  - [ ] Implement opt-in cloud backup for driver profiles and session summaries (never raw video).
  - [ ] End-to-end encryption for synced telemetry.
  - [ ] Safe conflict resolution between local SQLite and cloud store.
- **Deliverables:** Cloud sync adapter and settings toggles.
- **Acceptance Criteria:** Complete offline functionality is preserved; cloud sync failure never impedes local alerts.

---

### [ ] Phase 13 — Testing, Evaluation, and Validation
- **Goal:** Rigorously validate performance, latency, accuracy, and edge-case resilience.
- **Tasks:**
  - [ ] Benchmark capture FPS, inference FPS, and end-to-end frame latency under sustained 30-minute runs.
  - [ ] Test edge cases: Low-light, backlight, eyeglasses, sunglasses, head turns, camera occlusion.
  - [ ] Evaluate false positive rate during mirror checks and talking/yawning.
  - [ ] Document validation metrics, latency percentiles, and limitations.
- **Deliverables:** Comprehensive validation report and test suite.
- **Acceptance Criteria:** System demonstrates sustained 30 FPS on target laptop with zero frame buffer accumulation.

---

### [ ] Phase 14 — Packaging, Documentation, and Demonstration
- **Goal:** Package the application for seamless installation and evaluation on Windows.
- **Tasks:**
  - [ ] Create simple one-click launcher (`run_driveguard.bat`).
  - [ ] Package dependencies and models with automated setup script.
  - [ ] Write user guide, hardware wiring guide (ESP32), and troubleshooting manual.
  - [ ] Prepare recorded demo mode for reproducible testing without a live driver.
- **Deliverables:** Complete user documentation and desktop runner package.
- **Acceptance Criteria:** A new user can clone the repository, run the setup script, and start monitoring within 5 minutes.

---

## 4. Definition of Done for Phase 1 (Next Step)

Phase 1 is complete when:
1. Python 3.11 environment is created and verified.
2. OpenCV opens the HP EliteBook built-in webcam via DirectShow.
3. Live preview window opens with smooth rendering.
4. Measured Capture FPS, Consumer FPS, and Frame Age are overlaid in real time.
5. Zero frame lag is observed (old frames are automatically dropped).
6. Pressing `ESC` or `q` releases the camera and shuts down cleanly without hanging.
