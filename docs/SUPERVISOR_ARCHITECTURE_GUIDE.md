# DriveGuard — Academic Architecture & Engineering Guide

**Project:** DriveGuard — Personalized AI-Based Driver Drowsiness Detection and Early Warning System  
**Audience:** Developer, Academic Supervisor, External Examiners  
**Engineering Discipline:** Computer Engineering / Applied Machine Learning  

---

## 1. Project Overview & Motivation

Driver drowsiness and inattention contribute significantly to road traffic collisions globally. Traditional vehicle-based driver monitoring systems (DMS) often rely on fixed, population-wide thresholds for eye closure or require dedicated, expensive proprietary in-cabin ECUs.

**DriveGuard** is an open, modular, local-first driver monitoring system designed to run on accessible consumer computing hardware (e.g., laptop or edge computing SBC) using standard UVC-compliant optical sensors (laptop webcam, phone camera, or near-infrared USB camera) and external microcontroller warning peripherals (ESP32).

### Core Research Contributions
1. **Local-First Real-Time Edge Processing:** Eliminates cloud/network latency, ensuring immediate safety alerts and strict privacy preservation (zero video transmission to remote servers).
2. **Personalized Driver Behaviour Profile (PDBP):** Calibrates individual baseline eye aspect ratio (EAR) and head orientation across 4 natural driving gaze zones (Center, Left Mirror, Right Mirror, Dashboard).
3. **Bounded Adaptive Safety Bounds:** Combines statistical personalization (Median, MAD) with strict population safety limits, preventing the system from normalizing hazardous prolonged eye closures.
4. **Hardware-Decoupled Warning Dispatch:** Fail-safe serial bridge to an ESP32 microcontroller with autonomous visual, acoustic, and haptic indicators.

---

## 2. Directory Structure & Modular Breakdown

The repository is structured to enforce strict **Separation of Concerns (SoC)**:

```text
DriveGuard/
├── AGENT-GUIDE.MD              # Core engineering and development rules
├── README.md                   # Repository overview & setup guide
│
├── docs/                       # Project documentation & academic guides
│   ├── PHASE_BY_PHASE_PLAN.md  # 15-phase implementation roadmap
│   └── SUPERVISOR_ARCHITECTURE_GUIDE.md # This architecture & viva guide
│
├── backend/                    # Core Python 3.11 AI Engine
│   ├── app/
│   │   ├── camera/             # Frame acquisition abstraction layer
│   │   │   ├── base.py         # Abstract camera interface
│   │   │   ├── camera_config.py# Capture settings (resolution, FPS, backend)
│   │   │   ├── opencv_camera.py# Threaded OpenCV DirectShow capture adapter
│   │   │   ├── phone_camera.py # Phone IP/USB camera adapter (Phase 2)
│   │   │   └── frame_buffer.py # Thread-safe 1-slot latest-frame queue
│   │   │
│   │   ├── vision/             # Computer vision & landmark detection (Phase 4)
│   │   │   ├── face_landmarker.py # MediaPipe 478 face & iris mesh wrapper
│   │   │   └── face_quality.py    # Lighting & landmark confidence validator
│   │   │
│   │   ├── features/           # Mathematical physiological extraction (Phase 5)
│   │   │   ├── ear.py          # Left/Right Eye Aspect Ratio
│   │   │   ├── blink.py        # Sub-frame blink duration & reopening velocity
│   │   │   ├── perclos.py      # Rolling 60s Percentage of Eye Closure
│   │   │   ├── mar.py          # Mouth Aspect Ratio (yawning)
│   │   │   └── head_pose.py    # 6-DoF SolvePnP head orientation (pitch, yaw, roll)
│   │   │
│   │   ├── intelligence/       # Temporal reasoning & decision engine (Phase 6)
│   │   │   ├── drowsiness_engine.py # Multi-indicator temporal fusion
│   │   │   ├── decision_engine.py   # 5-state hysteresis decision FSM
│   │   │   └── confidence.py        # Evidence reliability weighting
│   │   │
│   │   ├── personalization/    # Driver-specific calibration (Phase 7)
│   │   │   ├── calibration.py       # 4-zone guided baseline acquisition
│   │   │   └── adaptive_thresholds.py # Bounded mathematical clamps
│   │   │
│   │   ├── alerts/             # Alert dispatch & warning hierarchy (Phase 8)
│   │   │   ├── alert_levels.py      # Severity enums (Normal to Critical)
│   │   │   └── alert_manager.py     # Cooldowns, sound chimes, visual alerts
│   │   │
│   │   ├── hardware/           # Microcontroller communication (Phase 9)
│   │   │   ├── esp32_controller.py  # Threaded serial worker & watchdog
│   │   │   └── serial_protocol.py   # Frame packet formatting
│   │   │
│   │   ├── storage/            # Local data persistence (Phase 10)
│   │   │   ├── database.py          # SQLite session manager
│   │   │   └── models.py            # SQLAlchemy schemas
│   │   │
│   │   ├── monitoring/         # Operational metrics (Phase 1, 3, 13)
│   │   │   └── metrics.py           # Capture FPS, inference latency, frame age
│   │   │
│   │   ├── api/                # Optional localhost server (Phase 11)
│   │   └── main.py             # Local application entrypoint
│   │
│   └── requirements.txt        # Pinned Python dependencies
│
├── frontend/                   # Optional Next.js 16 Web Dashboard (Phase 11)
├── esp32/                      # C++ Firmware for ESP32 warning module (Phase 9)
└── scripts/                    # Standalone test runners and benchmarks
    └── run_camera_test.py      # Phase 1 camera verification runner
```

---

## 3. Viva Defense: Key Architectural Questions & Answers

### Q1: Why not stream webcam frames from the browser to the backend using WebSockets or WebRTC?
**Answer:** In safety-critical driver monitoring, latency and jitter must be strictly bounded. Browser-based capture introduces:
1. Frame conversion overhead (`Canvas` / WebRTC encoder in JavaScript).
2. Dynamic network jitter buffers (adding 200–800 ms of unpredictable delay).
3. Browser tab throttling when in the background.
4. Total loss of native camera controls (DirectShow / UVC manual exposure and infrared lighting).

By capturing directly in Python via `cv2.CAP_DSHOW` with a dedicated thread and a 1-slot buffer (`maxlen=1`), frame latency is reduced to under 30 milliseconds, and old frames are automatically dropped rather than creating a dangerous processing backlog.

### Q2: How does the system prevent false alarms when the driver checks rear-view mirrors?
**Answer:** DriveGuard uses a **4-Zone Pose-Conditioned Calibration**. When the driver turns their head to check the left or right mirror (detected via 6-DoF SolvePnP head pose), the decision engine applies the calibrated baseline for that specific zone, preventing false alerts from temporary profile-angle eye narrowing or brief head rotation.

### Q3: What prevents the adaptive algorithm from learning that prolonged eye closure is "normal"?
**Answer:** The system enforces **bounded adaptive thresholds**. Personalization adjusts sensitivity relative to natural facial geometry (e.g., naturally narrow eyelids or deep-set eyes), but hard mathematical safety clamps are permanently active. For instance, any eye closure exceeding 1.5 seconds immediately bypasses personalization and triggers an urgent warning.

### Q4: What happens if the ESP32 hardware or camera disconnects while driving?
**Answer:** All modules follow the **Fault Isolation Principle**:
- If the ESP32 disconnects, the serial worker marks hardware as unavailable and continues local visual/audio alerts without stalling the vision thread.
- If the camera disconnects, the engine enters an explicit `INSUFFICIENT_DATA` / `CAMERA_FAULT` state, warning the driver rather than assuming the driver is asleep or awake.
