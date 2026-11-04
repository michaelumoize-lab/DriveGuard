# Deliverable 3: System Architecture, Wire Protocols & Database Schema

**Project:** DriveGuard  
**Component:** Full-Stack Contracts (Next.js + FastAPI + PostgreSQL + Optional ESP32 MQTT)  

---

## 1. Binary WebSocket Frame Wire Protocol (`/ws/v1/session/{session_id}`)

To avoid the $33\%$ bandwidth overhead and CPU serialization cost of Base64 JSON strings, DriveGuard uses a compact **16-byte Big-Endian Binary Header** prepended to each compressed JPEG frame payload:

```text
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|   Magic (0x4447 = "DG")       |  Version (1)  |  Mode Flags   |  Bytes 0..3
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                        Sequence Number                        |  Bytes 4..7 (uint32)
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                  Client Capture Timestamp (ms)                |  Bytes 8..15 (float64 / uint64)
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                     JPEG Binary Payload ...                   |  Bytes 16..N
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

- **Magic (`0x4447`):** Validates packet integrity (`"DG"`).
- **Version (`0x01`):** Protocol version `1`.
- **Mode Flags (`uint8`):** Bit 0 = `is_calibrating`, Bit 1 = `calibration_pose_step` (`00`=Center, `01`=Left, `10`=Right, `11`=Down).
- **Sequence Number (`uint32`):** Monotonically increasing frame counter for dropped-frame metrics.
- **Client Capture Timestamp (`float64`):** `performance.timeOrigin + performance.now()` in milliseconds so blink kinematics use exact camera capture time, completely immune to network jitter.

### Server-to-Client JSON Telemetry Response

```json
{
  "type": "TELEMETRY_UPDATE",
  "seq": 1042,
  "capture_ts_ms": 1793800412345.6,
  "inference_latency_ms": 14.2,
  "state": "WARNING",
  "risk_score": 0.68,
  "camera_quality": "GOOD",
  "metrics": {
    "ear_norm": 0.214,
    "ear_z": 2.15,
    "perclos_60s": 14.8,
    "last_blink_duration_ms": 340.0,
    "blink_rcr": 0.61,
    "blink_rate_per_min": 19.5,
    "mar": 0.18,
    "head_pitch_deg": -6.4,
    "head_yaw_deg": 3.1,
    "cnn_gate_triggered": true,
    "cnn_closed_prob": 0.84
  },
  "reasons": [
    "BLINK_REOPENING_VELOCITY_SLOWED",
    "BLINK_DURATION_ABOVE_PERSONAL_BASELINE",
    "INCREASED_PERCLOS"
  ],
  "baseline_drift_locked": true
}
```

---

## 2. PostgreSQL Database Schema (SQLAlchemy / DDL Blueprint)

```sql
-- 1. Users Table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    full_name VARCHAR(150) NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 2. Personalised Driver Behaviour Profiles (PDBP)
CREATE TABLE driver_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    profile_name VARCHAR(100) NOT NULL DEFAULT 'Default Driving Profile',
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    -- Baseline Anchor B0 (Immutable after calibration)
    ear_median FLOAT NOT NULL,
    ear_mad_scaled FLOAT NOT NULL,
    blink_dur_median_ms FLOAT NOT NULL,
    blink_dur_mad_ms FLOAT NOT NULL,
    blink_rcr_median FLOAT NOT NULL,
    blink_rcr_mad FLOAT NOT NULL,
    blink_freq_median FLOAT NOT NULL,
    blink_freq_mad FLOAT NOT NULL,
    neutral_pitch_deg FLOAT NOT NULL,
    neutral_yaw_deg FLOAT NOT NULL,
    -- Pose-Conditioned Polynomial Coefficients
    pose_coeff_pitch FLOAT NOT NULL DEFAULT 0.0,
    pose_coeff_yaw FLOAT NOT NULL DEFAULT 0.0,
    pose_coeff_pitch_sq FLOAT NOT NULL DEFAULT 0.0,
    -- Quality & Metadata
    calibration_quality_score FLOAT NOT NULL,
    calibrated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 3. Calibration Sessions
CREATE TABLE calibration_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    profile_id UUID REFERENCES driver_profiles(id) ON DELETE SET NULL,
    duration_sec FLOAT NOT NULL,
    valid_frames_ratio FLOAT NOT NULL,
    detected_blinks_count INT NOT NULL,
    status VARCHAR(32) NOT NULL CHECK (status IN ('COMPLETED', 'REJECTED_POOR_VISIBILITY', 'REJECTED_INSUFFICIENT_BLINKS')),
    raw_summary_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 4. Monitoring Sessions
CREATE TABLE monitoring_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    profile_id UUID NOT NULL REFERENCES driver_profiles(id),
    mode VARCHAR(32) NOT NULL DEFAULT 'PERSONALIZED', -- 'PERSONALIZED' or 'FIXED_BASELINE'
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    total_frames_processed INT NOT NULL DEFAULT 0,
    warning_count INT NOT NULL DEFAULT 0,
    critical_count INT NOT NULL DEFAULT 0,
    mean_inference_latency_ms FLOAT
);

-- 5. Explainable Detection Events (Alerts)
CREATE TABLE detection_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES monitoring_sessions(id) ON DELETE CASCADE,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('WARNING', 'CRITICAL', 'UNAVAILABLE')),
    risk_score FLOAT NOT NULL,
    reasons JSONB NOT NULL, -- e.g. ["BLINK_DURATION_ABOVE_PERSONAL_BASELINE", "INCREASED_PERCLOS"]
    snapshot_metrics JSONB NOT NULL, -- EAR, PERCLOS, blink_dur, head_pitch at alert onset
    reaction_latency_ms FLOAT -- Time until driver returned to alert behaviour
);

-- 6. Aggregated Window Measurements (Stored every 10 seconds, NOT every frame)
CREATE TABLE session_metric_windows (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES monitoring_sessions(id) ON DELETE CASCADE,
    window_start TIMESTAMPTZ NOT NULL,
    window_end TIMESTAMPTZ NOT NULL,
    mean_ear_norm FLOAT NOT NULL,
    perclos_pct FLOAT NOT NULL,
    mean_blink_dur_ms FLOAT,
    mean_blink_rcr FLOAT,
    max_risk_score FLOAT NOT NULL,
    dominant_state VARCHAR(20) NOT NULL
);
```

---

## 3. Repository Directory Blueprint

```text
DriveGuard/
├── docs/                                  # Planning & Technical Specifications
│   ├── 01_competitive_analysis_and_uniqueness.md
│   ├── 02_mathematical_and_algorithmic_spec.md
│   ├── 03_system_architecture_and_schemas.md
│   └── 04_experimental_protocol_and_wbs.md
├── backend/
│   ├── app/
│   │   ├── main.py                        # FastAPI entrypoint & lifespan handlers
│   │   ├── core/
│   │   │   ├── config.py                  # Pydantic Settings
│   │   │   └── security.py                # JWT & WebSocket token validation
│   │   ├── api/
│   │   │   ├── routes/
│   │   │   │   ├── auth.py
│   │   │   │   ├── profiles.py
│   │   │   │   ├── calibration.py
│   │   │   │   └── sessions.py
│   │   │   └── ws_telemetry.py            # Binary WebSocket handler + backpressure drop
│   │   ├── ai/
│   │   │   ├── landmarker.py              # MediaPipe Face Landmarker wrapper
│   │   │   ├── geometry.py                # 3D EAR, MAR, Euler head pose
│   │   │   ├── blink_kinematics.py        # Spline interpolation, duration, RCR velocity
│   │   │   ├── eye_cnn.py                 # MobileNetV3-Small ONNX/PyTorch cascade verifier
│   │   │   ├── pdbp_calibration.py        # Guided 4-pose calibration & Median/MAD computation
│   │   │   └── risk_engine.py             # Robust z-scores, hysteresis FSM & reason generator
│   │   ├── services/
│   │   │   ├── session_manager.py         # Per-session isolated state registry
│   │   │   └── mqtt_publisher.py          # Optional ESP32 alert publisher
│   │   └── db/
│   │       ├── models.py                  # SQLAlchemy 2.0 async models
│   │       └── session.py                 # AsyncPG database engine
│   ├── research/                          # Offline Evaluation & Training Scripts (Start Here!)
│   │   ├── train_mobilenetv3_mrl.py       # Fine-tuning script on MRL Eye Dataset
│   │   ├── evaluate_uta_rldd.py           # Fixed vs. Personalized ablation benchmark
│   │   └── prototype_webcam_cli.py        # Phase 1 standalone OpenCV/MediaPipe prototype
│   ├── tests/
│   └── requirements.txt
├── frontend/                              # Next.js 15 + TypeScript + Tailwind CSS
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx                   # Overview Dashboard
│   │   │   ├── calibrate/page.tsx         # Guided 4-zone interactive calibration wizard
│   │   │   ├── monitor/page.tsx           # Live monitoring HUD + explainable alert feed
│   │   │   └── analytics/page.tsx         # Fixed vs Personalized comparison & session trends
│   │   ├── components/
│   │   ├── hooks/
│   │   │   └── useBinaryVideoSocket.ts    # Binary header packer + frame rate regulator
│   │   └── lib/
│   └── package.json
└── firmware/                              # Optional ESP32 Alert Unit
    └── driveguard_esp32_mqtt/
        └── driveguard_esp32_mqtt.ino      # SSD1306 OLED + RGB LED + Buzzer + Heartbeat watchdog
```
