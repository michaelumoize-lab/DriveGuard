# DriveGuard — Repeatable Real-World Validation Procedure

> **Important Scope & Safety Statement:** Passing this validation procedure demonstrates repeatable empirical performance on recorded trials. DriveGuard is a research and engineering prototype and is **not** road-safety certified, ISO 26262 qualified, or production automotive certified.

---

## 1. Objective

Provide a standardized, repeatable field and bench validation protocol for evaluating the authoritative DriveGuard pipeline (`SessionAIEngine → LocalMonitoringService → DrowsinessEngine → AlertStateManager`) under realistic driving-cabin variability without relying solely on synthetic unit tests.

---

## 2. Test Matrix

### 2.1 Driver Cohort Diversity
Record at least **10 distinct participants** covering:
- Eyewear: no glasses, clear prescription glasses (anti-reflective and reflective lenses), thin-rim vs thick-rim frames, sunglasses (verify graceful `MONITORING_UNAVAILABLE` / degraded tracking rather than false green `NORMAL`).
- Facial characteristics: varied skin tones, eye aperture sizes (narrow vs wide baseline EAR), facial hair, headgear.
- Seating posture: upright, slightly reclined, short vs tall driver eye-line relative to camera.

### 2.2 Lighting & Cabin Conditions
Evaluate each participant across at least **4 illumination regimes**:
1. Even daytime indirect light (baseline).
2. Direct side sunlight / high dynamic range contrast (glare test).
3. Rapid shadow/bright transitions (tree-lined road / tunnel exit simulation).
4. Low-light / dusk cabin illumination (verify `FrameQualityAnalyzer` flags `SENSOR_DEGRADED` honestly rather than fabricating alerts or silent green `NORMAL`).

### 2.3 Camera Mount & Geometry
Record trials at:
- **Center dash / instrument-top mount** (55–70 cm distance, 0°–10° upward pitch).
- **Oblique vent / windshield mount** (60–80 cm distance, ±15°–25° horizontal offset — validates `neutral_rot_mat` & `calibration_is_mirrored` compensation).
- **Resolution & frame rate:** 640×480 @ 25 FPS (primary browser LiveKit profile) and 1280×720 @ 30 FPS (desktop profile).

---

## 3. Standardized Trial Protocol (Per Session)

Every session follows the same timed script with synchronized ground-truth annotations (`ground_truth.json`):

| Phase | Duration | Driver Action | Expected Ground Truth |
|---|---|---|---|
| **P0 — Calibration** | ~25s (or ~145s w/ blink stream) | Run 4-Zone guided calibration (`CENTER`, `LEFT_MIRROR`, `RIGHT_MIRROR`, `DASHBOARD` + optional 2-min blink cadence) | Calibration converges with `is_calibrated=true`, zero invalid `neutral_rot_mat`, no fabricated defaults |
| **P1 — Attentive Driving** | 120s | Look forward, blink naturally, speak briefly | `NORMAL` state; **0** false `WARNING`/`CRITICAL` alerts |
| **P2 — Mirror & Dashboard Glances** | 60s | Perform 3 left-mirror (1.5–2.5s), 3 right-mirror (1.5–2.5s), and 3 dashboard (1.5–2.5s) checks | Zones match ground truth; `MIRROR_CHECK_CONTEXT` / `DASHBOARD_CHECK_CONTEXT` suppresses false fatigue alerts |
| **P3 — Transient Occlusions** | 30s | Hand briefly crosses face (0.5s), then 3s full occlusion | Brief occlusion buffered by grace period; 3s occlusion transitions honestly to `MONITORING_UNAVAILABLE` / `FACE_LOST` |
| **P4 — Yawning & Mild Fatigue** | 60s | 2–3 natural/simulated yawns + slow blinking | `AUTHENTIC_YAWN` / `ATTENTION` or `WARNING` without false critical microsleep |
| **P5 — Sustained Eye Closure (Simulated Microsleep)** | 45s | 3 controlled eye closures of 1.8s–2.2s separated by 10s attentive recovery | `CRITICAL_WARNING` / `SUSTAINED_EYE_CLOSURE` triggered within target latency; recovers after alert eyes reopen |
| **P6 — Head Slump / Off-Road Distraction** | 45s | Sustained downward head drop (2.5s) and sustained off-road turn (>4s) | `NODDING_MICROSLEEP` / `SUSPECTED_HEAD_SLUMP` and `PERSISTENT_DISTRACTION` detected |

---

## 4. Quantitative Metrics & Acceptance Benchmarks

Compute all metrics using `scripts/validation/evaluate_recorded_sessions.py`:

| Metric | Definition | Research Target (Non-Certified) |
|---|---|---|
| **Critical Recall (1 - FNR)** | Fraction of ground-truth sustained closures (>= 1.5s) triggering `CRITICAL` within 2.5s of onset | `>= 95%` |
| **False Positive Rate (FPR)** | False `WARNING`/`CRITICAL` alerts per attentive driving hour (P1 + P2) | `<= 1.0 / hr` (0 during P2 mirror checks) |
| **Alert Detection Latency (P50 / P95)** | Time from ground-truth closure threshold crossing (1.5s) to emitted `CRITICAL` telemetry | `P50 <= 200 ms`, `P95 <= 400 ms` |
| **Pose-Zone Accuracy** | Frame-level agreement between predicted `pose_zone` and annotated cabin zone (P1 + P2) | `>= 90%` (0% invalid poses mapped to `CENTER`) |
| **Calibration Success Rate** | Fraction of participants completing 4-zone calibration on first attempt without rejection | `>= 90%` |
| **Pipeline Throughput & Latency** | Sustained processing FPS and P95 frame latency at 640×480 | `>= 15 FPS` sustained, `P95 <= 70 ms` |

---

## 5. Reproducible Evaluation Command

Run the automated CLI evaluator over recorded session JSONL telemetry and ground-truth JSON annotations:

```bash
python scripts/validation/evaluate_recorded_sessions.py \
  --telemetry path/to/session_telemetry.jsonl \
  --ground-truth path/to/ground_truth.json \
  --report-out path/to/validation_report.json
```

Verify the evaluator's metric calculations with its built-in deterministic self-test:

```bash
python scripts/validation/evaluate_recorded_sessions.py --self-test
```

### Expected Input Formats
- **`session_telemetry.jsonl`**: One JSON object per line matching `FrameTelemetryDTO` (fields: `timestamp_ms`, `state`, `drowsiness_state`, `pose_zone`, `face_detected`, `reason_codes`, `inference_latency_ms`).
- **`ground_truth.json`**:
  ```json
  {
    "session_id": "trial_001",
    "driver_profile_calibrated": true,
    "critical_episodes": [
      { "onset_ms": 15000, "end_ms": 17200, "label": "SUSTAINED_EYE_CLOSURE" }
    ],
    "attentive_intervals": [
      { "start_ms": 0, "end_ms": 12000 }
    ],
    "zone_intervals": [
      { "start_ms": 0, "end_ms": 12000, "zone": "CENTER" }
    ]
  }
  ```
