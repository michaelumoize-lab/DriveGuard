# DriveGuard

**A Personalised Adaptive AI Driver Drowsiness Detection and Early Warning System Using Computer Vision**

DriveGuard is a real-time driver-monitoring platform that learns an individual driver's normal facial, eye, and head-pose behaviour during a guided calibration session, monitors changes relative to that personal baseline, and provides progressive, explainable warnings when patterns associated with increasing drowsiness emerge.

---

## Key Technical Contributions

1. **Staged 4-Zone Pose-Conditioned Personal Baseline (`PDBP`):**
   - Evaluates separate behavioural baselines across four natural driving gaze zones: `CENTER` (primary reference), `LEFT_MIRROR`, `RIGHT_MIRROR`, and `DASHBOARD`.
   - Uses head-pose context to prevent normal mirror checks ($\le 3.0\text{ s}$) from triggering false low-EAR drowsiness alarms.
   - Computes directional robust deviation scores ($z_r$) using **Median** and scaled **Median Absolute Deviation ($\text{MAD}$)** with **Dual-Anchor Drift-Safe Adaptation**.
2. **Sub-Frame Blink Kinematics & Rolling Time-Weighted PERCLOS:**
   - Linearly interpolates threshold crossings using capture timestamps to measure **blink duration**, **closing velocity**, **reopening velocity**, and **Reopening-to-Closing Ratio ($\text{RCR}$)** alongside rolling 60-second time-weighted **PERCLOS**.
3. **Reliability-Weighted Confidence-Gated Eye Cascade:**
   - Fast Stage 1: **MediaPipe Face Landmarker** (478 3D landmarks + 52 blendshapes + $4 \times 4$ head pose matrix).
   - On-Demand Stage 2: **Eye-Crop Verifier (`MobileNetV3-Small` / ONNX)** invoked only when geometric reliability drops, EAR and blendshapes disagree, or glasses glare is detected.
   - Uses **Reliability-Weighted Arbitration** (`GEOMETRIC`, `CNN`, or `WEIGHTED_CONSENSUS`) rather than blindly trusting the CNN when eye crops are blurry or poorly exposed.

---

## Repository Structure

```text
DriveGuard/
├── docs/                                           # Project Specifications & Research Plan
│   ├── 01_competitive_analysis_and_uniqueness.md   # GitHub gap analysis & 5 differentiators
│   ├── 02_mathematical_and_algorithmic_spec.md     # Mathematical PDBP, kinematics & cascade spec
│   ├── 03_system_architecture_and_schemas.md       # Binary WebSocket wire protocol & DB schema
│   └── 04_experimental_protocol_and_wbs.md         # 5-stage ablation matrix & 12-week WBS
├── backend/
│   ├── app/
│   │   └── ai/                                     # Core Modular AI Engine
│   │       ├── geometry.py                         # 3D EAR, MAR, Head Pose & 4-Zone Classifier
│   │       ├── blink_kinematics.py                 # Sub-frame blink duration, RCR & PERCLOS
│   │       ├── pdbp_calibration.py                 # Guided 1-Zone & 4-Zone PDBP Calibration
│   │       ├── cascade_classifier.py               # Experiments A, B, C & Reliability Arbitration
│   │       ├── risk_engine.py                      # Adaptive Risk Score R_t & Hysteresis FSM
│   │       └── landmarker.py                       # MediaPipe Face Landmarker wrapper
│   ├── research/
│   │   ├── prototype_webcam_cli.py                 # Live Webcam HUD & Simulation Benchmark
│   │   └── outputs/                                # Sample calibrated profile & benchmark JSON
│   ├── tests/
│   │   └── test_ai_pipeline.py                     # Automated unit test suite
│   └── requirements.txt
└── README.md
```

---

## Quick Start (Phase 1 Prototype & Benchmark)

### 1. Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### 2. Run the Interactive Webcam Prototype
```bash
python -m backend.research.prototype_webcam_cli --camera 0
```
- Press `C` for **4-Zone Guided Calibration** (`CENTER` $\rightarrow$ `LEFT_MIRROR` $\rightarrow$ `RIGHT_MIRROR` $\rightarrow$ `DASHBOARD`).
- Press `1` for **Quick 1-Zone (`CENTER`) Calibration**.
- Press `A`, `B`, or `G` to switch live between **Experiment A (Geometric Only)**, **Experiment B (CNN Always)**, and **Experiment C (Confidence-Gated Cascade)**.
- Press `S` to save your profile and session telemetry, or `Q` / `ESC` to quit.

### 3. Run the Headless Experimental Benchmark & Unit Tests
```bash
python -m backend.research.prototype_webcam_cli --benchmark-simulation
python -m unittest discover -s backend/tests -v
```
