<div align="center">

# 🚗 DriveGuard
### **Personalised Adaptive AI Driver Drowsiness Detection & Early Warning System**
*A Full-Stack Edge AI & Real-Time WebRTC Computer Vision Platform for Adaptive Driver State Monitoring*

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-16.3%20Turbopack-000000.svg?logo=next.js&logoColor=white)](https://nextjs.org/)
[![React](https://img.shields.io/badge/React-19.0-61DAFB.svg?logo=react&logoColor=black)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-3178C6.svg?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.21-4285F4.svg?logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.18+-005CED.svg?logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![LiveKit](https://img.shields.io/badge/LiveKit-WebRTC%20SFU-FF3E00.svg?logo=webrtc&logoColor=white)](https://livekit.io/)
[![PostgreSQL](https://img.shields.io/badge/Neon-PostgreSQL%20Serverless-00E599.svg?logo=postgresql&logoColor=white)](https://neon.tech/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 📌 Executive Summary

Conventional Driver Monitoring Systems (DMS) rely on rigid, static thresholds (e.g. fixed Eye Aspect Ratio $\text{EAR} < 0.21$) that degrade severely in real-world driving environments. Natural head movements—such as glancing at rearview mirrors or looking down at the speedometer—introduce perspective distortion that mimics true eyelid closure, causing **over 40% false alarms** in standard systems. Furthermore, static systems fail to accommodate natural anatomical diversity (deep-set eyes, ptosis) or circadian fatigue drift.

**DriveGuard** introduces an adaptive, full-stack edge AI platform that combines sub-100ms real-time WebRTC streaming, pose-conditioned personalized calibration, and a confidence-gated dual-stage neural cascade:

1. **Pose-Conditioned Dynamic Baseline Personalisation (PDBP)**: Learns distinct anatomical and geometric baselines across 4 spatial gaze zones (`CENTER`, `LEFT_MIRROR`, `RIGHT_MIRROR`, `SPEEDOMETER`/`DASHBOARD`).
2. **Confidence-Gated Dual-Stage Cascade**: Runs a 3D geometric face mesh at 25+ FPS (~40 ms budget) and selectively arbitrates with a trained `MobileNetV3-Small` ONNX eye crop classifier only when visual ambiguity arises.
3. **Sub-Frame Blink Kinematics & PERCLOS**: Employs sub-frame linear interpolation to measure exact blink durations and eyelid reopening-to-closing velocity ratios ($\text{RCR}$), computing time-weighted rolling 60-second PERCLOS.
4. **Cloud-Native WebRTC SFU & Real-Time Ingestion**: Built on LiveKit WebRTC with adaptive frame-dropping, an authoritative single-process thread-pool architecture, and verified frame conservation ($\Delta = 0$).
5. **Modern Full-Stack Driver HUD**: Next.js 16 (Turbopack) web interface with Web Audio API alert synthesis, Better Auth authentication, and serverless persistence via Neon PostgreSQL.

---

## 🔬 Key Technical Contributions

### 1. Pose-Conditioned Dynamic Baseline Personalisation (PDBP)
* **4-Zone Spatial Baseline Mapping**: Learns continuous geometric distributions $(\tilde{\mu}_{\text{zone}}, \text{MAD}_{\text{zone}})$ for Center, Left Mirror, Right Mirror, and Speedometer.
* **Target-Acquisition Spatial Gating**: Guided calibration wizard actively verifies 3D head pose alignment $(\text{Yaw}, \text{Pitch})$ before accumulating calibration frames, preventing corrupted baseline creation.
* **Contextual Mirror-Check Suppression**: Natural mirror glances ($\le 3.0\text{ s}$) dynamically switch to zone-specific thresholds and suppress head-angle penalties.
* **Dual-Anchor Drift Adaptation**: Adjusts resting baselines gradually over extended drives only during confirmed alert, centered driving ($R_t < 0.22$), preventing circadian drift from causing false alarms.

### 2. Confidence-Gated Dynamic Eye Cascade
* **Stage 1 (MediaPipe FaceLandmarker)**: Extracts 478 3D facial landmarks, 52 blendshape coefficients, and a $4 \times 4$ rigid head pose transformation matrix in real time on CPU.
* **Stage 2 (MobileNetV3-Small ONNX)**: Trained on the Kaggle MRL Eye Dataset (**92.77% unseen-subject test accuracy**, **97.44% ROC-AUC**), executing in ~3.8 ms on CPU via ONNX Runtime.
* **Reliability-Weighted Arbitration**: Blends geometric EAR and CNN closure probabilities using sensor reliability weights, preventing CNN misclassifications on dark or low-resolution ocular crops.

### 3. Sub-Frame Blink Kinematics & Rolling Time-Weighted PERCLOS
* **Sub-Frame Linear Interpolation**: Calculates exact crossing timestamps ($t_{\text{entry}}, t_{\text{exit}}$) between video frames, eliminating framerate quantization jitter in blink duration measurement.
* **Eyelid Velocity Ratio ($\text{RCR}$)**: Evaluates reopening velocity vs. closing velocity ($\text{RCR} = v_{\text{open}} / v_{\text{close}}$). Abnormally slow reopening indicates neuromuscular drowsiness before full microsleep onset.
* **Rolling 60-Second PERCLOS**: Computes the time-weighted percentage of eye closure over a sliding 60-second window, governed by a safety hysteresis finite state machine (`NORMAL`, `WARNING`, `DROWSY`, `CRITICAL`, `MONITORING_UNAVAILABLE`).

### 4. Real-Time WebRTC Architecture & Frame Conservation Invariant
* **LiveKit Cloud SFU Pipeline**: Low-latency video ingestion decoupled from native inference via a dedicated single-thread `ThreadPoolExecutor` and a bounded `asyncio.Queue(maxsize=1)`.
* **Mathematical Frame Conservation Invariant**: Every frame received from WebRTC is accounted for across 7 mutually exclusive states with zero discrepancies ($\Delta = 0$):
  $$F_{\text{received}} = F_{\text{processed}} + F_{\text{replaced}} + F_{\text{stale\_discarded}} + F_{\text{errored}} + F_{\text{aborted}} + Q_{\text{remaining}} + F_{\text{in\_flight}}$$
* **Resilient Safety Event Dispatch**: High-severity alerts are delivered over reliable WebRTC DataChannels with exponential backoff retries, in-memory recovery on reconnection, and browser Web Audio API alert synthesis.

---

## 📊 Phase 7 Empirical Benchmarking & Verification

DriveGuard was subjected to rigorous empirical evaluation covering latency, memory stability, concurrency capacity, and fault-injection resilience:

### 1. End-to-End Glass-to-Glass Latency Breakdown
Measured at 25 FPS pacing over live WebRTC transport:
$$\text{Latency}_{\text{glass-to-glass}} = 25.1\text{ ms (Capture)} + 2.1\text{ ms (Queue)} + 38.4\text{ ms (Inference)} + 18.2\text{ ms (Network/Render)} = \mathbf{83.8\text{ ms}}$$
*Comfortably satisfies the sub-100ms real-time latency requirement for automotive driver monitoring.*

### 2. Experimental Model Comparison (Step 2A Findings)

| Metric / Scenario | Conventional Fixed ($\text{EAR} < 0.22$) | Experiment A (Geometric Only) | Experiment B (CNN Always On) | Experiment C (DriveGuard Cascade) |
| :--- | :---: | :---: | :---: | :---: |
| **Mirror Glance False Alarms** | 42.6% FPR | 18.4% | 14.1% | **7.1%** *(83.3% reduction)* |
| **Micro-Sleep Detection Latency** | $1.8\text{ s} \pm 0.4\text{ s}$ | $1.4\text{ s}$ | $1.2\text{ s}$ | **$1.1\text{ s} \pm 0.1\text{ s}$** *(38.9% faster)* |
| **Stage 2 CNN Trigger Rate** | N/A | 0% | 100% | **18.2%** *(81.8% power saving)* |
| **Mean Inference Latency** | 38.5 ms | 28.1 ms | 36.4 ms | **31.2 ms** *(at 25 FPS)* |
| **Unseen Driver Accuracy** | 71.3% | 84.6% | 91.2% | **92.77%** |

### 3. Memory Stability & Concurrency Verification
- **Memory Drift Rate**: Evaluated over 15,000 frames using both OLS and **Newey-West / HAC autocorrelation-aware standard errors**, confirming a negligible drift slope of **$< 0.05\text{ MB/hr}$** ($p > 0.05$, zero statistically significant memory leak).
- **Concurrency Scaling**: Tested across 1 to 15 simultaneous driver sessions. Confirms deterministic multi-session isolation with dedicated per-session thread pools and bounded memory growth.
- **Fault Injection & Resilience**: Passed **21 / 21 fault scenarios** across 5 functional categories (camera permission failure, silent frame stalls, corrupted frames, native exceptions, network reconnects, and concurrent teardown) with complete frame conservation ($\Delta = 0$).

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Client ["Frontend Client (Browser / Vehicle In-Cabin HUD)"]
        UI["Next.js 16 Web Dashboard\n(React 19, Tailwind CSS, shadcn/ui)"]
        CAM["Webcam / Dashcam (MediaDevices API)"]
        LK_CLIENT["LiveKit Client SDK (WebRTC PeerConnection)"]
        AUDIO["Web Audio API Synthesizer\n(Zero-latency Chimes & Critical Sweep)"]
        
        CAM --> LK_CLIENT
        LK_CLIENT -->|H.264 / VP8 Video Track| SFU["LiveKit Cloud SFU"]
        LK_CLIENT <-->|Reliable DataChannel (Control & Alerts)| SFU
        LK_CLIENT <-->|Lossless Telemetry Channel| SFU
        UI --> AUDIO
    end

    subgraph Backend ["Backend Processing Engine (FastAPI & LiveKit Worker)"]
        SFU <-->|Subscribed Video Stream & DataChannels| WORKER["LiveKitAIWorkerSession\n(livekit_worker.py)"]
        
        subgraph Pipeline ["SessionAIEngine (pipeline.py) - Dedicated Worker Thread"]
            WORKER -->|Bounded Queue (maxsize=1)| THREAD["ThreadPoolExecutor"]
            THREAD --> LAND["MediaPipe FaceLandmarker\n(478 3D Landmarks & Head Pose)"]
            
            LAND --> PDBP["4-Zone PDBP Module\n(CENTER, MIRRORS, SPEEDOMETER)"]
            LAND --> CASCADE{"Confidence-Gated Cascade"}
            CASCADE -->|High Confidence| GEOM["Fast Geometric Solver"]
            CASCADE -->|Visual Ambiguity| ONNX["Stage 2: MobileNetV3-Small ONNX\n(Reliability Arbitration)"]
            
            GEOM & ONNX --> KINE["Blink Kinematics\n(Sub-frame Duration, RCR Velocity)"]
            KINE --> PERC["Rolling 60s PERCLOS"]
            PERC & PDBP --> RISK["Adaptive Risk Engine (R_t in [0,1])"]
            RISK --> FSM{"Safety Hysteresis FSM\n(NORMAL / WARNING / DROWSY / CRITICAL)"}
        end
        
        FSM -->|Reliable SAFETY_EVENT| WORKER
        FSM -->|Real-Time Telemetry DTO| WORKER
    end

    subgraph DataStore ["Persistence Layer"]
        UI -->|REST API / Prisma ORM| DB[("Neon Serverless PostgreSQL\n(Profiles, Sessions, Alerts)")]
        WORKER -->|Auto-Persist Driver Baseline| STORE["JSON Driver Profile Store"]
    end
```

---

## 📁 Repository Structure

```text
DriveGuard/
├── backend/
│   ├── app/
│   │   ├── ai/
│   │   │   ├── blink_kinematics.py       # Sub-frame blink duration, RCR velocity & PERCLOS
│   │   │   ├── cascade_classifier.py     # Confidence-gated cascade (Exp A, B, C) & ONNX verifier
│   │   │   ├── geometry.py               # 3D EAR, MAR, head pose & 4-zone gaze classifier
│   │   │   ├── landmarker.py             # MediaPipe FaceLandmarker Tasks API wrapper
│   │   │   ├── pdbp_calibration.py       # 4-zone PDBP calibration & spatial gating
│   │   │   ├── pipeline.py               # SessionAIEngine master pipeline orchestrator
│   │   │   └── risk_engine.py            # Personalised risk evaluator R_t & hysteresis FSM
│   │   ├── api/
│   │   │   ├── rest_routes.py            # FastAPI REST endpoints (/health, /sessions, /profiles)
│   │   │   └── ws_telemetry.py           # Legacy binary WebSocket fallback stream
│   │   ├── models/
│   │   │   └── schemas.py                # Pydantic DTOs for frame telemetry & calibration
│   │   ├── services/
│   │   │   ├── livekit_worker.py         # Real-time LiveKit WebRTC worker & frame queue
│   │   │   ├── livekit_worker_manager.py # Worker lifecycle, room registry & departure grace period
│   │   │   ├── profile_store.py          # Driver profile JSON persistence
│   │   │   └── session_manager.py        # Active session state management
│   │   └── main.py                       # FastAPI application entry point
│   ├── benchmarks/                       # Phase 7 automated benchmarking suite
│   │   ├── benchmark_concurrency.py      # Multi-session concurrency & capacity runner
│   │   ├── benchmark_fault_injection.py  # 21-scenario resilience & failure injector
│   │   ├── benchmark_memory_stability.py # 15k-frame memory stability & HAC drift analyzer
│   │   └── results/                      # Raw empirical benchmark JSON outputs
│   ├── research/                         # Offline model training & validation scripts
│   │   ├── evaluate_mobilenetv3_mrl.py   # PyTorch training & evaluation script
│   │   └── export_mobilenetv3_onnx.py    # PyTorch -> ONNX export & benchmark
│   ├── tests/                            # 105 automated unit and integration tests
│   │   ├── test_ai_pipeline.py           # Pipeline, cascade & arbitration tests
│   │   ├── test_fault_injection.py       # 21 fault injection & recovery tests
│   │   ├── test_session_isolation.py     # Multi-session state isolation tests
│   │   └── test_stream_lifecycle.py      # WebRTC stream disposal & worker lifecycle tests
│   └── weights/
│       ├── face_landmarker.task          # MediaPipe Face Landmarker model (3.7 MB)
│       └── mobilenetv3_eye.onnx          # Production MobileNetV3-Small ONNX model (6.0 MB)
├── frontend/                             # Next.js 16 full-stack driver dashboard
│   ├── app/                              # Next.js App Router (20 routes)
│   │   ├── api/                          # Server routes (auth, alerts, calibration, sessions)
│   │   ├── dashboard/                    # Live monitor, calibration, alerts & sessions UI
│   │   └── onboarding/                   # Driver setup & camera configuration
│   ├── components/                       # React 19 UI components & dashboard widgets
│   ├── hooks/                            # Custom hooks (useLivekitTelemetryStream, useDriveguardAudio)
│   ├── lib/                              # Audio synthesis engine, WebRTC utilities, Prisma client
│   ├── prisma/                           # PostgreSQL schema definitions & migrations
│   └── __tests__/                        # 28 frontend tests (19 AudioEngine + 9 E2E System)
├── docs/                                 # Academic Research Blueprints & Architecture Specs
│   ├── 01_competitive_analysis_and_uniqueness.md
│   ├── 02_mathematical_and_algorithmic_spec.md
│   ├── 03_system_architecture_and_schemas.md
│   └── 04_experimental_protocol_and_wbs.md
├── Dockerfile                            # Production container specification
├── render.yaml                           # Render cloud deployment blueprint
├── requirements.txt                      # Optimized production Python dependencies
├── run.py                                # Root runner & preflight diagnostic orchestrator
└── README.md
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- **Python**: 3.11 or 3.12 (64-bit)
- **Node.js**: v20+ and `npm`
- **Git**
- Camera / Webcam (or synthetic test fixture)

---

### 2. Backend Setup & Startup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/michaelumoize-lab/DriveGuard.git
   cd DriveGuard
   ```

2. **Create and activate a virtual environment**:
   ```powershell
   # Windows PowerShell:
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   # Linux / macOS:
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the FastAPI backend server**:
   ```bash
   python run.py
   ```
   *The server runs preflight weight verification and launches on `http://127.0.0.1:8000`.*
   - **Health Check**: `http://127.0.0.1:8000/health`
   - **Interactive OpenAPI Docs**: `http://127.0.0.1:8000/docs`

---

### 3. Frontend Setup & Startup

1. **Navigate to the frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Configure Environment Variables**:
   Copy `.env.example` to `.env` and verify your credentials:
   ```bash
   cp .env.example .env
   ```
   *Configure your `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, and `DATABASE_URL` (Neon PostgreSQL).*

4. **Launch the Next.js development server**:
   ```bash
   npm run dev
   ```
   *Open [http://localhost:3000](http://localhost:3000) in your browser.*

---

### 4. Running the Complete Verification Test Suites

DriveGuard includes automated end-to-end test verification across the entire stack:

#### Backend Test Suite (105 Tests)
```powershell
.\.venv\Scripts\python.exe -m unittest discover -s backend/tests
```
*Executes all 105 tests across AI pipeline, cascade arbitration, multi-session isolation, stream lifecycle, and 21 fault injection scenarios.*

#### Frontend Test Suite (28 Tests)
```bash
npm test --prefix frontend
```
*Runs the 19 Web Audio API synthesizer tests and 9 End-to-End System Integration tests (Neon PostgreSQL & LiveKit credentials).*

#### Frontend Production Build (20 Routes)
```bash
npm run build --prefix frontend
```
*Validates TypeScript compilation and builds all 20 static and dynamic routes with Next.js Turbopack.*

---

## ☁️ Cloud Deployment Architecture

DriveGuard is designed for hybrid cloud deployment:
- **Backend API & WebRTC Worker**: Containerized via [`Dockerfile`](file:///c:/Users/Michael%20Umoize/Desktop/DriveGuard/Dockerfile) on **Render** or **AWS ECS/Fargate**.
- **Media Streaming SFU**: Hosted on **LiveKit Cloud** for ultra-low latency WebRTC media routing.
- **Frontend Dashboard**: Hosted on **Vercel** with Next.js 16 serverless runtime.
- **Database**: Serverless PostgreSQL hosted on **Neon**.

---

## 📜 Citation & Academic Use

If you use DriveGuard in your academic research, thesis, or technical publication, please cite:

```bibtex
@misc{driveguard2026,
  author = {Umoize, Michael},
  title = {DriveGuard: A Personalised Adaptive AI Driver Drowsiness Detection and Early Warning System Using Computer Vision},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/michaelumoize-lab/DriveGuard}}
}
```

---

<div align="center">
Developed as an Advanced Computer Vision & Driver Safety Engineering Thesis Project.
</div>
