# Deliverable 4: Experimental Protocol, Ablation Matrix & 12-Week WBS

**Project:** DriveGuard  
**Component:** Academic Evaluation Plan & Engineering Roadmap  

---

## 1. Ablation Study Matrix (Your Thesis Results Chapter)

To prove to your project examiners that **every component of DriveGuard earns its place**, run the following 5-configuration ablation study on identical participant test videos from **UTA-RLDD** and **NTHU-DDD**:

| Config ID | Detector Configuration | Calibration Used? | Features Used | CNN Verifier? | What It Proves in Your Report |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline 1 (`B1`)** | Conventional Static EAR | No ($\text{EAR} < 0.22$) | EAR only | No | Shows high false-alarm rate on narrow-eyed participants and mirror checks. |
| **Baseline 2 (`B2`)** | Static Multi-Feature | No (Fixed cutoffs) | EAR + PERCLOS + MAR | No | Shows temporal windowing helps, but inter-person variance still causes false alarms. |
| **DriveGuard-Scalar (`DG-S`)** | Forward-Only Personal Calibration | Yes (30s forward gaze) | Robust $z_{\text{EAR}} + z_{\text{PERCLOS}} + z_D$ | No | Demonstrates false-alarm reduction from personal Median/MAD scaling (RQ1). |
| **DriveGuard-Kinematic (`DG-K`)** | Pose-Conditioned + Blink Kinematics | Yes (45s 4-zone pose + kinematics) | Above + Pose normalization + Blink Reopening Ratio ($z_{\text{RCR}}$) | No | Demonstrates earlier detection of `LOW_VIGILANCE` and immunity to mirror checks (RQ2). |
| **DriveGuard-Full (`DG-Full`)** | Full Cascade System | Yes (45s 4-zone pose + kinematics) | Above + Confidence-Gated `MobileNetV3-Small` | Yes (On-Demand Gate) | Demonstrates robustness under glasses glare/dim lighting with minimal latency increase (RQ3). |

---

## 2. Dataset Partitioning & Anti-Leakage Protocol

1. **Subject-Disjoint Split (Mandatory):**
   - Never split frames from the same participant across training and testing sets.
   - Partition **UTA-RLDD (60 subjects)**:
     - **40 Subjects (Development/Validation Set):** Used to tune risk weights $(w_1 \dots w_6)$ or train the optional Random Forest/XGBoost aggregator.
     - **20 Subjects (Held-Out Test Set):** Strictly unseen during weight tuning.
2. **Within-Subject Personal Calibration Protocol on Held-Out Test Subjects:**
   - For each test subject $p \in \{1 \dots 20\}$:
     - Take the **first 45 seconds of their `Alert` video** to construct their Personalised Driver Behaviour Profile ($B_0^{(p)}$).
     - Evaluate detection performance on the **remaining portion of their `Alert` video** (to measure False Alarms per Hour) and on their complete **`Low Vigilance`** and **`Drowsy`** videos (to measure Recall, Precision, F1, and Early Detection Lead Time).

---

## 3. Primary Evaluation Metrics & Target Benchmarks

| Metric | Mathematical Definition / Method | Target Benchmark |
| :--- | :--- | :--- |
| **False Alarms per Hour (FA/h)** | Total `WARNING`/`CRITICAL` triggers during ground-truth `Alert` recordings normalized to 60 minutes. | $< 1.5\text{ FA/h}$ (vs. $> 8\text{ FA/h}$ for fixed EAR) |
| **Early Detection Lead Time ($\Delta t_{\text{lead}}$)** | Time difference between DriveGuard's first `WARNING` and the first prolonged microsleep ($> 800\text{ ms}$ eye closure). | $> 45\text{ seconds}$ before microsleep |
| **Subject-Wise Recall Variance ($\sigma_{\text{recall}}$)** | Standard deviation of recall across all 20 unseen test participants (measures fairness across eye shapes). | Reduced by $\ge 40\%$ vs. `B1` |
| **Cascade Trigger Rate (%)** | Percentage of frames where the `MobileNetV3-Small` gate is invoked. | $12\%\text{–}25\%$ of frames |
| **Server Inference Latency ($p_{50} / p_{95}$)** | Processing time per frame in FastAPI (`time.perf_counter()`). | $p_{50} < 15\text{ ms}$, $p_{95} < 30\text{ ms}$ |
| **End-to-End Alert Latency** | Client frame capture timestamp to browser alert render timestamp. | $< 120\text{ ms}$ on local/Wi-Fi, $< 220\text{ ms}$ on cloud |

---

## 4. 12-Week Work Breakdown Structure (WBS) & Deliverables Schedule

| Weeks | Phase | Concrete Engineering Tasks | Milestone Deliverable |
| :--- | :--- | :--- | :--- |
| **Weeks 1–2** | **Phase 1: Core CV & Kinematic Prototype** | Build standalone Python script with MediaPipe Face Landmarker, 3D EAR, MAR, Euler head pose, and spline-interpolated blink kinematics ($\text{RCR}$). | `research/prototype_webcam_cli.py` running at 30 FPS locally with live plots. |
| **Weeks 3–4** | **Phase 2: PDBP & Offline Dataset Benchmark** | Implement 4-pose calibration, Median/MAD robust $z$-scores, and hysteresis state machine. Run `B1`, `B2`, `DG-S`, `DG-K` on UTA-RLDD subset. | First quantitative table proving personalised calibration beats fixed thresholds. |
| **Weeks 5–6** | **Phase 3: `MobileNetV3-Small` Fine-Tuning & Cascade** | Fine-tune `MobileNetV3-Small` on MRL Eye Dataset, export to ONNX Runtime, and integrate the confidence-gated cascade (`DG-Full`). | Trained `.onnx` weights + confusion matrix & latency comparison script. |
| **Weeks 7–8** | **Phase 4: FastAPI Backend & Binary WebSocket** | Build FastAPI session registry, 16-byte binary WebSocket receiver, PostgreSQL async models, and stale-frame backpressure dropper. | Working backend passing automated load/session isolation tests. |
| **Weeks 9–10** | **Phase 5: Next.js Dashboard & Guided Calibration UI** | Build Next.js 15 app: 4-zone guided calibration wizard, live telemetry HUD, explainable alert cards, and side-by-side Fixed vs. Personalised demo mode. | End-to-end web application working on laptop webcam and smartphone browser. |
| **Weeks 11–12** | **Phase 6: Optional ESP32 + Final Thesis Experiments** | Flash ESP32 MQTT alert firmware (optional), run full 20-subject held-out evaluation + frame-rate sensitivity experiments, and write final thesis chapters. | Complete thesis evaluation plots, deployed demo, and defense presentation. |

---

## 5. High-Impact "Live Defense Demo" Feature Recommendation

Build a **"Dual-Engine Split View" (Academic Comparison Mode)** directly into the Next.js dashboard:
- On a single live webcam stream, run **Conventional Fixed Threshold (`EAR < 0.22`)** on the left card and **DriveGuard Personalised (`PDBP`)** on the right card simultaneously.
- During your project defense, have a tester with narrower eyes or glasses sit in front of the camera, check their side mirrors, and talk:
  - The **Left Card (Conventional)** will repeatedly flash false `DROWSY` alarms.
  - The **Right Card (DriveGuard)** will stay `NORMAL` during mirror checks and speech, then accurately trigger `WARNING` when the tester performs slow, heavy blinks.
- This single visual demonstration makes your research contribution immediately undeniable to any grading panel.
