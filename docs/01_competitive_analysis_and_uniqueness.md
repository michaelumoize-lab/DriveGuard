# Deliverable 1: GitHub Competitive Analysis & Uniqueness Strategy

**Project:** DriveGuard — A Personalised Adaptive AI Driver Drowsiness Detection and Early Warning System  
**Stage:** Planning & Architecture Specification  
**Date:** November 2026  

---

## 1. Executive Review of Your Current Plan

Your proposed technical specification is already in the **top 10% of undergraduate Computer Engineering proposals** because it avoids two common pitfalls:
1. It refuses to rely on a single static threshold (e.g., `EAR < 0.21`).
2. It separates **visual perception** (MediaPipe + fine-tuned `MobileNetV3-Small`) from **personal behavioral adaptation** (Personalised Driver Behaviour Profile using Median/MAD robust $z$-scores).

However, when external examiners or reviewers inspect a Computer Vision drowsiness project today, they compare it against existing open-source repositories on GitHub. Below is what currently exists online and where the hidden technical gaps are in your current specification.

---

## 2. Review of Existing GitHub Repositories & Literature

| Repository / Paper | Stack & Methods | What It Already Does | Critical Limitations (Your Opportunity) |
| :--- | :--- | :--- | :--- |
| **[`iice257/driver-monitor`](https://github.com/iice257/driver-monitor)** | Python, OpenCV, MediaPipe Face Landmarker (478 landmarks + 52 blendshapes), PyQt6 | Has **startup auto-calibration** (records open-eye EAR and neutral head pitch via `solvePnP`), computes PERCLOS, MAR yawning, EMA smoothing, and multi-level alerts. | 1. **Scalar calibration only**: Calibrates at $0^\circ$ forward gaze; looking at side mirrors drops 2D EAR and causes false positives.<br>2. **No CNN fallback**: Fails under thick glasses glare.<br>3. **No drift-safe baseline update**: Static after the first 3 seconds.<br>4. Desktop-only (no WebSocket/cloud session state or IoT hardware). |
| **[`rezaghoddoosian/Early-Drowsiness-Detection`](https://github.com/rezaghoddoosian/Early-Drowsiness-Detection)** *(Official UTA-RLDD Repo)* | Python, Dlib, Blink Kinematics, SVM / HM-LSTM | Extracts **blink closing velocity, reopening velocity, and Amplitude-Velocity Ratio (AVR)** to detect *early* low-vigilance before microsleeps happen. | 1. Offline video processing using outdated Dlib 68 landmarks.<br>2. No real-time web pipeline, no live guided calibration, and no explainable dashboard or hardware integration. |
| **[`berasteguip/driver-drowsiness-detector`](https://github.com/berasteguip/driver-drowsiness-detector)** | Python, modular tracking engines, user authentication | Separates user profiles and tracking engines cleanly. | Uses conventional frame-counting rules without robust statistical personalization ($z_r$) or blink kinematic analysis. |
| **[`Sania-hasann/Driver-Fatigue-Detection`](https://github.com/Sania-hasann/Driver-Fatigue-Detection)** & **[`Mustafaadel7/Eye-Blink-detection`](https://github.com/Mustafaadel7/Eye-Blink-detection.)** | OpenCV, MediaPipe / Dlib, EAR, MAR | Classic real-time EAR + MAR threshold counters. | Fixed universal thresholds (`EAR < 0.22` for $N$ consecutive frames). High false-alarm rate on naturally narrow-eyed drivers or when smiling/talking. |
| **Ersoy et al. (arXiv:2604.22479)** | Personalized EAR/MAR + CNN Verification | Pre-drive calibration to scale EAR/MAR cutoffs + CNN classifier on eye/mouth crops. | Does not model **pose-conditioned EAR distortion** or **irregular network frame arrival timestamps** over WebSockets. |

---

## 3. Five Concrete Innovations to Make DriveGuard Uniquely Defensible

Because repositories like `iice257/driver-monitor` already implement basic startup EAR scaling ($\text{Threshold} = 0.80 \times \text{EAR}_{\text{base}}$), a simple 30-second average of open-eye EAR is no longer enough to be truly unique. Implement these **five engineering differentiators** to make DriveGuard stand out academically and practically:

### Differentiator 1: Pose-Conditioned Personal Baseline $B(\theta_{\text{pitch}}, \theta_{\text{yaw}})$
* **The Flaw in Other Repos:** When a driver checks their left/right mirror (yaw $\pm 25^\circ$) or glances at the dashboard (pitch $-15^\circ$), perspective foreshortening changes the 2D landmark distances even if the eyes are wide open. Standard personalized systems trigger false alarms during mirror checks.
* **DriveGuard's Solution:** During the 45-second guided calibration, guide the driver through 4 quick head zones:
  1. **Center Road** ($0^\circ \text{ yaw}, 0^\circ \text{ pitch}$) — 25s natural blinking
  2. **Left Mirror** ($-25^\circ \text{ yaw}$) — 5s
  3. **Right Mirror** ($+25^\circ \text{ yaw}$) — 5s
  4. **Instrument Cluster** ($-15^\circ \text{ pitch}$) — 5s
* Store a **Pose-Conditioned Baseline Lookup / Regression Factor** $k(\theta_{\text{pitch}}, \theta_{\text{yaw}})$ so current EAR is normalized for head orientation before computing the deviation score $z_r$.

### Differentiator 2: Blink Kinematics for *True Early Warning* (Closing vs. Reopening Velocity)
* **The Flaw in Other Repos:** Waiting for a $900\text{ ms}$ prolonged closure means the driver is *already in a microsleep* (at $100\text{ km/h}$, a car travels $25\text{ meters}$ in $900\text{ ms}$).
* **DriveGuard's Solution:** Measure the first derivative of eye openness $\frac{d(\text{EAR})}{dt}$ during each blink event to capture:
  * **Eyelid Reopening Velocity ($v_{\text{open}}$):** Physiological studies (including the UTA-RLDD paper) prove that eyelid reopening slows down significantly during **early low-vigilance** long before prolonged closures occur.
  * **Blink Amplitude-Velocity Ratio (AVR):** Personalize AVR during calibration and use $z_{\text{AVR}}$ as your primary **Early Warning (`WARNING` state)** trigger.

### Differentiator 3: Confidence-Gated Hybrid Cascade (MediaPipe + On-Demand `MobileNetV3-Small`)
* **The Flaw in Your Current Plan:** Running `MobileNetV3-Small` on both eyes every frame wastes CPU, while running only EAR fails when glasses glare or low light distorts landmarks.
* **DriveGuard's Solution:** Implement an **Uncertainty-Gated Cascade**:
  1. Every frame runs MediaPipe Face Landmarker (extracting both **geometric EAR** and **MediaPipe's native `eyeBlinkLeft`/`eyeBlinkRight` blendshape scores**).
  2. Only trigger the fine-tuned `MobileNetV3-Small` eye-crop classifier when:
     * **Disagreement Gate:** Geometric EAR and Blendshape score disagree by $> 25\%$, OR
     * **Borderline Gate:** Robust deviation score $z_{\text{EAR}} \in [-1.5, -2.8]$, OR
     * **Illumination Gate:** Eye-crop histogram contrast/brightness indicates glare or dim lighting.
* **Academic Payoff:** You can benchmark **Cascade Mode** vs. **Always-On CNN** vs. **EAR-Only**, proving you achieve CNN-level accuracy at $< 20\%$ of the CNN compute cost.

### Differentiator 4: Timestamp-Interpolated Signal Reconstruction & Hybrid Edge Option
* **The Flaw in Section 6 of Your Plan:** At 15 FPS over WebSockets, frames are $66.7\text{ ms}$ apart. A $180\text{ ms}$ blink spans only 2–3 frames! Network jitter will cause severe quantization error in blink duration.
* **DriveGuard's Solution:**
  1. **Capture-Time Spline Reconstruction:** Attach a hardware capture timestamp (`performance.now()`) in an 8-byte binary header on every WebSocket JPEG frame. On the FastAPI backend, fit a **local quadratic/cubic spline** over the `(capture_timestamp, EAR)` buffer to estimate zero-crossings of the blink threshold with sub-frame ($\sim 10\text{ ms}$) resolution.
  2. **Optional Edge Telemetry Mode (Experiment 4):** Add a toggle in Next.js to run MediaPipe in the browser (WASM/WebGL) at 30 FPS and transmit only compact binary feature packets (`28 bytes/frame`) over WebSocket, comparing bandwidth and latency against raw JPEG streaming (`15-20 FPS`).

### Differentiator 5: Dual-Anchor Drift-Safe Adaptation & Alert Reaction Latency
* **The Flaw in Naive Adaptive Systems:** If a driver gradually gets tired over 90 minutes, an unconstrained moving average learns their fatigued state as the "new normal."
* **DriveGuard's Solution:**
  * **Dual-Anchor Profile:** Maintain an immutable **Calibration Anchor ($B_0$)** and an **Alert-Gated Rolling Baseline ($B_t$)** updated via an Exponentially Weighted Moving Median only when $R_t < 0.25$, camera quality is `GOOD`, and $|B_t - B_0| \le 0.10 B_0$ (hard $10\%$ drift clamp).
  * **Closed-Loop Vigilance Reaction Metric:** When a `WARNING` alert fires (via browser audio or ESP32 buzzer), DriveGuard measures **Driver Recovery Latency** (time until blink duration and head pose return to normal). Slow recovery ($> 4\text{ seconds}$) immediately escalates to `CRITICAL`.
