# DriveGuard — Experimental Evaluation & Ablation Results

## 1. Experiment 1: Fixed Threshold vs. 1-Zone vs. 4-Zone Pose-Conditioned Calibration (RQ1 & RQ2)

| Configuration | Total False Alarm Frames | Mirror-Check False Alarms | False Alarms / Hour | Precision (%) | Recall (%) | F1-Score (%) | Early Warning Lead Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Conventional Fixed (`EAR < 0.22`)** | 872 | 34 | 2104.6 | 55.8% | 100.0% | **71.6%** | 0.0 s |
| **1-Zone Personalised (`CENTER_ONLY`)** | 367 | 2 | 110.8 | 74.1% | 95.5% | **83.5%** | 37.5 s |
| **DriveGuard 4-Zone (`FOUR_ZONE` + Context)** | 0 | 0 | 0.0 | 100.0% | 92.1% | **95.9%** | 35.7 s |

## 2. Experiment 2: Confidence-Gated Cascade & Disagreement Arbitration (RQ3 & RQ4)

| Cascade Mode | Glare & Blur False Alarms | Total False Alarms | Precision (%) | Recall (%) | F1-Score (%) | CNN Trigger Rate (%) | Mean Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Experiment A: MediaPipe EAR + Blendshapes Only** | 370 | 537 | 67.2% | 100.0% | **80.4%** | 0.0% | 0.254 ms |
| **Experiment B: Stage-2 Verifier Always On (100%)** | 0 | 0 | 100.0% | 92.1% | **95.9%** | 100.0% | 1.090 ms |
| **Experiment C (Naive): Blindly Trust CNN on Disagreement** | 128 | 177 | 86.1% | 100.0% | **92.6%** | 40.0% | 0.592 ms |
| **Experiment C (DriveGuard): Reliability-Weighted Cascade** | 0 | 0 | 100.0% | 92.1% | **95.9%** | 40.0% | 0.596 ms |

## 3. Generated Thesis Figures (300 DPI)

- `fig1_ear_vs_zone_thresholds.png`: `C:\Users\Michael Umoize\Desktop\DriveGuard\backend\research\outputs\figures\fig1_ear_vs_zone_thresholds.png`
- `fig2_risk_timeline_and_early_warning.png`: `C:\Users\Michael Umoize\Desktop\DriveGuard\backend\research\outputs\figures\fig2_risk_timeline_and_early_warning.png`
- `fig3_calibration_false_alarm_comparison.png`: `C:\Users\Michael Umoize\Desktop\DriveGuard\backend\research\outputs\figures\fig3_calibration_false_alarm_comparison.png`
- `fig4_cascade_experiments_abc_tradeoff.png`: `C:\Users\Michael Umoize\Desktop\DriveGuard\backend\research\outputs\figures\fig4_cascade_experiments_abc_tradeoff.png`
