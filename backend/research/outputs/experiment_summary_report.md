# DriveGuard — Experimental Evaluation & Ablation Results

## 1. Experiment 1: Fixed Threshold vs. 1-Zone vs. 4-Zone Pose-Conditioned Calibration (RQ1 & RQ2)

| Configuration | Total False Alarm Frames | Mirror-Check False Alarms | False Alarms / Hour | Precision (%) | Recall (%) | F1-Score (%) | Early Warning Lead Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Conventional Fixed (`EAR < 0.22`)** | 956 | 34 | 1716.9 | 53.5% | 100.0% | **69.7%** | 0.0 s |
| **1-Zone Personalised (`CENTER_ONLY`)** | 899 | 2 | 166.2 | 55.0% | 100.0% | **71.0%** | 40.0 s |
| **DriveGuard 4-Zone (`FOUR_ZONE` + Context)** | 537 | 0 | 55.4 | 67.2% | 100.0% | **80.4%** | 40.0 s |

## 2. Experiment 2: Confidence-Gated Cascade & Disagreement Arbitration (RQ3 & RQ4)

| Cascade Mode | Glare & Blur False Alarms | Total False Alarms | Precision (%) | Recall (%) | F1-Score (%) | CNN Trigger Rate (%) | Mean Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Experiment A: MediaPipe EAR + Blendshapes Only** | 369 | 536 | 67.2% | 100.0% | **80.4%** | 0.0% | 0.816 ms |
| **Experiment B: Stage-2 Verifier Always On (100%)** | 370 | 537 | 67.2% | 100.0% | **80.4%** | 100.0% | 5.277 ms |
| **Experiment C (Naive): Blindly Trust CNN on Disagreement** | 370 | 537 | 67.2% | 100.0% | **80.4%** | 39.1% | 2.508 ms |
| **Experiment C (DriveGuard): Reliability-Weighted Cascade** | 370 | 537 | 67.2% | 100.0% | **80.4%** | 39.1% | 3.704 ms |

## 3. Generated Thesis Figures (300 DPI)

- `fig1_ear_vs_zone_thresholds.png`: `C:\Users\Michael Umoize\Desktop\DriveGuard\backend\research\outputs\figures\fig1_ear_vs_zone_thresholds.png`
- `fig2_risk_timeline_and_early_warning.png`: `C:\Users\Michael Umoize\Desktop\DriveGuard\backend\research\outputs\figures\fig2_risk_timeline_and_early_warning.png`
- `fig3_calibration_false_alarm_comparison.png`: `C:\Users\Michael Umoize\Desktop\DriveGuard\backend\research\outputs\figures\fig3_calibration_false_alarm_comparison.png`
- `fig4_cascade_experiments_abc_tradeoff.png`: `C:\Users\Michael Umoize\Desktop\DriveGuard\backend\research\outputs\figures\fig4_cascade_experiments_abc_tradeoff.png`
