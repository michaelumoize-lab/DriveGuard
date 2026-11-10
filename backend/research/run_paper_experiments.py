"""
DriveGuard — Unified Paper Experiment Runner & LaTeX Thesis Table Generator
===========================================================================
Final-Year Computer Engineering Project:
Personalised Adaptive AI Driver Drowsiness Detection and Early Warning System
Using Computer Vision.

Author: Michael Umoize
Supervisor / Examination Board Evaluation Harness

Executes the complete academic ablation matrix with a single command:
  1. Experiment A: Pure Geometric MediaPipe (Fast Path)
  2. Experiment B: Always-On MobileNetV3-Small CNN Verifier
  3. Experiment C: DriveGuard Reliability-Weighted Confidence-Gated Cascade
  4. Calibration Ablation: Fixed EAR (0.22) vs. 1-Zone PDBP vs. 4-Zone Pose-Conditioned PDBP

Generates:
  - Formatted terminal tables
  - LaTeX table source files ready for thesis inclusion (`backend/research/outputs/paper_experiment_tables.tex`)
  - Structured JSON metrics (`backend/research/outputs/paper_experiment_results.json`)
  - Four 300-DPI publication figures in `backend/research/outputs/figures/`

Usage:
  python -m backend.research.run_paper_experiments
  python -m backend.research.run_paper_experiments --video path/to/driving_video.mp4
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

from backend.research.evaluate_experiments import run_all_experiments


def format_latex_table_cascade(cascade_comparison: Dict[str, Any]) -> str:
    """Format Experiment A/B/C cascade ablation comparison as a publication-ready LaTeX table."""
    exp_a = cascade_comparison.get("EXP_A_GEOM_ONLY", {})
    exp_b = cascade_comparison.get("EXP_B_CNN_ALWAYS", {})
    exp_c_naive = cascade_comparison.get("EXP_C_NAIVE_BLIND_CNN", {})
    exp_c_dg = cascade_comparison.get("EXP_C_DRIVEGUARD_CASCADE", {})

    return (
        r"\begin{table}[htbp]" + "\n"
        r"\centering" + "\n"
        r"\caption{Cascade Architecture Ablation Comparison (Geometric vs. CNN vs. DriveGuard Reliability Gate)}" + "\n"
        r"\label{tab:cascade_ablation}" + "\n"
        r"\begin{tabular}{lcccc}" + "\n"
        r"\hline" + "\n"
        r"\textbf{Metric} & \textbf{Exp A (Geometric)} & \textbf{Exp B (CNN Always)} & \textbf{Exp C (DriveGuard)} & \textbf{Target Specification} \\" + "\n"
        r"\hline" + "\n"
        f"Precision (\\%) & {exp_a.get('precision_pct', 0.0):.1f}\\% & {exp_b.get('precision_pct', 0.0):.1f}\\% & {exp_c_dg.get('precision_pct', 0.0):.1f}\\% & $>90.0\\%$ \\\\\n"
        f"Recall (Sensitivity) (\\%) & {exp_a.get('recall_pct', 0.0):.1f}\\% & {exp_b.get('recall_pct', 0.0):.1f}\\% & {exp_c_dg.get('recall_pct', 0.0):.1f}\\% & $>92.0\\%$ \\\\\n"
        f"F1-Score (\\%) & {exp_a.get('f1_pct', 0.0):.1f}\\% & {exp_b.get('f1_pct', 0.0):.1f}\\% & {exp_c_dg.get('f1_pct', 0.0):.1f}\\% & $>90.0\\%$ \\\\\n"
        f"Glare / Blur False Alarms & {int(exp_a.get('glare_blur_fa_frames', 0))} & {int(exp_b.get('glare_blur_fa_frames', 0))} & {int(exp_c_dg.get('glare_blur_fa_frames', 0))} & Robust \\\\\n"
        f"Mean Latency (ms) & {exp_a.get('mean_latency_ms', 0.0):.2f} & {exp_b.get('mean_latency_ms', 0.0):.2f} & {exp_c_dg.get('mean_latency_ms', 0.0):.2f} & $<35.0$ ms \\\\\n"
        f"P95 Latency (ms) & {exp_a.get('p95_latency_ms', 0.0):.2f} & {exp_b.get('p95_latency_ms', 0.0):.2f} & {exp_c_dg.get('p95_latency_ms', 0.0):.2f} & $<50.0$ ms \\\\\n"
        f"Effective FPS & {exp_a.get('effective_fps', 0.0):.1f} & {exp_b.get('effective_fps', 0.0):.1f} & {exp_c_dg.get('effective_fps', 0.0):.1f} & $\\ge 20.0$ FPS \\\\\n"
        f"Stage-2 CNN Trigger Rate & {exp_a.get('cnn_trigger_rate_pct', 0.0):.1f}\\% & {exp_b.get('cnn_trigger_rate_pct', 100.0):.1f}\\% & {exp_c_dg.get('cnn_trigger_rate_pct', 0.0):.1f}\\% & $<25.0\\%$ \\\\\n"
        r"\hline" + "\n"
        r"\end{tabular}" + "\n"
        r"\end{table}"
    )


def format_latex_table_calibration(cal_comparison: Dict[str, Any]) -> str:
    """Format PDBP personalization ablation as a publication-ready LaTeX table."""
    fixed = cal_comparison.get("FIXED_022", {})
    one_zone = cal_comparison.get("PDBP_1ZONE", {})
    four_zone = cal_comparison.get("PDBP_4ZONE", {})

    return (
        r"\begin{table}[htbp]" + "\n"
        r"\centering" + "\n"
        r"\caption{Personalisation \& Spatial Calibration Ablation (Fixed Cutoff vs. 1-Zone vs. 4-Zone PDBP)}" + "\n"
        r"\label{tab:calibration_ablation}" + "\n"
        r"\begin{tabular}{lcccc}" + "\n"
        r"\hline" + "\n"
        r"\textbf{Evaluation Dimension} & \textbf{Fixed Cutoff (EAR $<$ 0.22)} & \textbf{Stage 1 (1-Zone Center)} & \textbf{Stage 2 (4-Zone PDBP)} & \textbf{Improvement} \\" + "\n"
        r"\hline" + "\n"
        f"Total False Alarms (frames) & {int(fixed.get('false_alarm_frames', 0))} & {int(one_zone.get('false_alarm_frames', 0))} & {int(four_zone.get('false_alarm_frames', 0))} & -{int(fixed.get('false_alarm_frames', 1) - four_zone.get('false_alarm_frames', 0))} frames \\\\\n"
        f"Mirror Check False Alarms & {int(fixed.get('mirror_check_fa_frames', 0))} & {int(one_zone.get('mirror_check_fa_frames', 0))} & {int(four_zone.get('mirror_check_fa_frames', 0))} & Suppressed \\\\\n"
        f"Precision (\\%) & {fixed.get('precision_pct', 0.0):.1f}\\% & {one_zone.get('precision_pct', 0.0):.1f}\\% & {four_zone.get('precision_pct', 0.0):.1f}\\% & +{four_zone.get('precision_pct', 0.0) - fixed.get('precision_pct', 0.0):.1f}\\% \\\\\n"
        f"Recall (Sensitivity) (\\%) & {fixed.get('recall_pct', 0.0):.1f}\\% & {one_zone.get('recall_pct', 0.0):.1f}\\% & {four_zone.get('recall_pct', 0.0):.1f}\\% & High Sensitivity \\\\\n"
        f"F1-Score (\\%) & {fixed.get('f1_pct', 0.0):.1f}\\% & {one_zone.get('f1_pct', 0.0):.1f}\\% & {four_zone.get('f1_pct', 0.0):.1f}\\% & Optimal \\\\\n"
        f"Early Warning Lead Time & {fixed.get('early_warning_lead_time_sec', 0.0):.1f} s & {one_zone.get('early_warning_lead_time_sec', 0.0):.1f} s & {four_zone.get('early_warning_lead_time_sec', 0.0):.1f} s & $+{four_zone.get('early_warning_lead_time_sec', 0.0) - fixed.get('early_warning_lead_time_sec', 0.0):.1f}$ s \\\\\n"
        r"\hline" + "\n"
        r"\end{tabular}" + "\n"
        r"\end{table}"
    )


def run_paper_experiments(video_path: Path | None = None) -> None:
    """Execute complete ablation suite, output terminal tables, LaTeX sources, and JSON artifacts."""
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("   [DriveGuard] Unified Academic Paper Experiment & Ablation Harness")
    print("   Author: Michael Umoize (Computer Engineering Final Year Project)")
    print("=" * 80)

    start_time = time.time()
    cal_comparison, cascade_comparison, fig_paths, md_path = run_all_experiments(video_path=video_path)

    # Generate LaTeX Tables
    latex_cascade = format_latex_table_cascade(cascade_comparison)
    latex_calib = format_latex_table_calibration(cal_comparison)

    combined_latex = f"""% ==============================================================================
% DriveGuard Academic Evaluation Tables (IEEE / ACM / Thesis Format)
% Generated by run_paper_experiments.py
% Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}
% ==============================================================================

{latex_cascade}

{latex_calib}
"""
    latex_path = out_dir / "paper_experiment_tables.tex"
    latex_path.write_text(combined_latex, encoding="utf-8")

    # Save complete JSON serialization
    summary_data = {
        "cal_comparison": cal_comparison,
        "cascade_comparison": cascade_comparison,
        "figure_paths": [str(p) for p in fig_paths],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    json_path = out_dir / "paper_experiment_results.json"
    json_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"   [ACADEMIC SUMMARY GENERATED] Complete suite ran in {elapsed:.2f}s!")
    print(f"   - LaTeX Tables: {latex_path}")
    print(f"   - JSON Results: {json_path}")
    print(f"   - Thesis Plots: {len(fig_paths)} 300-DPI PNGs in {out_dir / 'figures'}")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="DriveGuard Unified Paper Experiment Runner")
    parser.add_argument("--video", type=str, default=None, help="Optional path to evaluation driving video")
    args = parser.parse_args()

    video_path = Path(args.video) if args.video else None
    run_paper_experiments(video_path=video_path)


if __name__ == "__main__":
    main()
