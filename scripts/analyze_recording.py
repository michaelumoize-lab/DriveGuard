"""
DriveGuard — Session Recording Analysis & Architectural Anomaly Audit Tool.

Inspects recorded session telemetry (telemetry.csv) and flags:
1. Zone boundary chattering & rapid flickering (< 4 frames / < 120ms).
2. Asymmetric optical artifacts & false eye closures during mirror glances.
3. Kinematic angular velocity spikes (> 540 deg/s landmark snapping).
4. Tracking dropouts, prolonged face loss, and recovery latency.
5. Safety rule checks (Euro NCAP mirror glance grace periods).
6. Yawn vs. speech discrimination accuracy.
7. Lighting/sharpness correlation with tracking failures.

Usage:
    # Analyze the most recent recording session
    python scripts/analyze_recording.py

    # Analyze a specific session folder
    python scripts/analyze_recording.py --session data/recordings/session_20261119_153000
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class AnomalyEvent:
    frame_id: int
    timestamp_ms: float
    category: str
    severity: str  # "HIGH", "MEDIUM", "LOW"
    description: str
    observed_values: Dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""


class RecordingAnalyzer:
    """Performs static and temporal rule-based audit on recorded DriveGuard telemetry."""

    def __init__(self, session_dir: str | Path) -> None:
        self.session_dir = Path(session_dir)
        self.csv_path = self.session_dir / "telemetry.csv"
        self.meta_path = self.session_dir / "session_metadata.json"
        self.report_path = self.session_dir / "audit_report.md"

        if not self.csv_path.is_file():
            raise FileNotFoundError(f"Telemetry CSV not found in: {self.session_dir}")

        self.metadata: Dict[str, Any] = {}
        if self.meta_path.is_file():
            try:
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            except Exception:
                pass

        self.rows: List[Dict[str, Any]] = []
        self._load_csv()

        self.anomalies: List[AnomalyEvent] = []

    def _load_csv(self) -> None:
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                parsed: Dict[str, Any] = {}
                for k, v in r.items():
                    if v == "" or v is None:
                        parsed[k] = None
                    elif v in ("True", "true"):
                        parsed[k] = True
                    elif v in ("False", "false"):
                        parsed[k] = False
                    else:
                        try:
                            if "." in v:
                                parsed[k] = float(v)
                            else:
                                parsed[k] = int(v)
                        except ValueError:
                            parsed[k] = v
                self.rows.append(parsed)

    def analyze(self) -> List[AnomalyEvent]:
        """Runs all diagnostic checks across the telemetry dataset."""
        self.anomalies.clear()
        if not self.rows:
            return self.anomalies

        self._check_zone_chattering()
        self._check_asymmetric_mirror_closures()
        self._check_kinematic_velocity_spikes()
        self._check_tracking_dropouts()
        self._check_mirror_glance_safety_override()
        self._check_yawn_speech_discrimination()
        self._check_optical_degradation()

        # Sort anomalies by frame_id
        self.anomalies.sort(key=lambda a: a.frame_id)
        return self.anomalies

    def _check_zone_chattering(self) -> None:
        """Rule 1: Detects rapid zone oscillations (< 4 frames / ~120ms)."""
        n = len(self.rows)
        i = 0
        while i < n:
            zone = self.rows[i].get("pose_zone")
            if zone in (None, "UNKNOWN"):
                i += 1
                continue

            j = i
            while j < n and self.rows[j].get("pose_zone") == zone:
                j += 1
            run_length = j - i

            # If zone active for only 1 to 3 frames, check if surrounded by same zone
            if run_length < 4 and i > 0 and j < n:
                prev_zone = self.rows[i - 1].get("pose_zone")
                next_zone = self.rows[j].get("pose_zone")
                if prev_zone == next_zone and prev_zone not in (None, "UNKNOWN", zone):
                    yaw = self.rows[i].get("rel_yaw_deg", 0.0)
                    pitch = self.rows[i].get("rel_pitch_deg", 0.0)
                    self.anomalies.append(
                        AnomalyEvent(
                            frame_id=self.rows[i]["frame_id"],
                            timestamp_ms=self.rows[i]["timestamp_ms"],
                            category="ZONE_CHATTERING",
                            severity="LOW" if run_length > 1 else "MEDIUM",
                            description=f"Rapid zone flicker to '{zone}' for only {run_length} frame(s) within '{prev_zone}'.",
                            observed_values={
                                "transient_zone": zone,
                                "surrounding_zone": prev_zone,
                                "frames": run_length,
                                "rel_yaw_deg": yaw,
                                "rel_pitch_deg": pitch,
                            },
                            recommendation=f"Increase zone transition hysteresis deadband around yaw={yaw:.1f}° or pitch={pitch:.1f}°.",
                        )
                    )
            i = j

    def _check_asymmetric_mirror_closures(self) -> None:
        """Rule 2: Detects false eye closure during mirror glance due to foreshortening."""
        for r in self.rows:
            zone = r.get("pose_zone")
            is_closed = r.get("is_eye_closed")
            if zone in ("LEFT_MIRROR", "RIGHT_MIRROR") and is_closed:
                ear_l = r.get("ear_left")
                ear_r = r.get("ear_right")
                thr = r.get("ear_closure_threshold", 0.20)
                if ear_l is not None and ear_r is not None and thr is not None:
                    max_ear = max(ear_l, ear_r)
                    min_ear = min(ear_l, ear_r)
                    diff = max_ear - min_ear
                    # Genuine asymmetric foreshortening false-closure requires one eye to stay
                    # comfortably open above threshold while the other eye collapsed far below it.
                    if max_ear >= thr + 0.03 and diff >= 0.06:
                        open_eye = "LEFT_LANDMARK" if ear_l >= ear_r else "RIGHT_LANDMARK"
                        closed_eye = "RIGHT_LANDMARK" if ear_l >= ear_r else "LEFT_LANDMARK"
                        self.anomalies.append(
                            AnomalyEvent(
                                frame_id=r["frame_id"],
                                timestamp_ms=r["timestamp_ms"],
                                category="OPTICAL_FORESHORTENING_CLOSURE",
                                severity="HIGH",
                                description=(
                                    f"False eye closure flagged during {zone}. "
                                    f"One eye ({open_eye}) remained open (EAR={max_ear:.3f} >= thr={thr:.3f} + 0.03), "
                                    f"while the other ({closed_eye}) narrowed to {min_ear:.3f} (diff={diff:.3f})."
                                ),
                                observed_values={
                                    "zone": zone,
                                    "rel_yaw_deg": r.get("rel_yaw_deg"),
                                    "ear_left": ear_l,
                                    "ear_right": ear_r,
                                    "threshold": thr,
                                },
                                recommendation="Increase yaw-weighting sensitivity or verify unilateral wink exclusion during mirror glances.",
                            )
                        )

    def _check_kinematic_velocity_spikes(self) -> None:
        """Rule 3: Detects physically impossible angular velocity spikes (> 540 deg/s)."""
        for i in range(1, len(self.rows)):
            prev_r = self.rows[i - 1]
            curr_r = self.rows[i]
            if not prev_r.get("face_detected") or not curr_r.get("face_detected"):
                continue

            dt_ms = curr_r["timestamp_ms"] - prev_r["timestamp_ms"]
            if dt_ms <= 0 or dt_ms > 200:
                continue

            dyaw = abs(curr_r.get("yaw_deg", 0.0) - prev_r.get("yaw_deg", 0.0))
            dpitch = abs(curr_r.get("pitch_deg", 0.0) - prev_r.get("pitch_deg", 0.0))
            dt_s = dt_ms / 1000.0

            vel_yaw = dyaw / dt_s
            vel_pitch = dpitch / dt_s

            if vel_yaw > 540.0 or vel_pitch > 450.0:
                self.anomalies.append(
                    AnomalyEvent(
                        frame_id=curr_r["frame_id"],
                        timestamp_ms=curr_r["timestamp_ms"],
                        category="KINEMATIC_LANDMARK_JITTER",
                        severity="MEDIUM",
                        description=f"Angular velocity spike ({vel_yaw:.0f}°/s yaw, {vel_pitch:.0f}°/s pitch) over {dt_ms:.1f}ms.",
                        observed_values={
                            "yaw_delta": dyaw,
                            "pitch_delta": dpitch,
                            "dt_ms": dt_ms,
                            "blur_variance": curr_r.get("blur_variance"),
                        },
                        recommendation="Check video at this frame for sudden lighting flare, camera vibration, or facial occlusion.",
                    )
                )

    def _check_tracking_dropouts(self) -> None:
        """Rule 4: Detects prolonged face loss gaps (> 1.5s)."""
        n = len(self.rows)
        i = 0
        while i < n:
            if not self.rows[i].get("face_detected"):
                j = i
                while j < n and not self.rows[j].get("face_detected"):
                    j += 1
                duration_ms = self.rows[j - 1]["timestamp_ms"] - self.rows[i]["timestamp_ms"]
                frames_lost = j - i
                if duration_ms > 1500.0:
                    self.anomalies.append(
                        AnomalyEvent(
                            frame_id=self.rows[i]["frame_id"],
                            timestamp_ms=self.rows[i]["timestamp_ms"],
                            category="PROLONGED_FACE_LOSS",
                            severity="HIGH",
                            description=f"Prolonged face tracking loss for {duration_ms:.0f}ms ({frames_lost} frames).",
                            observed_values={
                                "lost_start_frame": self.rows[i]["frame_id"],
                                "lost_end_frame": self.rows[j - 1]["frame_id"],
                                "duration_ms": duration_ms,
                                "mean_luminance": self.rows[i].get("luminance"),
                            },
                            recommendation="Inspect camera placement or cabin illumination. Check if driver head rotated beyond camera field of view.",
                        )
                    )
                i = j
            else:
                i += 1

    def _check_mirror_glance_safety_override(self) -> None:
        """Rule 5: Checks if distraction warning fired during legitimate mirror check (< 1.5s)."""
        n = len(self.rows)
        i = 0
        while i < n:
            zone = self.rows[i].get("pose_zone")
            if zone in ("LEFT_MIRROR", "RIGHT_MIRROR"):
                j = i
                while j < n and self.rows[j].get("pose_zone") == zone:
                    j += 1
                dwell_ms = self.rows[j - 1]["timestamp_ms"] - self.rows[i]["timestamp_ms"]

                # If glance was brief (< 1500ms) but triggered WARNING or CRITICAL
                if dwell_ms <= 1500.0:
                    for k in range(i, j):
                        state = self.rows[k].get("drowsiness_state")
                        if state in ("WARNING", "CRITICAL"):
                            self.anomalies.append(
                                AnomalyEvent(
                                    frame_id=self.rows[k]["frame_id"],
                                    timestamp_ms=self.rows[k]["timestamp_ms"],
                                    category="FALSE_DISTRACTION_DURING_GLANCE",
                                    severity="HIGH",
                                    description=f"Distraction {state} fired during short {zone} check (dwell={dwell_ms:.0f}ms <= 1500ms).",
                                    observed_values={
                                        "zone": zone,
                                        "dwell_ms": dwell_ms,
                                        "reason": self.rows[k].get("primary_reason"),
                                    },
                                    recommendation="Verify mirror glance safety context freezes distraction accumulation during allowable glance window.",
                                )
                            )
                            break
                i = j
            else:
                i += 1

    def _check_yawn_speech_discrimination(self) -> None:
        """Rule 6: Audits MAR spikes against authentic yawn requirements."""
        n = len(self.rows)
        i = 0
        while i < n:
            mar = self.rows[i].get("mar", 0.0)
            if mar is not None and mar >= 0.45:
                j = i
                while j < n and (self.rows[j].get("mar") or 0.0) >= 0.45:
                    j += 1
                open_ms = self.rows[j - 1]["timestamp_ms"] - self.rows[i]["timestamp_ms"]
                any_yawn = any(self.rows[k].get("is_yawning") for k in range(i, j))

                # False positive: short mouth opening (< 1200ms, likely speech/talking) flagged as yawn
                if open_ms < 1200.0 and any_yawn:
                    self.anomalies.append(
                        AnomalyEvent(
                            frame_id=self.rows[i]["frame_id"],
                            timestamp_ms=self.rows[i]["timestamp_ms"],
                            category="YAWN_SPEECH_FALSE_POSITIVE",
                            severity="MEDIUM",
                            description=f"Short mouth opening ({open_ms:.0f}ms) incorrectly flagged as an authentic yawn.",
                            observed_values={"duration_ms": open_ms, "max_mar": max(self.rows[k].get("mar", 0) for k in range(i, j))},
                            recommendation="Enforce minimum physiological yawn opening envelope duration (>= 1.8s) to reject speech.",
                        )
                    )
                i = j
            else:
                i += 1

    def _check_optical_degradation(self) -> None:
        """Rule 7: Flags prolonged severe optical degradation (low light < 20 or blur < 40)."""
        low_light_count = sum(1 for r in self.rows if (r.get("luminance") or 100) < 20.0)
        blurry_count = sum(1 for r in self.rows if (r.get("blur_variance") or 500) < 40.0)
        total = len(self.rows)
        if total > 0:
            if (low_light_count / total) > 0.25:
                self.anomalies.append(
                    AnomalyEvent(
                        frame_id=1,
                        timestamp_ms=self.rows[0]["timestamp_ms"],
                        category="OPTICAL_LIGHTING_DEGRADATION",
                        severity="MEDIUM",
                        description=f"{low_light_count} frames ({low_light_count/total*100:.1f}%) suffered from extreme low luminance (< 20).",
                        observed_values={"low_light_ratio": low_light_count / total},
                        recommendation="Improve in-cabin illumination or enable near-infrared (NIR) camera lighting.",
                    )
                )
            if (blurry_count / total) > 0.25:
                self.anomalies.append(
                    AnomalyEvent(
                        frame_id=1,
                        timestamp_ms=self.rows[0]["timestamp_ms"],
                        category="OPTICAL_BLUR_DEGRADATION",
                        severity="MEDIUM",
                        description=f"{blurry_count} frames ({blurry_count/total*100:.1f}%) suffered from high motion blur (variance < 40).",
                        observed_values={"blurry_ratio": blurry_count / total},
                        recommendation="Tighten camera mount to reduce vibration or increase shutter speed/frame rate.",
                    )
                )

    def generate_report(self) -> str:
        """Builds a comprehensive markdown audit report."""
        total_frames = len(self.rows)
        duration_s = (
            (self.rows[-1]["timestamp_ms"] - self.rows[0]["timestamp_ms"]) / 1000.0
            if total_frames > 1
            else 0.0
        )
        avg_fps = (total_frames / duration_s) if duration_s > 0 else 0.0

        high_count = sum(1 for a in self.anomalies if a.severity == "HIGH")
        med_count = sum(1 for a in self.anomalies if a.severity == "MEDIUM")
        low_count = sum(1 for a in self.anomalies if a.severity == "LOW")

        lines = [
            f"# DriveGuard Telemetry Audit Report",
            f"**Session:** `{self.session_dir.name}`  ",
            f"**Driver ID:** `{self.metadata.get('driver_id', 'N/A')}`  ",
            f"**Total Frames:** {total_frames:,} ({duration_s:.1f}s @ {avg_fps:.1f} FPS)  ",
            "",
            "## 1. Executive Summary",
            "",
            f"| Metric | Value |",
            f"|---|---|",
            f"| Total Anomaly Events | **{len(self.anomalies)}** |",
            f"| High Severity (Safety/False Alarm Risks) | **{high_count}** |",
            f"| Medium Severity (Jitter/Flicker) | **{med_count}** |",
            f"| Low Severity (Micro-Chattering) | **{low_count}** |",
            "",
        ]

        if not self.anomalies:
            lines.extend([
                "> [!NOTE]",
                "> **Zero anomalies detected!** All physiological metrics, zone transitions, and kinematic velocities were fully consistent with DriveGuard safety specifications.",
                "",
            ])
        else:
            lines.extend([
                "## 2. Detected Anomaly Log",
                "",
                "| Frame ID | Time (s) | Severity | Category | Description | Recommendation |",
                "|:---:|:---:|:---:|:---|:---|:---|",
            ])
            for a in self.anomalies:
                t_sec = (a.timestamp_ms - self.rows[0]["timestamp_ms"]) / 1000.0
                sev_badge = f"**{a.severity}**" if a.severity == "HIGH" else a.severity
                lines.append(
                    f"| `{a.frame_id}` | {t_sec:.2f}s | {sev_badge} | `{a.category}` | {a.description} | {a.recommendation} |"
                )
            lines.append("")

            # Actionable engineering takeaways
            lines.extend([
                "## 3. Targeted Tuning Recommendations",
                "",
            ])
            cats = set(a.category for a in self.anomalies)
            if "ZONE_CHATTERING" in cats:
                lines.append("- **Cockpit Zone Hysteresis:** Some boundary transitions flickered. Consider widening the zone transition deadband from 1.5° to 2.0°.")
            if "OPTICAL_FORESHORTENING_CLOSURE" in cats:
                lines.append("- **Yaw-Weighted EAR Tuning:** During mirror glances, the far eye narrowed due to perspective. Increase weighting factor on the near eye.")
            if "KINEMATIC_LANDMARK_JITTER" in cats:
                lines.append("- **Camera Mount Stabilization:** Angular velocity spikes detected. Ensure the phone or camera mount is rigidly fixed.")
            if "FALSE_DISTRACTION_DURING_GLANCE" in cats:
                lines.append("- **Mirror Grace Window:** Ensure the distraction timer strictly respects the 1.5-second mirror check freeze.")
            if "YAWN_SPEECH_FALSE_POSITIVE" in cats:
                lines.append("- **Yawn Temporal Envelope:** Ensure transient mouth openings (< 1.5s) are suppressed as conversational speech.")
            lines.append("")

        report_content = "\n".join(lines)
        with open(self.report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        return report_content


def main() -> None:
    parser = argparse.ArgumentParser(description="DriveGuard Telemetry Audit Tool")
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="Path to session directory (defaults to newest in data/recordings/)",
    )
    args = parser.parse_args()

    session_path: Optional[Path] = None
    if args.session:
        session_path = Path(args.session)
    else:
        recordings_root = Path("data/recordings")
        if recordings_root.is_dir():
            subdirs = sorted([d for d in recordings_root.iterdir() if d.is_dir()], key=lambda d: d.name)
            if subdirs:
                session_path = subdirs[-1]

    if not session_path or not session_path.is_dir():
        print("[ERROR] No valid recording session found to analyze.")
        sys.exit(1)

    print(f"[*] Analyzing recording session: {session_path}")
    analyzer = RecordingAnalyzer(session_path)
    anomalies = analyzer.analyze()
    report = analyzer.generate_report()

    print("\n" + report)
    print(f"\n[OK] Full markdown audit report written to: {analyzer.report_path}")


if __name__ == "__main__":
    main()
