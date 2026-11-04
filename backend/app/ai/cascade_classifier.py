"""
Confidence-Gated Eye Classifier Cascade & Experimental Benchmark Engine.

Supports the three experimental evaluation modes defined in the specification:
- ExperimentMode.A_GEOMETRIC_ONLY: MediaPipe EAR + Blendshapes only (fast first-stage)
- ExperimentMode.B_CNN_ALWAYS: MobileNetV3-Small runs on every valid frame
- ExperimentMode.C_CONFIDENCE_GATED_CASCADE: Runs MobileNetV3-Small only when geometric
  confidence is low, EAR and blendshapes disagree, or EAR is in the borderline zone.

Crucial Design Principle:
When the cascade triggers due to disagreement, it does NOT blindly assume the CNN is right.
Instead, it evaluates `geometric_reliability` vs `cnn_reliability` (derived from eye-crop
sharpness, exposure, glare saturation, and classifier margin) and performs Reliability-Weighted
Arbitration while logging an audit trail to evaluate gating performance.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Deque, Optional, Tuple

import cv2
import numpy as np

from backend.app.ai.geometry import EyeCropQuality, FrameGeometryFeatures


class ExperimentMode(str, Enum):
    """Experimental modes for benchmarking the eye-state analysis pipeline."""

    A_GEOMETRIC_ONLY = "EXPERIMENT_A_GEOMETRIC_ONLY"
    B_CNN_ALWAYS = "EXPERIMENT_B_CNN_ALWAYS"
    C_CONFIDENCE_GATED_CASCADE = "EXPERIMENT_C_CONFIDENCE_GATED_CASCADE"


@dataclass
class DisagreementAuditRecord:
    """Logged whenever geometric analysis and MobileNetV3-Small disagree significantly."""

    timestamp_ms: float
    gate_reason: str
    geom_closure_score: float
    geom_reliability: float
    cnn_closed_prob: float
    cnn_reliability: float
    fused_closure_score: float
    winning_authority: str  # "GEOMETRIC", "CNN", or "WEIGHTED_CONSENSUS"


@dataclass
class CascadeStageResult:
    """Output of the Confidence-Gated Eye Classifier Cascade for a single frame."""

    experiment_mode: ExperimentMode
    geom_closure_score: float
    geom_reliability: float
    disagreement_ear_vs_blendshape: float
    cnn_invoked: bool
    gate_trigger_reason: str
    cnn_closed_prob: Optional[float]
    cnn_open_prob: Optional[float]
    cnn_reliability: Optional[float]
    fused_closure_score: float
    winning_authority: str
    stage_latency_ms: float


DEFAULT_EYE_ONNX_PATH = Path(__file__).resolve().parents[2] / "weights" / "mobilenetv3_eye.onnx"


class MobileNetV3EyeVerifier:
    """
    Lightweight Eye-Crop Classifier (OPEN vs CLOSED).
    Uses an ONNX Runtime session if `weights/mobilenetv3_eye.onnx` is present,
    or falls back to an analytical pupil/eyelid intensity-gradient crop verifier
    for immediate Stage 1 webcam prototyping before MRL Eye Dataset fine-tuning.
    """

    def __init__(self, onnx_model_path: Optional[str | Path] = None) -> None:
        self.onnx_session = None
        self.input_name: Optional[str] = None
        self.is_pretrained_onnx: bool = False

        resolved_path = Path(onnx_model_path) if onnx_model_path is not None else DEFAULT_EYE_ONNX_PATH
        if resolved_path.is_file():
            try:
                import onnxruntime as ort

                self.onnx_session = ort.InferenceSession(
                    str(resolved_path), providers=["CPUExecutionProvider"]
                )
                self.input_name = self.onnx_session.get_inputs()[0].name
                self.is_pretrained_onnx = True
            except Exception:
                self.onnx_session = None

    def predict_eye_crop(self, crop_quality: Optional[EyeCropQuality]) -> Tuple[float, float, float]:
        """
        Predict (open_probability, closed_probability, cnn_reliability) for a single eye crop.
        """
        if crop_quality is None or crop_quality.crop_rgb is None or crop_quality.overall_reliability < 0.15:
            return 0.5, 0.5, 0.0

        crop = crop_quality.crop_rgb

        if self.onnx_session is not None and self.input_name is not None:
            # Convert RGB webcam crop to 3-channel luminance to bridge the IR (MRL dataset) <-> RGB (webcam) domain gap
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            lum3 = np.stack([gray, gray, gray], axis=-1).astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = (lum3 - mean) / std
            tensor = np.transpose(img, (2, 0, 1))[np.newaxis, ...]
            logits = self.onnx_session.run(None, {self.input_name: tensor})[0][0]
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            open_prob = float(probs[0])
            closed_prob = float(probs[1])
        else:
            # Prototype stage fallback: analyze dark iris/pupil visibility & vertical vs horizontal Sobel energy
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            h, w = gray.shape
            center_band = gray[int(h * 0.25) : int(h * 0.75), int(w * 0.20) : int(w * 0.80)]

            # Open eyes exhibit dark pupil/iris core and strong vertical edge contrast around iris boundary
            p15 = float(np.percentile(center_band, 15))
            p85 = float(np.percentile(center_band, 85))
            contrast = max(0.0, p85 - p15) / 255.0

            sobel_x = cv2.Sobel(center_band, cv2.CV_32F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(center_band, cv2.CV_32F, 0, 1, ksize=3)
            energy_x = float(np.mean(np.abs(sobel_x))) + 1e-4
            energy_y = float(np.mean(np.abs(sobel_y))) + 1e-4
            # Closed eyelids have predominantly horizontal crease edges (energy_y >> energy_x)
            hv_ratio = energy_x / energy_y

            open_score = float(np.clip(1.6 * contrast + 0.65 * (hv_ratio - 0.35), 0.02, 0.98))
            open_prob = open_score
            closed_prob = 1.0 - open_prob

        # Classifier certainty margin in [0, 1]
        margin = abs(closed_prob - open_prob)
        cnn_reliability = float(np.clip(crop_quality.overall_reliability * (0.55 + 0.45 * margin), 0.0, 1.0))
        return open_prob, closed_prob, cnn_reliability

    def predict_binocular(
        self,
        left_crop: Optional[EyeCropQuality],
        right_crop: Optional[EyeCropQuality],
    ) -> Tuple[float, float, float]:
        """Combine predictions across left and right eye crops weighted by crop quality."""
        l_open, l_closed, l_rel = self.predict_eye_crop(left_crop)
        r_open, r_closed, r_rel = self.predict_eye_crop(right_crop)

        total_rel = l_rel + r_rel
        if total_rel < 1e-5:
            return 0.5, 0.5, 0.0

        closed_prob = (l_closed * l_rel + r_closed * r_rel) / total_rel
        open_prob = 1.0 - closed_prob
        combined_rel = max(l_rel, r_rel) * 0.7 + min(l_rel, r_rel) * 0.3
        return float(open_prob), float(closed_prob), float(combined_rel)


class ConfidenceGatedEyeCascade:
    """
    Executes the two-stage Confidence-Gated Eye Classification pipeline and records
    experimental telemetry for comparing Experiments A, B, and C.
    """

    def __init__(
        self,
        mode: ExperimentMode = ExperimentMode.C_CONFIDENCE_GATED_CASCADE,
        onnx_model_path: Optional[str | Path] = None,
        min_geom_reliability_gate: float = 0.68,
        max_blendshape_disagreement_gate: float = 0.24,
    ) -> None:
        self.mode = mode
        self.verifier = MobileNetV3EyeVerifier(onnx_model_path=onnx_model_path)
        self.min_geom_reliability_gate = min_geom_reliability_gate
        self.max_blendshape_disagreement_gate = max_blendshape_disagreement_gate

        # Telemetry & audit trail counters
        self.total_frames: int = 0
        self.cnn_invocations: int = 0
        self.disagreement_events: Deque[DisagreementAuditRecord] = deque(maxlen=300)

    @property
    def cnn_trigger_rate(self) -> float:
        if self.total_frames == 0:
            return 0.0
        return self.cnn_invocations / self.total_frames

    def evaluate(
        self,
        geom: FrameGeometryFeatures,
        reference_open_ear: float,
        close_threshold_ear: float,
    ) -> CascadeStageResult:
        """
        Run the configured cascade mode (`A_GEOMETRIC_ONLY`, `B_CNN_ALWAYS`, or
        `C_CONFIDENCE_GATED_CASCADE`) on the current frame's measurements.
        """
        t0 = time.perf_counter()
        self.total_frames += 1

        if not geom.face_detected:
            return CascadeStageResult(
                experiment_mode=self.mode,
                geom_closure_score=0.0,
                geom_reliability=0.0,
                disagreement_ear_vs_blendshape=1.0,
                cnn_invoked=False,
                gate_trigger_reason="NO_FACE",
                cnn_closed_prob=None,
                cnn_open_prob=None,
                cnn_reliability=None,
                fused_closure_score=0.0,
                winning_authority="NONE",
                stage_latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        # 1. Compute continuous geometric eye-closure score in [0, 1]
        denom = max(0.04, reference_open_ear - (close_threshold_ear * 0.75))
        ear_closure = float(np.clip((reference_open_ear - geom.yaw_weighted_ear) / denom, 0.0, 1.0))

        if geom.blendshape_blink_mean is not None:
            bs_closure = float(np.clip(geom.blendshape_blink_mean, 0.0, 1.0))
            disagreement = abs(ear_closure - bs_closure)
            geom_closure_score = 0.65 * ear_closure + 0.35 * bs_closure
        else:
            disagreement = 0.0
            geom_closure_score = ear_closure

        # 2. Decide whether to invoke Stage 2 (MobileNetV3-Small Eye Verifier)
        should_invoke_cnn = False
        gate_reason = "HIGH_GEOM_CONFIDENCE"

        if self.mode == ExperimentMode.A_GEOMETRIC_ONLY:
            should_invoke_cnn = False
            gate_reason = "EXP_A_GEOM_ONLY"
        elif self.mode == ExperimentMode.B_CNN_ALWAYS:
            should_invoke_cnn = True
            gate_reason = "EXP_B_ALWAYS_ON"
        else:
            # Experiment C: Confidence-Gated Cascade
            borderline_ear = abs(geom.yaw_weighted_ear - close_threshold_ear) <= 0.028
            low_crop_glare = (
                (geom.left_eye_crop is not None and geom.left_eye_crop.glare_fraction > 0.06)
                or (geom.right_eye_crop is not None and geom.right_eye_crop.glare_fraction > 0.06)
            )
            if geom.geometric_reliability < self.min_geom_reliability_gate:
                should_invoke_cnn = True
                gate_reason = "LOW_GEOMETRIC_RELIABILITY"
            elif disagreement > self.max_blendshape_disagreement_gate:
                should_invoke_cnn = True
                gate_reason = "EAR_BLENDSHAPE_DISAGREEMENT"
            elif borderline_ear:
                should_invoke_cnn = True
                gate_reason = "BORDERLINE_CLOSURE_THRESHOLD"
            elif low_crop_glare:
                should_invoke_cnn = True
                gate_reason = "GLASSES_GLARE_DETECTED"

        if not should_invoke_cnn:
            return CascadeStageResult(
                experiment_mode=self.mode,
                geom_closure_score=geom_closure_score,
                geom_reliability=geom.geometric_reliability,
                disagreement_ear_vs_blendshape=disagreement,
                cnn_invoked=False,
                gate_trigger_reason=gate_reason,
                cnn_closed_prob=None,
                cnn_open_prob=None,
                cnn_reliability=None,
                fused_closure_score=geom_closure_score,
                winning_authority="GEOMETRIC",
                stage_latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        # 3. Run Stage 2 Classifier & Reliability-Weighted Arbitration
        self.cnn_invocations += 1
        open_prob, closed_prob, cnn_rel = self.verifier.predict_binocular(
            geom.left_eye_crop, geom.right_eye_crop
        )

        g_rel = max(0.05, geom.geometric_reliability)
        c_rel = max(0.05, cnn_rel)

        # Crucial requirement: Do NOT blindly assume the CNN is right when disagreement occurs!
        # Weight both sources by their measured reliability scores.
        if c_rel < 0.25 and g_rel >= 0.50:
            # Eye crop is too blurry/dark for CNN to be trusted
            fused_closure = geom_closure_score
            authority = "GEOMETRIC_FALLBACK_BLURRY_CROP"
        elif g_rel > c_rel * 1.35:
            fused_closure = 0.72 * geom_closure_score + 0.28 * closed_prob
            authority = "GEOMETRIC"
        elif c_rel > g_rel * 1.35:
            fused_closure = 0.28 * geom_closure_score + 0.72 * closed_prob
            authority = "CNN"
        else:
            fused_closure = (g_rel * geom_closure_score + c_rel * closed_prob) / (g_rel + c_rel)
            authority = "WEIGHTED_CONSENSUS"

        if abs(geom_closure_score - closed_prob) >= 0.30:
            self.disagreement_events.append(
                DisagreementAuditRecord(
                    timestamp_ms=geom.timestamp_ms,
                    gate_reason=gate_reason,
                    geom_closure_score=geom_closure_score,
                    geom_reliability=g_rel,
                    cnn_closed_prob=closed_prob,
                    cnn_reliability=c_rel,
                    fused_closure_score=float(fused_closure),
                    winning_authority=authority,
                )
            )

        return CascadeStageResult(
            experiment_mode=self.mode,
            geom_closure_score=geom_closure_score,
            geom_reliability=geom.geometric_reliability,
            disagreement_ear_vs_blendshape=disagreement,
            cnn_invoked=True,
            gate_trigger_reason=gate_reason,
            cnn_closed_prob=closed_prob,
            cnn_open_prob=open_prob,
            cnn_reliability=cnn_rel,
            fused_closure_score=float(np.clip(fused_closure, 0.0, 1.0)),
            winning_authority=authority,
            stage_latency_ms=(time.perf_counter() - t0) * 1000.0,
        )
