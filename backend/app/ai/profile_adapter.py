"""
DriveGuard Profile Adapter Module.
Provides pure, bidirectional, schema-validated conversion between DriverBehaviourProfile
(runtime AI dataclass) and DriverProfileDTO (Pydantic persistence/API schema).

Architecture Rules:
1. Pure functions: conversion never mutates source objects.
2. Authority: ZoneBaseline.dynamic_closure_threshold property is the runtime physiological
   single source of truth. The stored DTO threshold serves as a transmission check and is
   validated against the authoritative formula with tolerance epsilon = 0.0005.
3. Timestamp integrity: conversion never invents or normalizes timestamps.
4. Drift reset: dto_to_profile strictly resets active_center_ear_median to center_ear_median (B0).
5. Exact mode preservation: "4ZONE", "FOUR_ZONE", and other valid modes are preserved verbatim.
"""

from __future__ import annotations

import copy
from typing import Dict

import numpy as np

from backend.app.features.head_pose import PoseZone
from backend.app.ai.pdbp_calibration import DriverBehaviourProfile, ZoneBaseline
from backend.app.schemas.profile import DriverProfileDTO, ZoneBaselineDTO

# Maximum allowable floating-point and 4-decimal serialization rounding discrepancy
# Derived mathematically from |m_dto - m| + 2.3 * |s_dto - s| + |T_stored - T_runtime| <= 0.000215
DYNAMIC_CLOSURE_THRESHOLD_TOLERANCE = 0.0005


def profile_to_dto(profile: DriverBehaviourProfile) -> DriverProfileDTO:
    """
    Convert an in-memory DriverBehaviourProfile dataclass into a validated DriverProfileDTO.

    Does not mutate the source profile.
    """
    zone_dtos: Dict[str, ZoneBaselineDTO] = {}
    for zk, zb in profile.zone_baselines.items():
        zk_name = zk.value if isinstance(zk, PoseZone) else str(zk)
        zone_dtos[zk_name] = ZoneBaselineDTO(
            zone=zk_name,
            ear_median=round(zb.ear_median, 4),
            ear_mad_scaled=round(zb.ear_mad_scaled, 4),
            dynamic_closure_threshold=round(zb.dynamic_closure_threshold, 4),
            pitch_median_deg=round(zb.pitch_median_deg, 2),
            yaw_median_deg=round(zb.yaw_median_deg, 2),
            sample_count=zb.sample_count,
            ear_left_median=round(zb.ear_left_median, 4) if zb.ear_left_median is not None else None,
            ear_right_median=round(zb.ear_right_median, 4) if zb.ear_right_median is not None else None,
            blendshape_open_median=round(zb.blendshape_open_median, 3) if zb.blendshape_open_median is not None else None,
            confidence_median=round(zb.confidence_median, 3) if zb.confidence_median is not None else None,
        )

    is_calibrated = (profile.calibration_mode != "UNCALIBRATED_DEFAULT")

    return DriverProfileDTO(
        version=profile.version,
        driver_id=profile.driver_id,
        calibrated_at=profile.calibrated_at,
        is_calibrated=is_calibrated,
        calibration_mode=profile.calibration_mode,
        zone_baselines=zone_dtos,
        center_ear_median=round(profile.center_ear_median, 4),
        blink_duration_median_ms=round(profile.blink_duration_median_ms, 2),
        blink_duration_mad_ms=round(profile.blink_duration_mad_ms, 2),
        blink_rcr_median=round(profile.blink_rcr_median, 4),
        blink_rcr_mad=round(profile.blink_rcr_mad, 4),
        blink_freq_median_per_min=round(profile.blink_freq_median_per_min, 2),
        blink_freq_mad_per_min=round(profile.blink_freq_mad_per_min, 2),
        baseline_perclos_pct=round(profile.baseline_perclos_pct, 2),
        blink_amplitude_median=round(profile.blink_amplitude_median, 4),
        temporal_blinks_sampled=profile.temporal_blinks_sampled,
        temporal_window_sec=round(profile.temporal_window_sec, 1),
        sensitivity_level=profile.sensitivity_level,
        neutral_pitch_deg=round(profile.neutral_pitch_deg, 2),
        neutral_yaw_deg=round(profile.neutral_yaw_deg, 2),
        calibration_reliability=round(profile.calibration_reliability, 3),
        neutral_rot_mat=(
            [[round(float(val), 6) for val in row] for row in profile.neutral_rot_mat]
            if getattr(profile, "neutral_rot_mat", None) is not None
            else None
        ),
        calibration_is_mirrored=getattr(profile, "calibration_is_mirrored", None),
    )


def dto_to_profile(dto: DriverProfileDTO) -> DriverBehaviourProfile:
    """
    Convert a validated DriverProfileDTO into an in-memory DriverBehaviourProfile dataclass.

    Validates dynamic closure threshold consistency against the physiological formula.
    Resets active_center_ear_median strictly to the B0 anchor (center_ear_median).
    Does not mutate the source DTO.
    """
    baselines: Dict[str, ZoneBaseline] = {}

    for zk_str, zb_dto in dto.zone_baselines.items():
        if zk_str.startswith("_"):
            continue

        # Validate stored dynamic_closure_threshold against authoritative formula
        raw_expected = min(zb_dto.ear_median * 0.76, zb_dto.ear_median - 2.3 * zb_dto.ear_mad_scaled)
        expected_threshold = float(np.clip(raw_expected, 0.12, 0.22))
        discrepancy = abs(zb_dto.dynamic_closure_threshold - expected_threshold)

        if discrepancy > DYNAMIC_CLOSURE_THRESHOLD_TOLERANCE:
            raise ValueError(
                f"Materially inconsistent dynamic_closure_threshold for zone '{zk_str}': "
                f"stored={zb_dto.dynamic_closure_threshold}, expected={expected_threshold:.4f} "
                f"from ear_median={zb_dto.ear_median} and ear_mad_scaled={zb_dto.ear_mad_scaled} "
                f"(discrepancy={discrepancy:.6f} > tolerance={DYNAMIC_CLOSURE_THRESHOLD_TOLERANCE})"
            )

        # ZoneBaseline dynamic_closure_threshold is a property; it is not passed to __init__
        baselines[zk_str] = ZoneBaseline(
            zone=zk_str,
            ear_median=zb_dto.ear_median,
            ear_mad_scaled=zb_dto.ear_mad_scaled,
            pitch_median_deg=zb_dto.pitch_median_deg,
            yaw_median_deg=zb_dto.yaw_median_deg,
            sample_count=zb_dto.sample_count,
            ear_left_median=zb_dto.ear_left_median,
            ear_right_median=zb_dto.ear_right_median,
            blendshape_open_median=zb_dto.blendshape_open_median,
            confidence_median=zb_dto.confidence_median,
        )

    # Determine center_ear_mad_scaled from CENTER zone baseline if present
    if "CENTER" in dto.zone_baselines:
        center_mad = dto.zone_baselines["CENTER"].ear_mad_scaled
    else:
        center_mad = 0.020

    # Ensure calibration_mode is not None
    cal_mode = dto.calibration_mode or ("4ZONE" if dto.is_calibrated else "UNCALIBRATED_DEFAULT")

    return DriverBehaviourProfile(
        driver_id=dto.driver_id,
        calibration_mode=cal_mode,
        center_ear_median=dto.center_ear_median,
        center_ear_mad_scaled=center_mad,
        blink_duration_median_ms=dto.blink_duration_median_ms,
        blink_duration_mad_ms=dto.blink_duration_mad_ms,
        blink_rcr_median=dto.blink_rcr_median,
        blink_rcr_mad=dto.blink_rcr_mad,
        blink_freq_median_per_min=getattr(dto, "blink_freq_median_per_min", 16.0) or 16.0,
        blink_freq_mad_per_min=getattr(dto, "blink_freq_mad_per_min", 4.5) or 4.5,
        neutral_pitch_deg=dto.neutral_pitch_deg,
        neutral_yaw_deg=dto.neutral_yaw_deg,
        calibration_reliability=dto.calibration_reliability,
        zone_baselines=baselines,
        active_center_ear_median=dto.center_ear_median,  # Strictly reset to B0 anchor
        baseline_perclos_pct=getattr(dto, "baseline_perclos_pct", 3.5) or 3.5,
        blink_amplitude_median=getattr(dto, "blink_amplitude_median", 0.14) or 0.14,
        temporal_blinks_sampled=int(getattr(dto, "temporal_blinks_sampled", 0) or 0),
        temporal_window_sec=float(getattr(dto, "temporal_window_sec", 120.0) or 120.0),
        sensitivity_level=getattr(dto, "sensitivity_level", "STANDARD") or "STANDARD",
        version=dto.version,
        calibrated_at=dto.calibrated_at,
        neutral_rot_mat=(
            np.asarray(dto.neutral_rot_mat, dtype=np.float64)
            if getattr(dto, "neutral_rot_mat", None) is not None
            else None
        ),
        calibration_is_mirrored=getattr(dto, "calibration_is_mirrored", None),
    )
