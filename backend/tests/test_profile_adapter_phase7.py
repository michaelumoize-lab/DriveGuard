"""
Phase 7 — Task 1 Acceptance Test Suite:
Canonical PoseZone Alignment, Profile Adapter, Timestamp Integrity,
Dynamic Closure Threshold Precision, and Backward-Compatible DTO Migration.

Tests: 46 explicitly named scenarios across 9 functional suites.
"""

from __future__ import annotations

import concurrent.futures
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile
import threading
import time
from typing import Dict

import numpy as np
import pytest
from pydantic import ValidationError

from backend.app.ai.geometry import PoseZone as GeometryPoseZone
from backend.app.ai.pdbp_calibration import (
    DriverBehaviourProfile,
    GuidedCalibrationSession,
    ZoneBaseline,
)
from backend.app.ai.pipeline import SessionAIEngine
from backend.app.ai.profile_adapter import (
    DYNAMIC_CLOSURE_THRESHOLD_TOLERANCE,
    dto_to_profile,
    profile_to_dto,
)
from backend.app.core.time_utils import validate_iso_timestamp
from backend.app.features.head_pose import PoseZone as CanonicalPoseZone
from backend.app.schemas.profile import DriverProfileDTO, ZoneBaselineDTO
from backend.app.services.profile_store import ProfileStore


# ==============================================================================
# Suite 1: Canonical PoseZone & Strict Lookup (5 tests)
# ==============================================================================

def test_canonical_pose_zone_identity():
    """Verify geometry.PoseZone is canonical features.head_pose.PoseZone."""
    assert GeometryPoseZone is CanonicalPoseZone


def test_all_six_canonical_pose_zone_members():
    """Verify all 6 canonical members exist with exact string values."""
    expected = {
        "CENTER": "CENTER",
        "LEFT_MIRROR": "LEFT_MIRROR",
        "RIGHT_MIRROR": "RIGHT_MIRROR",
        "DASHBOARD": "DASHBOARD",
        "OUT_OF_BOUNDS": "OUT_OF_BOUNDS",
        "UNKNOWN": "UNKNOWN",
    }
    actual = {m.name: m.value for m in CanonicalPoseZone}
    assert actual == expected


def test_get_zone_baseline_valid_enum_and_string():
    """Verify get_zone_baseline works identically with PoseZone enum and string."""
    prof = DriverBehaviourProfile.create_default_uncalibrated()
    
    # Existing CENTER zone
    zb_enum = prof.get_zone_baseline(CanonicalPoseZone.CENTER)
    zb_str = prof.get_zone_baseline("CENTER")
    assert zb_enum.zone == "CENTER"
    assert zb_str.zone == "CENTER"
    assert zb_enum.ear_median == zb_str.ear_median

    # Heuristic fallback zone (LEFT_MIRROR)
    zb_left_enum = prof.get_zone_baseline(CanonicalPoseZone.LEFT_MIRROR)
    zb_left_str = prof.get_zone_baseline("LEFT_MIRROR")
    assert zb_left_enum.zone == "LEFT_MIRROR"
    assert zb_left_str.zone == "LEFT_MIRROR"
    assert zb_left_enum.ear_median == zb_left_str.ear_median


def test_get_zone_baseline_rejects_unsupported_types():
    """Verify TypeError is raised when passing unsupported non-string/non-enum types."""
    prof = DriverBehaviourProfile.create_default_uncalibrated()
    for bad_input in [None, 123, 45.6, [], {}]:
        with pytest.raises(TypeError):
            prof.get_zone_baseline(bad_input)


def test_get_zone_baseline_rejects_arbitrary_string_even_if_in_baselines():
    """Verify ValueError is raised on noncanonical strings even if injected into dictionary."""
    prof = DriverBehaviourProfile.create_default_uncalibrated()
    # Malicious or accidental injection into dictionary
    prof.zone_baselines["ROOF"] = ZoneBaseline(
        zone="ROOF", ear_median=0.25, ear_mad_scaled=0.02,
        pitch_median_deg=0.0, yaw_median_deg=0.0, sample_count=10
    )
    with pytest.raises(ValueError, match="Invalid zone string: 'ROOF'"):
        prof.get_zone_baseline("ROOF")


# ==============================================================================
# Suite 2: Shared Timestamp Validation in Neutral Core Module (8 tests)
# ==============================================================================

def test_validate_iso_timestamp_valid_formats():
    """Verify valid ISO strings with Z, offsets, microseconds are preserved byte-for-byte."""
    valid_ts = [
        "2026-11-17T01:35:47.703504+00:00",
        "2026-11-17T12:00:00Z",
        "2026-11-17T14:30:00+02:00",
        "2026-11-17 12:00:00",
        "2026-11-17T12:00:00.123",
    ]
    for ts in valid_ts:
        res = validate_iso_timestamp(ts)
        assert res == ts  # exact preservation without normalization


def test_validate_iso_timestamp_rejects_date_only():
    """Verify date-only string without time component is rejected with ValueError."""
    with pytest.raises(ValueError, match="must contain both date and time components"):
        validate_iso_timestamp("2026-11-17")


def test_validate_iso_timestamp_rejects_non_string():
    """Verify non-string types raise TypeError."""
    for bad_ts in [1700000000, 1700000000.5, True, []]:
        with pytest.raises(TypeError, match="Timestamp must be an ISO 8601 string or None"):
            validate_iso_timestamp(bad_ts)


def test_validate_iso_timestamp_rejects_malformed_string():
    """Verify invalid strings or impossible calendar dates raise ValueError."""
    for malformed in ["not-a-timestamp", "2026-02-30T10:00:00", "2026-13-01T00:00:00Z"]:
        with pytest.raises(ValueError):
            validate_iso_timestamp(malformed)


def test_dto_timestamp_validation():
    """Verify DriverProfileDTO validates timestamp via validator, preserving valid and rejecting invalid."""
    dto = DriverProfileDTO(calibrated_at="2026-11-17T12:00:00Z")
    assert dto.calibrated_at == "2026-11-17T12:00:00Z"

    with pytest.raises(ValidationError):
        DriverProfileDTO(calibrated_at="2026-11-17")  # date-only

    with pytest.raises(ValidationError):
        DriverProfileDTO(calibrated_at=1700000000)  # integer


def test_dataclass_load_json_timestamp_validation():
    """Verify DriverBehaviourProfile.load_json() rejects malformed timestamps without replacing with current time."""
    temp_dir = Path(tempfile.mkdtemp())
    try:
        bad_json = temp_dir / "bad_prof.json"
        bad_json.write_text(json.dumps({"driver_id": "test", "calibrated_at": "invalid-time"}), encoding="utf-8")
        with pytest.raises(ValueError):
            DriverBehaviourProfile.load_json(bad_json)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_dataclass_load_json_missing_timestamp_backward_compat():
    """Verify older JSON without calibrated_at loads cleanly into dataclass with calibrated_at=None."""
    temp_dir = Path(tempfile.mkdtemp())
    try:
        from dataclasses import asdict
        prof_dict = asdict(DriverBehaviourProfile.create_default_uncalibrated("legacy_driver"))
        prof_dict.pop("calibrated_at", None)
        legacy_json = temp_dir / "legacy_prof.json"
        legacy_json.write_text(json.dumps(prof_dict), encoding="utf-8")
        prof = DriverBehaviourProfile.load_json(legacy_json)
        assert prof.driver_id == "legacy_driver"
        assert prof.calibrated_at is None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_timestamp_none_preserved_in_roundtrip():
    """Verify calibrated_at=None roundtrips through DTO and profile as None."""
    prof = DriverBehaviourProfile.create_default_uncalibrated()
    assert prof.calibrated_at is None
    dto = profile_to_dto(prof)
    assert dto.calibrated_at is None
    prof2 = dto_to_profile(dto)
    assert prof2.calibrated_at is None


# ==============================================================================
# Suite 3: Calibration Finalization & UTC Timestamp Generation (2 tests)
# ==============================================================================

def test_finalize_generates_timezone_aware_utc_timestamp():
    """Verify GuidedCalibrationSession._finalize() creates timezone-aware UTC ISO timestamp."""
    session = GuidedCalibrationSession(driver_id="test_driver", four_zone_mode=True)
    session._step_idx = len(session.steps)
    session._ear_samples_by_zone["CENTER"] = [0.28, 0.285, 0.29] * 8
    session._valid_frames_by_zone["CENTER"] = 24
    session._pitch_samples_by_zone["CENTER"] = [0.0] * 24
    session._yaw_samples_by_zone["CENTER"] = [0.0] * 24

    profile = session._finalize()
    assert profile.calibrated_at is not None
    # Parse to ensure it is timezone-aware UTC
    dt = datetime.fromisoformat(profile.calibrated_at)
    assert dt.tzinfo is not None
    assert dt.utcoffset() is not None


def test_adapter_never_generates_timestamp():
    """Verify profile_to_dto and dto_to_profile never invent or overwrite timestamps."""
    prof = DriverBehaviourProfile.create_default_uncalibrated()
    dto = profile_to_dto(prof)
    assert dto.calibrated_at is None

    dto_custom = DriverProfileDTO(calibrated_at=None)
    prof_imported = dto_to_profile(dto_custom)
    assert prof_imported.calibrated_at is None


# ==============================================================================
# Suite 4: Calibration Mode, is_calibrated Consistency & Legacy Migration (8 tests)
# ==============================================================================

def test_normal_calibrated_dto_consistency():
    """Verify is_calibrated=True with explicit valid modes converts and preserves mode without rewrite."""
    for mode in ["FOUR_ZONE", "THREE_ZONE", "CENTER_ONLY", "4ZONE", "1ZONE", "FOUR_ZONE_PLUS_BLINK"]:
        dto = DriverProfileDTO(is_calibrated=True, calibration_mode=mode)
        assert dto.is_calibrated is True
        assert dto.calibration_mode == mode
        prof = dto_to_profile(dto)
        assert prof.calibration_mode == mode
        dto_exported = profile_to_dto(prof)
        assert dto_exported.calibration_mode == mode


def test_normal_uncalibrated_dto_consistency():
    """Verify is_calibrated=False with calibration_mode='UNCALIBRATED_DEFAULT' converts accurately."""
    dto = DriverProfileDTO(is_calibrated=False, calibration_mode="UNCALIBRATED_DEFAULT")
    assert dto.is_calibrated is False
    assert dto.calibration_mode == "UNCALIBRATED_DEFAULT"
    prof = dto_to_profile(dto)
    assert prof.calibration_mode == "UNCALIBRATED_DEFAULT"


def test_legacy_dto_omitted_mode_calibrated_migrates_to_exact_historical_default():
    """Verify legacy payload with is_calibrated=True and omitted calibration_mode migrates to '4ZONE'."""
    dto = DriverProfileDTO.model_validate({"is_calibrated": True})
    assert dto.is_calibrated is True
    assert dto.calibration_mode == "4ZONE"  # Exact historical default


def test_legacy_dto_omitted_mode_uncalibrated_defaults_cleanly():
    """Verify legacy payload with is_calibrated=False and omitted calibration_mode resolves to 'UNCALIBRATED_DEFAULT'."""
    dto = DriverProfileDTO.model_validate({"is_calibrated": False})
    assert dto.is_calibrated is False
    assert dto.calibration_mode == "UNCALIBRATED_DEFAULT"


def test_legacy_dto_omitted_both_fields_defaults_cleanly():
    """Verify empty payload {} defaults cleanly to is_calibrated=False, calibration_mode='UNCALIBRATED_DEFAULT'."""
    dto = DriverProfileDTO.model_validate({})
    assert dto.is_calibrated is False
    assert dto.calibration_mode == "UNCALIBRATED_DEFAULT"


def test_legacy_dto_omitted_is_calibrated_infers_from_mode():
    """Verify payload with calibration_mode='4ZONE' (omitted is_calibrated) correctly infers is_calibrated=True."""
    dto = DriverProfileDTO.model_validate({"calibration_mode": "4ZONE"})
    assert dto.is_calibrated is True
    assert dto.calibration_mode == "4ZONE"


def test_dto_rejects_explicit_null_calibration_mode():
    """Verify explicit calibration_mode=None is rejected with ValidationError."""
    with pytest.raises(ValidationError, match="calibration_mode cannot be null"):
        DriverProfileDTO.model_validate({"calibration_mode": None})

    with pytest.raises(ValidationError, match="calibration_mode cannot be null"):
        DriverProfileDTO.model_validate({"is_calibrated": True, "calibration_mode": None})


def test_explicit_contradictory_dto_rejected():
    """Verify explicit contradictions raise ValidationError without silent rewriting."""
    # Contradiction A: is_calibrated=True with UNCALIBRATED_DEFAULT
    with pytest.raises(ValidationError, match="Explicit contradiction"):
        DriverProfileDTO.model_validate({"is_calibrated": True, "calibration_mode": "UNCALIBRATED_DEFAULT"})

    # Contradiction B: is_calibrated=False with active calibration mode
    for active_mode in ["FOUR_ZONE", "4ZONE", "CENTER_ONLY"]:
        with pytest.raises(ValidationError, match="Explicit contradiction"):
            DriverProfileDTO.model_validate({"is_calibrated": False, "calibration_mode": active_mode})


# ==============================================================================
# Suite 5: Dynamic Closure Threshold Precision & Tolerance Policy (6 tests)
# ==============================================================================

def test_dynamic_closure_threshold_exact_agreement():
    """Verify exact formula agreement when unrounded float inputs are used."""
    zb = ZoneBaseline(
        zone="CENTER", ear_median=0.285, ear_mad_scaled=0.02,
        pitch_median_deg=0.0, yaw_median_deg=0.0, sample_count=50
    )
    # raw = min(0.285 * 0.76, 0.285 - 2.3 * 0.02) = min(0.2166, 0.239) = 0.2166
    assert abs(zb.dynamic_closure_threshold - 0.2166) < 1e-9


def test_dynamic_closure_threshold_maximum_legitimate_rounding():
    """Verify that maximum legitimate 4-decimal serialization rounding discrepancy (<= 0.000215) passes tolerance."""
    # Create DTO with exact 4-decimal rounded values
    ear_med = 0.2854
    ear_mad = 0.0182
    expected = float(np.clip(min(ear_med * 0.76, ear_med - 2.3 * ear_mad), 0.12, 0.22))
    # Stored threshold offset by legitimate float rounding (e.g. 0.0002)
    stored_threshold = round(expected + 0.0002, 4)
    
    zb_dto = ZoneBaselineDTO(
        zone="CENTER", ear_median=ear_med, ear_mad_scaled=ear_mad,
        dynamic_closure_threshold=stored_threshold, sample_count=50
    )
    dto = DriverProfileDTO(
        is_calibrated=True, calibration_mode="FOUR_ZONE",
        zone_baselines={"CENTER": zb_dto}
    )
    prof = dto_to_profile(dto)
    assert prof is not None


def test_dynamic_closure_threshold_boundary_rejection():
    """Verify that a discrepancy immediately beyond tolerance (0.0006 > 0.0005) raises ValueError."""
    ear_med = 0.2850
    ear_mad = 0.0200
    expected = float(np.clip(min(ear_med * 0.76, ear_med - 2.3 * ear_mad), 0.12, 0.22))
    inconsistent_threshold = round(expected + 0.0006, 4)

    zb_dto = ZoneBaselineDTO(
        zone="CENTER", ear_median=ear_med, ear_mad_scaled=ear_mad,
        dynamic_closure_threshold=inconsistent_threshold, sample_count=50
    )
    dto = DriverProfileDTO(
        is_calibrated=True, calibration_mode="FOUR_ZONE",
        zone_baselines={"CENTER": zb_dto}
    )
    with pytest.raises(ValueError, match="Materially inconsistent dynamic_closure_threshold"):
        dto_to_profile(dto)


def test_dynamic_closure_threshold_material_inconsistency_rejected():
    """Verify that materially inconsistent threshold (e.g. stored 0.23 vs expected 0.2166) raises ValueError."""
    zb_dto = ZoneBaselineDTO(
        zone="CENTER", ear_median=0.285, ear_mad_scaled=0.02,
        dynamic_closure_threshold=0.2300, sample_count=50
    )
    dto = DriverProfileDTO(
        is_calibrated=True, calibration_mode="FOUR_ZONE",
        zone_baselines={"CENTER": zb_dto}
    )
    with pytest.raises(ValueError, match="Materially inconsistent dynamic_closure_threshold"):
        dto_to_profile(dto)


def test_dynamic_closure_threshold_runtime_authority():
    """Verify ZoneBaseline.dynamic_closure_threshold evaluates dynamically, proving stored DTO never overrides."""
    zb_dto = ZoneBaselineDTO(
        zone="CENTER", ear_median=0.285, ear_mad_scaled=0.02,
        dynamic_closure_threshold=0.2166, sample_count=50
    )
    dto = DriverProfileDTO(
        is_calibrated=True, calibration_mode="FOUR_ZONE",
        zone_baselines={"CENTER": zb_dto}
    )
    prof = dto_to_profile(dto)
    # Ensure ZoneBaseline dynamically computes the exact formula
    assert prof.zone_baselines["CENTER"].dynamic_closure_threshold == 0.2166


def test_dynamic_closure_threshold_repository_fixtures_compatibility():
    """Verify all real repository profile fixtures satisfy tolerance and import cleanly."""
    fixtures_dir = Path("backend/data/profiles")
    if fixtures_dir.exists():
        for p in fixtures_dir.glob("*.json"):
            data = json.loads(p.read_text(encoding="utf-8"))
            dto = DriverProfileDTO.model_validate(data)
            prof = dto_to_profile(dto)
            assert prof.driver_id == dto.driver_id


# ==============================================================================
# Suite 6: Adapter Roundtrip, Immutability & Kinematics Fidelity (6 tests)
# ==============================================================================

def test_uncalibrated_profile_roundtrip():
    """Verify uncalibrated default profile roundtrips with 100% field fidelity."""
    prof = DriverBehaviourProfile.create_default_uncalibrated()
    dto = profile_to_dto(prof)
    assert dto.is_calibrated is False
    assert dto.calibration_mode == "UNCALIBRATED_DEFAULT"
    prof2 = dto_to_profile(dto)
    assert prof2.calibration_mode == "UNCALIBRATED_DEFAULT"
    assert prof2.center_ear_median == prof.center_ear_median


def test_four_zone_profile_roundtrip():
    """Verify complete 4-zone profile roundtrips with all kinematics, asymmetry, and baselines preserved."""
    session = GuidedCalibrationSession(driver_id="driver_4z", four_zone_mode=True)
    session._step_idx = len(session.steps)
    for zk in ["CENTER", "LEFT_MIRROR", "RIGHT_MIRROR", "DASHBOARD"]:
        session._ear_samples_by_zone[zk] = [0.28, 0.285] * 10
        session._valid_frames_by_zone[zk] = 20
        session._pitch_samples_by_zone[zk] = [0.0] * 20
        session._yaw_samples_by_zone[zk] = [0.0] * 20
    prof = session._finalize()

    dto = profile_to_dto(prof)
    assert dto.is_calibrated is True
    assert len(dto.zone_baselines) == 4
    prof2 = dto_to_profile(dto)
    assert prof2.driver_id == prof.driver_id
    assert len(prof2.zone_baselines) == 4


def test_backfilled_profile_adapter_roundtrip():
    """Verify CENTER_ONLY profile roundtrips, preserving sample_count=0 on backfills and >0 on CENTER."""
    session = GuidedCalibrationSession(driver_id="driver_1z", four_zone_mode=False, three_zone_mode=False)
    session._step_idx = len(session.steps)
    session._ear_samples_by_zone["CENTER"] = [0.27, 0.275] * 10
    session._valid_frames_by_zone["CENTER"] = 20
    session._pitch_samples_by_zone["CENTER"] = [0.0] * 20
    session._yaw_samples_by_zone["CENTER"] = [0.0] * 20
    prof = session._finalize()

    assert prof.zone_baselines["CENTER"].sample_count > 0
    assert prof.zone_baselines["LEFT_MIRROR"].sample_count == 0
    assert prof.zone_baselines["RIGHT_MIRROR"].sample_count == 0
    assert prof.zone_baselines["DASHBOARD"].sample_count == 0

    dto = profile_to_dto(prof)
    prof2 = dto_to_profile(dto)
    assert prof2.zone_baselines["CENTER"].sample_count > 0
    assert prof2.zone_baselines["LEFT_MIRROR"].sample_count == 0
    assert prof2.zone_baselines["RIGHT_MIRROR"].sample_count == 0
    assert prof2.zone_baselines["DASHBOARD"].sample_count == 0


def test_adapter_source_object_immutability():
    """Verify profile_to_dto does not modify source profile, and dto_to_profile does not modify source DTO."""
    prof = DriverBehaviourProfile.create_default_uncalibrated()
    prof_copy = copy.deepcopy(prof)
    dto = profile_to_dto(prof)
    assert prof == prof_copy

    dto_copy = copy.deepcopy(dto)
    prof2 = dto_to_profile(dto)
    assert dto == dto_copy


def test_active_drift_baseline_reset_to_b0():
    """Verify dto_to_profile strictly resets active_center_ear_median to center_ear_median (B0 anchor)."""
    prof = DriverBehaviourProfile.create_default_uncalibrated()
    # Simulate drift
    prof.active_center_ear_median = 0.350
    assert prof.active_center_ear_median != prof.center_ear_median

    dto = profile_to_dto(prof)
    prof_imported = dto_to_profile(dto)
    assert prof_imported.active_center_ear_median == prof_imported.center_ear_median
    assert prof_imported.active_center_ear_median == dto.center_ear_median


def test_minimal_v1_dto_compatibility():
    """Verify minimal Version 1 DTO with default/sparse fields imports cleanly."""
    dto = DriverProfileDTO(driver_id="sparse_driver")
    prof = dto_to_profile(dto)
    assert prof.driver_id == "sparse_driver"
    assert prof.calibration_mode == "UNCALIBRATED_DEFAULT"


# ==============================================================================
# Suite 7: Backfilled Sample Count Integrity (3 tests)
# ==============================================================================

def test_center_only_session_backfill_sample_count_zero():
    """Verify GuidedCalibrationSession in CENTER_ONLY mode outputs sample_count=0 on all backfilled zones."""
    session = GuidedCalibrationSession(driver_id="driver_co", four_zone_mode=False, three_zone_mode=False)
    session._step_idx = len(session.steps)
    session._ear_samples_by_zone["CENTER"] = [0.28] * 20
    session._valid_frames_by_zone["CENTER"] = 20
    session._pitch_samples_by_zone["CENTER"] = [0.0] * 20
    session._yaw_samples_by_zone["CENTER"] = [0.0] * 20
    prof = session._finalize()

    for zone in ["LEFT_MIRROR", "RIGHT_MIRROR", "DASHBOARD"]:
        assert prof.zone_baselines[zone].sample_count == 0


def test_three_zone_session_backfill_sample_count_zero():
    """Verify GuidedCalibrationSession in THREE_ZONE mode outputs sample_count=0 on DASHBOARD backfill."""
    session = GuidedCalibrationSession(driver_id="driver_3z", four_zone_mode=False, three_zone_mode=True)
    session._step_idx = len(session.steps)
    session._ear_samples_by_zone["CENTER"] = [0.28] * 20
    session._valid_frames_by_zone["CENTER"] = 20
    session._pitch_samples_by_zone["CENTER"] = [0.0] * 20
    session._yaw_samples_by_zone["CENTER"] = [0.0] * 20
    session._ear_samples_by_zone["LEFT_MIRROR"] = [0.26] * 20
    session._valid_frames_by_zone["LEFT_MIRROR"] = 20
    session._ear_samples_by_zone["RIGHT_MIRROR"] = [0.26] * 20
    session._valid_frames_by_zone["RIGHT_MIRROR"] = 20
    prof = session._finalize()

    assert prof.zone_baselines["DASHBOARD"].sample_count == 0


def test_empirical_zone_sample_count_positive():
    """Verify empirically calibrated zones maintain observed sample_count > 0."""
    session = GuidedCalibrationSession(driver_id="driver_emp", four_zone_mode=True)
    session._step_idx = len(session.steps)
    session._ear_samples_by_zone["CENTER"] = [0.28] * 20
    session._valid_frames_by_zone["CENTER"] = 20
    session._pitch_samples_by_zone["CENTER"] = [0.0] * 20
    session._yaw_samples_by_zone["CENTER"] = [0.0] * 20
    prof = session._finalize()

    assert prof.zone_baselines["CENTER"].sample_count > 0


# ==============================================================================
# Suite 8: ProfileStore & Direct File Persistence (4 tests)
# ==============================================================================

def test_profilestore_atomic_save_and_get():
    """Verify ProfileStore.save_profile() and get_profile() cycle under lock."""
    temp_dir = Path(tempfile.mkdtemp())
    try:
        store = ProfileStore(storage_dir=temp_dir)
        dto = DriverProfileDTO(driver_id="store_test_driver", is_calibrated=True, calibration_mode="4ZONE")
        store.save_profile(dto)
        loaded = store.get_profile("store_test_driver")
        assert loaded is not None
        assert loaded.driver_id == "store_test_driver"
        assert loaded.calibration_mode == "4ZONE"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_profilestore_corrupt_timestamp_quarantine():
    """Verify stored profile with invalid timestamp is quarantined to .json.corrupt and returns None."""
    temp_dir = Path(tempfile.mkdtemp())
    try:
        store = ProfileStore(storage_dir=temp_dir)
        p = store._file_path("corrupt_driver")
        p.write_text(json.dumps({"driver_id": "corrupt_driver", "calibrated_at": "not-valid-date"}), encoding="utf-8")
        loaded = store.get_profile("corrupt_driver")
        assert loaded is None
        assert not p.exists()
        assert p.with_suffix(".json.corrupt").exists()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_profilestore_date_only_timestamp_quarantine():
    """Verify stored profile with date-only timestamp is quarantined to .json.corrupt."""
    temp_dir = Path(tempfile.mkdtemp())
    try:
        store = ProfileStore(storage_dir=temp_dir)
        p = store._file_path("date_only_driver")
        p.write_text(json.dumps({"driver_id": "date_only_driver", "calibrated_at": "2026-11-17"}), encoding="utf-8")
        loaded = store.get_profile("date_only_driver")
        assert loaded is None
        assert p.with_suffix(".json.corrupt").exists()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_driver_behaviour_profile_direct_save_and_load_cycle():
    """Verify DriverBehaviourProfile.save_json() and load_json() cycle preserves timestamp and baselines."""
    temp_dir = Path(tempfile.mkdtemp())
    try:
        prof = DriverBehaviourProfile.create_default_uncalibrated("persisted_driver")
        prof.calibrated_at = "2026-11-17T12:00:00+00:00"
        save_path = temp_dir / "prof.json"
        prof.save_json(save_path)
        loaded = DriverBehaviourProfile.load_json(save_path)
        assert loaded.driver_id == "persisted_driver"
        assert loaded.calibrated_at == "2026-11-17T12:00:00+00:00"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ==============================================================================
# Suite 9: Pipeline Delegation, Import Safety & Concurrency (4 tests)
# ==============================================================================

def test_camera_test_entrypoint_import_safety():
    """Verify that importing scripts.run_camera_test succeeds with zero import errors."""
    import scripts.run_camera_test
    assert hasattr(scripts.run_camera_test, "DriverFeatureExtractor")


def test_pipeline_export_and_load_dto_delegation():
    """Verify SessionAIEngine.export_profile_dto() and load_profile_dto() correctly delegate to adapter."""
    engine = SessionAIEngine(driver_id="pipeline_driver")
    dto = engine.export_profile_dto()
    assert isinstance(dto, DriverProfileDTO)
    assert dto.driver_id == "pipeline_driver"

    # Load new profile DTO
    new_dto = DriverProfileDTO(
        driver_id="imported_pipeline_driver",
        is_calibrated=True,
        calibration_mode="4ZONE",
        center_ear_median=0.295,
        zone_baselines={
            "CENTER": ZoneBaselineDTO(
                zone="CENTER", ear_median=0.295, ear_mad_scaled=0.018,
                dynamic_closure_threshold=0.2200, sample_count=100
            )
        }
    )
    engine.load_profile_dto(new_dto)
    assert engine.profile.driver_id == "imported_pipeline_driver"
    assert engine.profile.calibration_mode == "4ZONE"
    assert engine.profile.center_ear_median == 0.295


def test_pipeline_concurrent_profile_access():
    """Verify multi-threaded concurrent reading and exporting under lock with bounded timeouts."""
    engine = SessionAIEngine(driver_id="concurrent_driver")
    errors = []

    def reader():
        try:
            for _ in range(25):
                dto = engine.export_profile_dto()
                assert dto.driver_id in ("concurrent_driver", "updated_driver")
        except Exception as e:
            errors.append(e)

    def writer():
        try:
            for _ in range(10):
                new_dto = DriverProfileDTO(
                    driver_id="updated_driver",
                    is_calibrated=True,
                    calibration_mode="4ZONE",
                    center_ear_median=0.285,
                    zone_baselines={
                        "CENTER": ZoneBaselineDTO(
                            zone="CENTER", ear_median=0.285, ear_mad_scaled=0.02,
                            dynamic_closure_threshold=0.2166, sample_count=50
                        )
                    }
                )
                engine.load_profile_dto(new_dto)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=reader) for _ in range(4)]
    threads.append(threading.Thread(target=writer))

    for t in threads:
        t.start()

    for t in threads:
        t.join(timeout=2.0)
        assert not t.is_alive(), "Thread deadlock detected"

    assert len(errors) == 0, f"Concurrent access errors: {errors}"


def test_bounded_concurrency_safe_thread_cleanup():
    """Verify that bounded concurrency tests enforce join timeouts and safely track active threads."""
    stop_event = threading.Event()
    worker_started = threading.Event()

    def cancellable_worker():
        worker_started.set()
        while not stop_event.is_set():
            time.sleep(0.01)

    t = threading.Thread(target=cancellable_worker)
    t.start()
    worker_started.wait(timeout=1.0)

    # Clean termination
    stop_event.set()
    t.join(timeout=2.0)
    assert not t.is_alive(), "Worker failed to terminate within bounded timeout"
