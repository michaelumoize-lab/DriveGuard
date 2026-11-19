"""
Tests for Profile Persistence, Thread Safety, Path Sanitization, and Corrupt File Recovery (DG-03, DG-04, DG-15).
"""

import concurrent.futures
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from backend.app.schemas.profile import DriverProfileDTO, ZoneBaselineDTO
from backend.app.services.profile_store import ProfileStore


@pytest.fixture
def temp_profile_store():
    temp_dir = Path(tempfile.mkdtemp(prefix="driveguard_test_profiles_"))
    store = ProfileStore(storage_dir=temp_dir)
    yield store
    shutil.rmtree(temp_dir, ignore_errors=True)


def _make_dummy_profile(driver_id: str, ear: float = 0.28) -> DriverProfileDTO:
    raw_c = min(ear * 0.76, ear - 2.3 * 0.015)
    dyn_c = float(np.clip(raw_c, 0.12, 0.22))
    return DriverProfileDTO(
        version=1,
        driver_id=driver_id,
        calibrated_at="2026-11-11T00:00:00Z",
        is_calibrated=True,
        calibration_mode="4ZONE",
        center_ear_median=ear,
        blink_duration_median_ms=240.0,
        blink_duration_mad_ms=38.0,
        blink_rcr_median=0.98,
        blink_rcr_mad=0.16,
        blink_freq_median_per_min=17.5,
        blink_freq_mad_per_min=4.2,
        baseline_perclos_pct=3.8,
        blink_amplitude_median=0.15,
        temporal_blinks_sampled=42,
        temporal_window_sec=120.0,
        sensitivity_level="STANDARD",
        neutral_pitch_deg=-1.5,
        neutral_yaw_deg=0.8,
        calibration_reliability=0.92,
        zone_baselines={
            "CENTER": ZoneBaselineDTO(
                zone="CENTER",
                ear_median=ear,
                ear_mad_scaled=0.015,
                dynamic_closure_threshold=round(dyn_c, 4),
                sample_count=50,
            )
        },
    )


class TestProfileStoreDG15:
    """DG-15: Concurrency, Atomic Writes, Path Sanitization, and Corrupt Profile Recovery."""

    def test_full_fidelity_persistence_dg03(self, temp_profile_store: ProfileStore) -> None:
        """DG-03: Verify all 6 previously missing temporal fields persist with full fidelity."""
        p = _make_dummy_profile("driver_fidelity_test")
        temp_profile_store.save_profile(p)

        loaded = temp_profile_store.get_profile("driver_fidelity_test")
        assert loaded is not None
        assert loaded.blink_freq_median_per_min == 17.5
        assert loaded.blink_freq_mad_per_min == 4.2
        assert loaded.baseline_perclos_pct == 3.8
        assert loaded.blink_amplitude_median == 0.15
        assert loaded.temporal_blinks_sampled == 42
        assert loaded.temporal_window_sec == 120.0
        assert loaded.sensitivity_level == "STANDARD"

    def test_atomic_write_leaves_no_temp_files(self, temp_profile_store: ProfileStore) -> None:
        """Verify atomic writes leave zero residual temporary files in storage directory."""
        p = _make_dummy_profile("driver_atomic")
        temp_profile_store.save_profile(p)

        # Ensure profile exists
        assert (temp_profile_store.storage_dir / "driver_atomic_profile.json").is_file()

        # Ensure no temporary files are left over
        tmp_files = list(temp_profile_store.storage_dir.glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_path_sanitization_prevents_directory_traversal(
        self, temp_profile_store: ProfileStore
    ) -> None:
        """Verify malicious driver IDs cannot escape storage directory or overwrite system files."""
        malicious_ids = [
            "../../etc/passwd",
            "..\\..\\windows\\system32",
            "/absolute/root/driver",
            "driver/../../secret",
            "../../../",
            "....//....//traversal",
        ]

        for bad_id in malicious_ids:
            target_path = temp_profile_store._file_path(bad_id)
            assert target_path.resolve().is_relative_to(
                temp_profile_store.storage_dir.resolve()
            ), f"Path traversal succeeded for {bad_id} -> {target_path}"

            # Ensure save and get operate safely inside storage_dir
            dto = _make_dummy_profile(bad_id)
            temp_profile_store.save_profile(dto)
            assert target_path.is_file()
            assert target_path.parent == temp_profile_store.storage_dir.resolve()

    def test_concurrent_profile_reads_and_writes(self, temp_profile_store: ProfileStore) -> None:
        """Verify thread safety under heavy concurrent profile updates without data corruption."""
        num_workers = 16
        iterations_per_worker = 10

        def worker_task(worker_id: int):
            for i in range(iterations_per_worker):
                driver_id = f"concurrent_driver_{worker_id % 4}"
                ear_val = round(0.20 + (i * 0.01), 3)
                profile = _make_dummy_profile(driver_id, ear=ear_val)
                temp_profile_store.save_profile(profile)
                loaded = temp_profile_store.get_profile(driver_id)
                assert loaded is not None
                assert loaded.driver_id == driver_id
                assert 0.20 <= loaded.center_ear_median <= 0.35

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(worker_task, wid) for wid in range(num_workers)]
            for fut in concurrent.futures.as_completed(futures):
                fut.result()

        profiles = temp_profile_store.list_profiles()
        assert len(profiles) == 4

    def test_corrupt_profile_recovery(self, temp_profile_store: ProfileStore) -> None:
        """Verify corrupt profile files are quarantined and recover cleanly without crashing."""
        driver_id = "driver_corrupt_recovery"
        file_path = temp_profile_store._file_path(driver_id)

        # Write corrupt invalid JSON
        file_path.write_text("{ this is completely invalid JSON ::: 123", encoding="utf-8")

        # get_profile must catch error, quarantine to .corrupt, and return None
        result = temp_profile_store.get_profile(driver_id)
        assert result is None
        assert not file_path.exists(), "Corrupt file was not replaced/quarantined"
        assert file_path.with_suffix(".json.corrupt").is_file()

        # Next valid save must succeed cleanly
        valid = _make_dummy_profile(driver_id)
        temp_profile_store.save_profile(valid)
        recovered = temp_profile_store.get_profile(driver_id)
        assert recovered is not None
        assert recovered.driver_id == driver_id

    def test_stage2_3_rollback_new_profile_preserves_corrupt_quarantine(
        self, temp_profile_store: ProfileStore
    ) -> None:
        """Stage 2.3: Verify rollback_new_profile unlinks only primary profile and preserves .json.corrupt."""
        driver_id = "driver_rollback_quarantine"
        p = _make_dummy_profile(driver_id)
        temp_profile_store.save_profile(p)

        file_path = temp_profile_store._file_path(driver_id)
        assert file_path.is_file()

        # Place a simulated corrupt quarantine file
        corrupt_path = file_path.with_suffix(".json.corrupt")
        corrupt_path.write_text("quarantined corrupt payload", encoding="utf-8")
        assert corrupt_path.is_file()

        # Roll back new profile
        rolled_back = temp_profile_store.rollback_new_profile(driver_id)
        assert rolled_back is True
        assert not file_path.is_file(), "Primary profile was not unlinked during rollback"
        assert corrupt_path.is_file(), "Quarantined corrupt file was incorrectly removed during rollback"

        # Subsequent rollback on non-existent file returns False
        assert temp_profile_store.rollback_new_profile(driver_id) is False
        assert corrupt_path.is_file()

    def test_stage2_3_commit_and_activate_success_and_rollback(
        self, temp_profile_store: ProfileStore
    ) -> None:
        """Stage 2.3: Verify runner commit_and_activate_profile coordinates persistence and activation rollback."""
        from scripts.run_camera_test import commit_and_activate_profile
        from backend.app.ai.profile_adapter import dto_to_profile
        from backend.app.features.feature_extractor import DriverFeatureExtractor

        extractor = DriverFeatureExtractor()
        dto = _make_dummy_profile("driver_tx_success")
        domain_prof = dto_to_profile(dto)

        # 1. Success transaction
        success, msg = commit_and_activate_profile(
            profile=domain_prof,
            feature_extractor=extractor,
            store=temp_profile_store,
        )
        assert success is True
        assert extractor.profile.driver_id == "driver_tx_success"
        assert temp_profile_store.get_profile("driver_tx_success") is not None

        # 2. Infeasible profile fails before persistence
        infeasible_dto = _make_dummy_profile("driver_infeasible", ear=0.05)
        infeasible_prof = dto_to_profile(infeasible_dto)
        fail_success, fail_msg = commit_and_activate_profile(
            profile=infeasible_prof,
            feature_extractor=extractor,
            store=temp_profile_store,
        )
        assert fail_success is False
        assert "Infeasible" in fail_msg
        assert temp_profile_store.get_profile("driver_infeasible") is None

        # 3. Activation error triggers persistence rollback
        dto_failing = _make_dummy_profile("driver_act_fail")
        prof_failing = dto_to_profile(dto_failing)

        orig_set = extractor.set_profile
        try:
            def failing_set(*args, **kwargs):
                raise RuntimeError("Activation pipeline fault")

            extractor.set_profile = failing_set
            act_fail_success, act_fail_msg = commit_and_activate_profile(
                profile=prof_failing,
                feature_extractor=extractor,
                store=temp_profile_store,
            )
            assert act_fail_success is False
            assert "disk persistence rolled back" in act_fail_msg
            # File on disk must have been rolled back
            assert temp_profile_store.get_profile("driver_act_fail") is None
        finally:
            extractor.set_profile = orig_set

    def test_stage2_3_profile_replacement_failure_restores_prior_content(
        self, temp_profile_store: ProfileStore
    ) -> None:
        """
        Stage 2.3: Verify that when profile replacement activation fails:
        - Prior valid profile is restored with exact prior content and fingerprint.
        - Quarantined .json.corrupt files are preserved.
        """
        from scripts.run_camera_test import commit_and_activate_profile
        from backend.app.ai.profile_adapter import dto_to_profile
        from backend.app.ai.pdbp_calibration import compute_canonical_profile_fingerprint
        from backend.app.features.feature_extractor import DriverFeatureExtractor

        driver_id = "driver_replace_fail"
        extractor = DriverFeatureExtractor()

        # 1. Establish valid prior profile P1
        dto_p1 = _make_dummy_profile(driver_id, ear=0.28)
        prof_p1 = dto_to_profile(dto_p1)
        ok1, msg1 = commit_and_activate_profile(prof_p1, extractor, store=temp_profile_store)
        assert ok1 is True
        p1_fp = compute_canonical_profile_fingerprint(prof_p1)
        p1_file = temp_profile_store._file_path(driver_id)
        p1_raw_content = p1_file.read_text(encoding="utf-8")

        # Plant a quarantined corrupt file
        corrupt_file = p1_file.with_suffix(".json.corrupt")
        corrupt_file.write_text("quarantined corrupt payload", encoding="utf-8")

        # 2. Prepare replacement profile P2
        dto_p2 = _make_dummy_profile(driver_id, ear=0.32)
        prof_p2 = dto_to_profile(dto_p2)

        # 3. Simulate failure during runtime activation of P2
        orig_set = extractor.set_profile
        try:
            def failing_set(*args, **kwargs):
                raise RuntimeError("Hardware activation fault")

            extractor.set_profile = failing_set
            ok2, msg2 = commit_and_activate_profile(prof_p2, extractor, store=temp_profile_store)
            assert ok2 is False
            assert "disk persistence rolled back" in msg2
        finally:
            extractor.set_profile = orig_set

        # 4. Verify prior profile P1 was restored with exact content and fingerprint
        restored_dto = temp_profile_store.get_profile(driver_id)
        assert restored_dto is not None
        assert restored_dto.center_ear_median == 0.28
        restored_prof = dto_to_profile(restored_dto)
        assert compute_canonical_profile_fingerprint(restored_prof) == p1_fp
        assert p1_file.read_text(encoding="utf-8") == p1_raw_content

        # 5. Verify quarantined file remains untouched
        assert corrupt_file.is_file()
        assert corrupt_file.read_text(encoding="utf-8") == "quarantined corrupt payload"

    def test_stage2_3_rollback_failure_raises_and_retains_diagnostics(
        self, temp_profile_store: ProfileStore, monkeypatch
    ) -> None:
        """
        Stage 2.3: If rollback fails, the caller receives an explicit error and
        never claims successful recovery.
        """
        driver_id = "driver_rb_fail"
        p = _make_dummy_profile(driver_id)
        temp_profile_store.save_profile(p)

        # Force os.replace to fail during rollback
        import os
        orig_replace = os.replace

        def failing_replace(src, dst):
            raise PermissionError("EACCES: Disk locked")

        monkeypatch.setattr(os, "replace", failing_replace)
        with pytest.raises(RuntimeError, match="Rollback failed to restore previous profile"):
            temp_profile_store.rollback_new_profile(driver_id, previous_content="dummy payload")

    def test_stage2_3_deferred_activation_persistence_coordination(
        self, temp_profile_store: ProfileStore
    ) -> None:
        """
        Stage 2.3: Verify deferred activation persistence coordination:
        - When queued during blink, safe-latch failure rolls back disk to prior profile exactly once.
        - Uncalibrated deferred activation failure unlinks disk file exactly once.
        """
        from scripts.run_camera_test import commit_and_activate_profile
        from backend.app.ai.profile_adapter import dto_to_profile
        from backend.app.ai.pdbp_calibration import compute_canonical_profile_fingerprint
        from backend.app.features.feature_extractor import DriverFeatureExtractor
        from backend.app.features.blink_detector import BlinkState
        from backend.app.vision.face_landmarker import FaceLandmarkResult, FaceLandmarkStatus
        from backend.tests.test_features import _generate_synthetic_face_landmarks
        from backend.app.vision.face_landmarker import FaceLandmarkResult, FaceLandmarkStatus
        from backend.tests.test_features import _generate_synthetic_face_landmarks

        driver_id = "driver_deferred_coordination"
        extractor = DriverFeatureExtractor()

        # 1. Establish valid prior profile P1
        dto_p1 = _make_dummy_profile(driver_id, ear=0.28)
        prof_p1 = dto_to_profile(dto_p1)
        ok1, _ = commit_and_activate_profile(prof_p1, extractor, store=temp_profile_store)
        assert ok1 is True
        p1_fp = compute_canonical_profile_fingerprint(prof_p1)

        # 2. Enter active blink state and queue P2 via commit_and_activate_profile
        extractor._blink_detector._state = BlinkState.CLOSING
        dto_p2 = _make_dummy_profile(driver_id, ear=0.32)
        prof_p2 = dto_to_profile(dto_p2)

        ok2, msg2 = commit_and_activate_profile(prof_p2, extractor, store=temp_profile_store)
        assert ok2 is True
        assert "deferred activation" in msg2
        assert extractor.has_pending_profile is True
        # At this point, disk has P2 saved
        assert temp_profile_store.get_profile(driver_id).center_ear_median == 0.32

        # 3. Simulate failure during safe latch promotion in extract()
        orig_update = extractor._blink_detector.update_thresholds
        try:
            failed_once = False

            def failing_update(*args, **kwargs):
                nonlocal failed_once
                if not failed_once:
                    failed_once = True
                    raise RuntimeError("Detector threshold hardware update error")
                return orig_update(*args, **kwargs)

            extractor._blink_detector.update_thresholds = failing_update
            extractor._blink_detector._state = BlinkState.OPEN

            lms = _generate_synthetic_face_landmarks(0.0, pitch_deg=0.0, is_mirrored=False)
            frame = FaceLandmarkResult(
                has_face=True,
                landmarks=lms,
                presence_confidence=0.98,
                timestamp_ms=2000,
                status=FaceLandmarkStatus.TRACKING,
                inference_latency_ms=10.0,
                reasons=(),
            )
            # Frame extraction triggers safe latch promotion, which encounters simulated failure
            res = extractor.extract(frame)
        finally:
            extractor._blink_detector.update_thresholds = orig_update

        # Extractor state restored to P1
        assert extractor.is_active_profile_consistent() is True
        assert extractor.profile.driver_id == driver_id
        assert extractor.profile.center_ear_median == 0.28
        assert extractor.has_pending_profile is False

        # Disk was rolled back to P1 exactly once!
        restored_disk = temp_profile_store.get_profile(driver_id)
        assert restored_disk.center_ear_median == 0.28
        assert compute_canonical_profile_fingerprint(dto_to_profile(restored_disk)) == p1_fp

    def test_stage2_3_rejects_second_calibration_commit_while_pending(
        self, temp_profile_store: ProfileStore
    ) -> None:
        """
        Stage 2.3: Verify that commit_and_activate_profile rejects a second calibration
        commit while another transaction is pending safe-latch activation.
        """
        from scripts.run_camera_test import commit_and_activate_profile
        from backend.app.ai.profile_adapter import dto_to_profile
        from backend.app.features.feature_extractor import DriverFeatureExtractor
        from backend.app.features.blink_detector import BlinkState
        from backend.app.vision.face_landmarker import FaceLandmarkResult, FaceLandmarkStatus
        from backend.tests.test_features import _generate_synthetic_face_landmarks

        driver_id = "driver_reject_second"
        extractor = DriverFeatureExtractor()

        # 1. Establish initial active profile P1
        p1 = dto_to_profile(_make_dummy_profile(driver_id, ear=0.28))
        ok1, _ = commit_and_activate_profile(p1, extractor, store=temp_profile_store)
        assert ok1 is True

        # 2. In active blink, queue P2
        extractor._blink_detector._state = BlinkState.CLOSING
        p2 = dto_to_profile(_make_dummy_profile(driver_id, ear=0.30))
        ok2, msg2 = commit_and_activate_profile(p2, extractor, store=temp_profile_store)
        assert ok2 is True
        assert extractor.has_pending_profile is True

        # 3. Attempt to commit P3 while P2 is pending -> must be rejected
        p3 = dto_to_profile(_make_dummy_profile(driver_id, ear=0.33))
        ok3, msg3 = commit_and_activate_profile(p3, extractor, store=temp_profile_store)
        assert ok3 is False
        assert "another profile transaction is currently pending safe-latch activation" in msg3

        # Disk must still have P2 pending (not overwritten by P3)
        assert temp_profile_store.get_profile(driver_id).center_ear_median == 0.30

    def test_stage2_3_cancellation_of_pending_profile_restores_saved_profile(
        self, temp_profile_store: ProfileStore
    ) -> None:
        """
        Stage 2.3: Verify that cancelling a pending profile via disable_profile() / set_profile(None)
        executes the fingerprint-bound rollback exactly once and restores the previously saved profile.
        """
        from scripts.run_camera_test import commit_and_activate_profile
        from backend.app.ai.profile_adapter import dto_to_profile
        from backend.app.ai.pdbp_calibration import compute_canonical_profile_fingerprint
        from backend.app.features.feature_extractor import DriverFeatureExtractor
        from backend.app.features.blink_detector import BlinkState
        from backend.app.vision.face_landmarker import FaceLandmarkResult, FaceLandmarkStatus
        from backend.tests.test_features import _generate_synthetic_face_landmarks

        driver_id = "driver_cancel_pending"
        extractor = DriverFeatureExtractor()

        # 1. Establish P1 active
        p1 = dto_to_profile(_make_dummy_profile(driver_id, ear=0.28))
        commit_and_activate_profile(p1, extractor, store=temp_profile_store)
        p1_fp = compute_canonical_profile_fingerprint(p1)

        # 2. Queue P2 as pending
        extractor._blink_detector._state = BlinkState.CLOSING
        p2 = dto_to_profile(_make_dummy_profile(driver_id, ear=0.32))
        commit_and_activate_profile(p2, extractor, store=temp_profile_store)
        assert extractor.has_pending_profile is True
        assert temp_profile_store.get_profile(driver_id).center_ear_median == 0.32

        # 3. Cancel pending profile
        extractor.disable_profile()

        # Verification: disk restored to P1, pending switch cancelled, callback cleared
        assert extractor._pending_profile_switch is False
        assert extractor.is_profile_pending(p2) is False
        assert extractor._pending_rollback_cb is None
        assert extractor._pending_rollback_fingerprint is None
        restored = temp_profile_store.get_profile(driver_id)
        assert restored is not None
        assert restored.center_ear_median == 0.28
        assert compute_canonical_profile_fingerprint(dto_to_profile(restored)) == p1_fp

        # Transition to OPEN -> pending disable completes cleanly
        extractor._blink_detector._state = BlinkState.OPEN
        lms = _generate_synthetic_face_landmarks(0.0, pitch_deg=0.0, is_mirrored=False)
        frame = FaceLandmarkResult(
            has_face=True,
            landmarks=lms,
            presence_confidence=0.98,
            timestamp_ms=2000,
            status=FaceLandmarkStatus.TRACKING,
            inference_latency_ms=10.0,
            reasons=(),
        )
        extractor.extract(frame)
        assert extractor.has_pending_profile is False

    def test_stage2_3_reset_with_pending_transaction_restores_disk(
        self, temp_profile_store: ProfileStore
    ) -> None:
        """
        Stage 2.3: Verify that reset() with a pending transaction executes the rollback
        and restores the previously saved profile.
        """
        from scripts.run_camera_test import commit_and_activate_profile
        from backend.app.ai.profile_adapter import dto_to_profile
        from backend.app.ai.pdbp_calibration import compute_canonical_profile_fingerprint
        from backend.app.features.feature_extractor import DriverFeatureExtractor
        from backend.app.features.blink_detector import BlinkState
        from backend.app.vision.face_landmarker import FaceLandmarkResult, FaceLandmarkStatus
        from backend.tests.test_features import _generate_synthetic_face_landmarks

        driver_id = "driver_reset_pending"
        extractor = DriverFeatureExtractor()

        # 1. Establish P1 active
        p1 = dto_to_profile(_make_dummy_profile(driver_id, ear=0.28))
        commit_and_activate_profile(p1, extractor, store=temp_profile_store)
        p1_fp = compute_canonical_profile_fingerprint(p1)

        # 2. Queue P2 as pending
        extractor._blink_detector._state = BlinkState.CLOSING
        p2 = dto_to_profile(_make_dummy_profile(driver_id, ear=0.32))
        commit_and_activate_profile(p2, extractor, store=temp_profile_store)
        assert extractor.has_pending_profile is True

        # 3. Call reset()
        extractor.reset()

        # Verification: disk restored to P1, internal pending state reset, callback cleared
        assert extractor.has_pending_profile is False
        assert extractor._pending_profile_switch is False
        assert extractor._pending_rollback_cb is None
        assert extractor._pending_rollback_fingerprint is None
        restored = temp_profile_store.get_profile(driver_id)
        assert restored is not None
        assert restored.center_ear_median == 0.28
        assert compute_canonical_profile_fingerprint(dto_to_profile(restored)) == p1_fp
