"""
Tests for Profile Persistence, Thread Safety, Path Sanitization, and Corrupt File Recovery (DG-03, DG-04, DG-15).
"""

import concurrent.futures
import json
import shutil
import tempfile
from pathlib import Path

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
                dynamic_closure_threshold=ear - 0.05,
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
