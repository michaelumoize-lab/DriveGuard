"""
Profile Persistence Service for Driver Behaviour Profiles (PDBP).
Hardened for thread safety, atomic file writes, path sanitization, corrupt file recovery (DG-15),
versioned metadata (`engine_type`, `profile_uuid`, `profile_version`, `base_server_version`,
`content_hash`, `sync_state`), local history archival on overwrite, and conflict-aware reconciliation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from backend.app.core.config import settings
from backend.app.schemas.profile import DriverProfileDTO

logger = logging.getLogger("driveguard.profile_store")


def compute_profile_content_hash(profile_dto: DriverProfileDTO) -> str:
    """
    Compute a deterministic SHA-256 hash over canonical biometric & calibration fields,
    ignoring volatile sync/version metadata so identical calibrations always hash equally.
    """
    zones_canonical: Dict[str, Any] = {}
    for zone_key in sorted(profile_dto.zone_baselines.keys()):
        zb = profile_dto.zone_baselines[zone_key]
        zones_canonical[zone_key] = {
            "zone": zb.zone,
            "ear_median": round(float(zb.ear_median), 6),
            "ear_mad_scaled": round(float(zb.ear_mad_scaled), 6),
            "dynamic_closure_threshold": round(float(zb.dynamic_closure_threshold), 6),
            "pitch_median_deg": round(float(zb.pitch_median_deg), 4),
            "yaw_median_deg": round(float(zb.yaw_median_deg), 4),
            "sample_count": int(zb.sample_count),
        }

    rot_canonical = None
    if profile_dto.neutral_rot_mat is not None:
        rot_canonical = [
            [round(float(val), 6) for val in row]
            for row in profile_dto.neutral_rot_mat
        ]

    canonical_obj = {
        "engine_type": (profile_dto.engine_type or "PYTHON").upper(),
        "is_calibrated": bool(profile_dto.is_calibrated),
        "calibration_mode": profile_dto.calibration_mode,
        "center_ear_median": round(float(profile_dto.center_ear_median), 6),
        "blink_duration_median_ms": round(float(profile_dto.blink_duration_median_ms), 4),
        "blink_duration_mad_ms": round(float(profile_dto.blink_duration_mad_ms), 4),
        "blink_rcr_median": round(float(profile_dto.blink_rcr_median), 6),
        "blink_rcr_mad": round(float(profile_dto.blink_rcr_mad), 6),
        "neutral_pitch_deg": round(float(profile_dto.neutral_pitch_deg), 4),
        "neutral_yaw_deg": round(float(profile_dto.neutral_yaw_deg), 4),
        "neutral_rot_mat": rot_canonical,
        "calibration_is_mirrored": profile_dto.calibration_is_mirrored,
        "zone_baselines": zones_canonical,
    }
    encoded = json.dumps(canonical_obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ProfileStore:
    """
    Thread-safe, atomic file-backed profile store for Python driver baselines.
    Serves as the authoritative local operational store for standalone Python AI
    and tracks version/sync metadata for optional PostgreSQL reconciliation.
    """

    def __init__(self, storage_dir: Optional[Path] = None) -> None:
        self.storage_dir = storage_dir or settings.PROFILES_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _file_path(self, driver_id: str) -> Path:
        """
        Derive a safe, collision-resistant, path-traversal-proof filesystem path for driver_id.
        """
        if not driver_id or not isinstance(driver_id, str):
            driver_id = "driver_default"

        sanitized = driver_id.replace("/", "").replace("\\", "").replace("\0", "").strip()
        if not sanitized or sanitized in (".", ".."):
            sanitized = "driver_default"

        safe_prefix = re.sub(r"[^a-zA-Z0-9_-]", "_", sanitized)[:48]
        if safe_prefix == sanitized:
            filename = f"{safe_prefix}_profile.json"
        else:
            hash_suffix = hashlib.sha256(driver_id.encode("utf-8")).hexdigest()[:12]
            filename = f"{safe_prefix}_{hash_suffix}_profile.json"

        target_path = (self.storage_dir / filename).resolve()
        base_dir = self.storage_dir.resolve()

        if not target_path.is_relative_to(base_dir):
            raise ValueError(f"Path traversal detected in driver_id '{driver_id}'.")

        return target_path

    def _archive_existing_profile(self, path: Path, driver_id: str, existing: Optional[DriverProfileDTO]) -> Optional[Path]:
        """Archive the existing profile JSON before overwriting so no local calibration history is lost."""
        if not path.is_file():
            return None
        try:
            history_dir = self.storage_dir / "history"
            history_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            ver = existing.profile_version if existing is not None else 0
            safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", driver_id or "driver_default")[:40]
            archive_path = history_dir / f"{safe_id}_v{ver}_{ts}_{uuid.uuid4().hex[:6]}.json"
            archive_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
            return archive_path
        except Exception as exc:
            logger.warning("[ProfileStore] Failed to archive previous profile for '%s': %s", driver_id, exc)
            return None

    def get_profile(self, driver_id: str) -> Optional[DriverProfileDTO]:
        """
        Load and parse a stored DriverProfileDTO by driver_id under thread lock.
        Gracefully handles and isolates corrupted profile files (DG-15).
        """
        with self._lock:
            try:
                p = self._file_path(driver_id)
            except ValueError as err:
                logger.error(f"[ProfileStore] Path resolution error for driver '{driver_id}': {err}")
                return None

            if not p.is_file():
                return None

            try:
                raw_text = p.read_text(encoding="utf-8")
                data = json.loads(raw_text)
                return DriverProfileDTO.model_validate(data)
            except (json.JSONDecodeError, ValidationError, UnicodeDecodeError) as err:
                logger.error(
                    f"[ProfileStore] Corrupt profile file detected for driver '{driver_id}' at {p}: {err}. "
                    "Quarantining corrupt file and returning None."
                )
                try:
                    corrupt_backup = p.with_suffix(".json.corrupt")
                    if p.is_file():
                        os.replace(p, corrupt_backup)
                except Exception as q_err:
                    logger.warning(f"[ProfileStore] Failed to quarantine corrupt profile {p}: {q_err}")
                return None
            except Exception as exc:
                logger.error(f"[ProfileStore] Unexpected error reading profile {p}: {exc}", exc_info=True)
                return None

    def save_profile(
        self,
        profile_dto: DriverProfileDTO,
        *,
        archive_previous: bool = True,
    ) -> DriverProfileDTO:
        """
        Enrich versioning/hash metadata, archive any superseded profile, and atomically
        write DriverProfileDTO to disk using temporary files (DG-15).
        Thread-safe under lock.
        """
        with self._lock:
            p = self._file_path(profile_dto.driver_id)
            existing: Optional[DriverProfileDTO] = None
            if p.is_file():
                try:
                    existing = DriverProfileDTO.model_validate(json.loads(p.read_text(encoding="utf-8")))
                except Exception:
                    existing = None

            computed_hash = compute_profile_content_hash(profile_dto)
            update_fields: Dict[str, Any] = {
                "engine_type": (profile_dto.engine_type or "PYTHON").upper(),
                "content_hash": computed_hash,
            }

            if not profile_dto.profile_uuid:
                update_fields["profile_uuid"] = str(uuid.uuid4())

            if existing is not None:
                prev_hash = existing.content_hash or compute_profile_content_hash(existing)
                if prev_hash != computed_hash:
                    if archive_previous:
                        self._archive_existing_profile(p, profile_dto.driver_id, existing)
                    # Auto-increment local version if caller did not advance it explicitly
                    if profile_dto.profile_version <= existing.profile_version:
                        update_fields["profile_version"] = existing.profile_version + 1
                    if profile_dto.base_server_version is None and existing.base_server_version is not None:
                        update_fields["base_server_version"] = existing.base_server_version
                    if "sync_state" not in profile_dto.model_fields_set:
                        update_fields["sync_state"] = "PENDING_UPLOAD"
                else:
                    # Preserve existing UUID & version on identical re-save unless explicitly changed
                    if "profile_uuid" not in profile_dto.model_fields_set and existing.profile_uuid:
                        update_fields["profile_uuid"] = existing.profile_uuid
                    if profile_dto.profile_version < existing.profile_version:
                        update_fields["profile_version"] = existing.profile_version

            enriched = profile_dto.model_copy(update=update_fields)
            json_payload = enriched.model_dump_json(indent=2)

            temp_file = None
            temp_path = None
            try:
                temp_file = tempfile.NamedTemporaryFile(
                    dir=self.storage_dir,
                    prefix="prof_tmp_",
                    suffix=".tmp",
                    delete=False,
                    mode="w",
                    encoding="utf-8",
                )
                temp_path = Path(temp_file.name)
                temp_file.write(json_payload)
                temp_file.flush()
                os.fsync(temp_file.fileno())
                temp_file.close()

                os.replace(temp_path, p)
                return enriched
            except Exception as exc:
                logger.error(f"[ProfileStore] Failed atomic write for profile '{profile_dto.driver_id}': {exc}", exc_info=True)
                if temp_path and temp_path.exists():
                    try:
                        temp_path.unlink()
                    except Exception:
                        pass
                raise

    def reconcile_profile(
        self,
        driver_id: str,
        remote_profile: DriverProfileDTO,
    ) -> Dict[str, Any]:
        """
        Reconcile a local PYTHON profile against an incoming remote/PostgreSQL PYTHON profile
        without silently overwriting unsynced local work. Archives any superseded version.
        """
        if (remote_profile.engine_type or "PYTHON").upper() != "PYTHON":
            raise ValueError("Python ProfileStore strictly refuses non-PYTHON profile reconciliation.")

        with self._lock:
            local = self.get_profile(driver_id)
            remote_hash = compute_profile_content_hash(remote_profile)

            if local is None:
                adopted = self.save_profile(
                    remote_profile.model_copy(
                        update={
                            "driver_id": driver_id,
                            "engine_type": "PYTHON",
                            "content_hash": remote_hash,
                            "base_server_version": remote_profile.profile_version,
                            "sync_state": "SYNCED",
                        }
                    ),
                    archive_previous=False,
                )
                return {
                    "action": "ADOPTED_REMOTE",
                    "archived_previous": False,
                    "profile": adopted,
                }

            local_hash = local.content_hash or compute_profile_content_hash(local)
            if local_hash == remote_hash:
                synced = self.save_profile(
                    local.model_copy(
                        update={
                            "profile_version": max(local.profile_version, remote_profile.profile_version),
                            "base_server_version": max(
                                local.base_server_version or 0,
                                remote_profile.profile_version,
                            ),
                            "sync_state": "SYNCED",
                        }
                    ),
                    archive_previous=False,
                )
                return {
                    "action": "UNCHANGED",
                    "archived_previous": False,
                    "profile": synced,
                }

            has_unsynced_local = local.sync_state in ("LOCAL_ONLY", "PENDING_UPLOAD")
            remote_is_newer_version = remote_profile.profile_version > (local.base_server_version or 0)

            if not has_unsynced_local and remote_profile.profile_version >= local.profile_version:
                adopted = self.save_profile(
                    remote_profile.model_copy(
                        update={
                            "driver_id": driver_id,
                            "engine_type": "PYTHON",
                            "content_hash": remote_hash,
                            "base_server_version": remote_profile.profile_version,
                            "sync_state": "SYNCED",
                        }
                    ),
                    archive_previous=True,
                )
                return {
                    "action": "ADOPTED_REMOTE",
                    "archived_previous": True,
                    "profile": adopted,
                }

            if has_unsynced_local and remote_is_newer_version:
                # Concurrent offline edits: compare calibrated_at deterministically & archive loser
                local_ts = local.calibrated_at or ""
                remote_ts = remote_profile.calibrated_at or ""
                if remote_ts > local_ts:
                    adopted = self.save_profile(
                        remote_profile.model_copy(
                            update={
                                "driver_id": driver_id,
                                "engine_type": "PYTHON",
                                "content_hash": remote_hash,
                                "base_server_version": remote_profile.profile_version,
                                "sync_state": "SYNCED",
                            }
                        ),
                        archive_previous=True,
                    )
                    return {
                        "action": "CONFLICT_RESOLVED_REMOTE_NEWER",
                        "archived_previous": True,
                        "profile": adopted,
                    }
                # Local is newer: preserve local with CONFLICT_PENDING_REVIEW so uploader can promote
                marked = self.save_profile(
                    local.model_copy(
                        update={
                            "base_server_version": remote_profile.profile_version,
                            "sync_state": "CONFLICT_PENDING_REVIEW",
                        }
                    ),
                    archive_previous=False,
                )
                return {
                    "action": "CONFLICT_PENDING_LOCAL_NEWER",
                    "archived_previous": False,
                    "profile": marked,
                }

            return {
                "action": "LOCAL_AHEAD_PENDING_UPLOAD",
                "archived_previous": False,
                "profile": local,
            }

    def list_profiles(self) -> List[str]:
        """List all available driver IDs stored in the directory under thread lock."""
        with self._lock:
            driver_ids: List[str] = []
            for f in self.storage_dir.glob("*_profile.json"):
                if f.name.endswith(".corrupt") or f.name.startswith("prof_tmp_"):
                    continue
                name = f.stem.replace("_profile", "")
                driver_ids.append(name)
            return sorted(set(driver_ids))

    def delete_profile(self, driver_id: str) -> bool:
        """Delete a stored profile by driver_id under thread lock."""
        with self._lock:
            try:
                p = self._file_path(driver_id)
            except ValueError:
                return False

            deleted = False
            if p.is_file():
                try:
                    p.unlink()
                    deleted = True
                except Exception as exc:
                    logger.error(f"[ProfileStore] Error unlinking {p}: {exc}")

            corrupt_path = p.with_suffix(".json.corrupt")
            if corrupt_path.is_file():
                try:
                    corrupt_path.unlink()
                except Exception:
                    pass

            return deleted

    def rollback_new_profile(
        self,
        driver_id: str,
        previous_content: Optional[str | bytes] = None,
    ) -> bool:
        """
        Roll back a newly persisted profile file by driver_id under thread lock.
        - If previous_content is None (new profile creation failed), unlinks the primary profile.
        - If previous_content is provided (profile replacement failed), restores exact previous contents.
        - Strictly preserves any existing quarantined corrupt files (.json.corrupt).
        - If rollback fails, raises an exception and logs diagnostic error; never claims success.
        """
        with self._lock:
            try:
                p = self._file_path(driver_id)
            except ValueError as err:
                logger.error(f"[ProfileStore] Path resolution error during rollback for driver '{driver_id}': {err}")
                return False

            if previous_content is None:
                if p.is_file():
                    try:
                        p.unlink()
                        logger.info(f"[ProfileStore] Successfully unlinked newly created profile for driver '{driver_id}'.")
                        return True
                    except Exception as exc:
                        logger.critical(f"[ProfileStore] Failed to unlink profile {p} during rollback: {exc}")
                        raise RuntimeError(f"Rollback failed to unlink new profile: {exc}") from exc
                return False
            else:
                try:
                    payload = previous_content if isinstance(previous_content, str) else previous_content.decode("utf-8")
                    temp_file = tempfile.NamedTemporaryFile(
                        dir=self.storage_dir,
                        prefix="prof_rb_",
                        suffix=".tmp",
                        delete=False,
                        mode="w",
                        encoding="utf-8",
                    )
                    temp_path = Path(temp_file.name)
                    temp_file.write(payload)
                    temp_file.flush()
                    os.fsync(temp_file.fileno())
                    temp_file.close()
                    os.replace(temp_path, p)
                    logger.info(f"[ProfileStore] Successfully restored previous persisted profile for driver '{driver_id}'.")
                    return True
                except Exception as exc:
                    logger.critical(f"[ProfileStore] Failed to restore previous profile for driver '{driver_id}'")
                    raise RuntimeError(f"Rollback failed to restore previous profile: {exc}") from exc


profile_store = ProfileStore()
