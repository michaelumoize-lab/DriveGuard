"""
Profile Persistence Service for Driver Behaviour Profiles (PDBP).
Hardened for thread safety, atomic file writes, path sanitization, and corrupt file recovery (DG-15).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import List, Optional

from pydantic import ValidationError

from backend.app.core.config import settings
from backend.app.schemas.profile import DriverProfileDTO

logger = logging.getLogger("driveguard.profile_store")


class ProfileStore:
    """
    Thread-safe, atomic file-backed profile store for driver baselines.
    Serves as local cache and fallback for PostgreSQL calibration profiles.
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

        # Sanitize against path traversal attacks (DG-15)
        # Strip directory separators and null bytes
        sanitized = driver_id.replace("/", "").replace("\\", "").replace("\0", "").strip()
        if not sanitized or sanitized in (".", ".."):
            sanitized = "driver_default"

        # If driver_id contains special characters, append a deterministic SHA-256 hash to prevent collisions
        safe_prefix = re.sub(r"[^a-zA-Z0-9_-]", "_", sanitized)[:48]
        if safe_prefix == sanitized:
            filename = f"{safe_prefix}_profile.json"
        else:
            hash_suffix = hashlib.sha256(driver_id.encode("utf-8")).hexdigest()[:12]
            filename = f"{safe_prefix}_{hash_suffix}_profile.json"

        target_path = (self.storage_dir / filename).resolve()
        base_dir = self.storage_dir.resolve()

        # Enforce that the target path remains strictly inside the storage directory
        if not target_path.is_relative_to(base_dir):
            raise ValueError(f"Path traversal detected in driver_id '{driver_id}'.")

        return target_path

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
                    # Quarantine corrupt file so it does not permanently poison subsequent reads
                    corrupt_backup = p.with_suffix(".json.corrupt")
                    if p.is_file():
                        os.replace(p, corrupt_backup)
                except Exception as q_err:
                    logger.warning(f"[ProfileStore] Failed to quarantine corrupt profile {p}: {q_err}")
                return None
            except Exception as exc:
                logger.error(f"[ProfileStore] Unexpected error reading profile {p}: {exc}", exc_info=True)
                return None

    def save_profile(self, profile_dto: DriverProfileDTO) -> None:
        """
        Serialize and atomically write a DriverProfileDTO to disk using temporary files (DG-15).
        Thread-safe under lock.
        """
        with self._lock:
            p = self._file_path(profile_dto.driver_id)
            json_payload = profile_dto.model_dump_json(indent=2)

            # Atomic write pattern: write to temporary file in same filesystem directory, then replace
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

                # Atomic replace on NTFS and POSIX
                os.replace(temp_path, p)
            except Exception as exc:
                logger.error(f"[ProfileStore] Failed atomic write for profile '{profile_dto.driver_id}': {exc}", exc_info=True)
                if temp_path and temp_path.exists():
                    try:
                        temp_path.unlink()
                    except Exception:
                        pass
                raise

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

            # Also clean up any quarantined corrupt file
            corrupt_path = p.with_suffix(".json.corrupt")
            if corrupt_path.is_file():
                try:
                    corrupt_path.unlink()
                except Exception:
                    pass

            return deleted


profile_store = ProfileStore()
