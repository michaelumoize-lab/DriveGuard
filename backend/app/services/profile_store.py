"""
Profile Persistence Service for Driver Behaviour Profiles (PDBP).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from backend.app.core.config import settings
from backend.app.schemas.profile import DriverProfileDTO


class ProfileStore:
    """Thread-safe file-backed profile store for driver baselines."""

    def __init__(self, storage_dir: Optional[Path] = None) -> None:
        self.storage_dir = storage_dir or settings.PROFILES_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _file_path(self, driver_id: str) -> Path:
        safe_id = "".join(c for c in driver_id if c.isalnum() or c in ("-", "_"))
        return self.storage_dir / f"{safe_id}_profile.json"

    def get_profile(self, driver_id: str) -> Optional[DriverProfileDTO]:
        """Load and parse a stored DriverProfileDTO by driver_id."""
        p = self._file_path(driver_id)
        if not p.is_file():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return DriverProfileDTO.model_validate(data)
        except Exception:
            return None

    def save_profile(self, profile_dto: DriverProfileDTO) -> None:
        """Serialize and save a DriverProfileDTO to disk."""
        p = self._file_path(profile_dto.driver_id)
        p.write_text(profile_dto.model_dump_json(indent=2), encoding="utf-8")

    def list_profiles(self) -> List[str]:
        """List all available driver IDs stored in the directory."""
        driver_ids: List[str] = []
        for f in self.storage_dir.glob("*_profile.json"):
            name = f.stem.replace("_profile", "")
            driver_ids.append(name)
        return sorted(driver_ids)

    def delete_profile(self, driver_id: str) -> bool:
        """Delete a stored profile by driver_id."""
        p = self._file_path(driver_id)
        if p.is_file():
            p.unlink()
            return True
        return False


profile_store = ProfileStore()
