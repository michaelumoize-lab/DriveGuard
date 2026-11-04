"""
REST Endpoints for Personalised Driver Behaviour Profiles (PDBP).
"""

from typing import List
from fastapi import APIRouter, HTTPException

from backend.app.schemas.profile import DriverProfileDTO
from backend.app.services.profile_store import profile_store

router = APIRouter(prefix="/profiles", tags=["Driver Profiles"])


@router.get("", response_model=List[str])
def list_profiles() -> List[str]:
    """List all driver IDs with stored calibration profiles."""
    return profile_store.list_profiles()


@router.get("/{driver_id}", response_model=DriverProfileDTO)
def get_profile(driver_id: str) -> DriverProfileDTO:
    """Retrieve full 4-zone physiological baseline profile for a driver."""
    profile = profile_store.get_profile(driver_id)
    if profile is None:
        raise HTTPException(
            status_code=404,
            detail=f"Driver profile '{driver_id}' not found.",
        )
    return profile


@router.post("", response_model=DriverProfileDTO)
def save_profile(profile: DriverProfileDTO) -> DriverProfileDTO:
    """Save or update a driver's physiological profile."""
    profile_store.save_profile(profile)
    return profile


@router.delete("/{driver_id}", response_model=dict)
def delete_profile(driver_id: str) -> dict:
    """Delete a driver profile."""
    deleted = profile_store.delete_profile(driver_id)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Driver profile '{driver_id}' not found.",
        )
    return {"status": "DELETED", "driver_id": driver_id}
