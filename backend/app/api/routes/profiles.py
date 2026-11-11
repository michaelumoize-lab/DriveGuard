"""
REST Endpoints for Personalised Driver Behaviour Profiles (PDBP).
Secured with cryptographic backend token authentication and resource ownership checks.
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.core.auth import AuthenticatedUser, get_current_user, verify_resource_ownership
from backend.app.schemas.profile import DriverProfileDTO
from backend.app.services.profile_store import profile_store

router = APIRouter(prefix="/profiles", tags=["Driver Profiles"])


@router.get("", response_model=List[str])
def list_profiles(
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> List[str]:
    """List driver IDs with stored calibration profiles owned by the authenticated user."""
    all_profiles = profile_store.list_profiles()
    if current_user.is_admin:
        return all_profiles
    return [p for p in all_profiles if p == current_user.user_id]


@router.get("/{driver_id}", response_model=DriverProfileDTO)
def get_profile(
    driver_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> DriverProfileDTO:
    """Retrieve full physiological baseline profile for an authorized driver."""
    verify_resource_ownership(current_user, driver_id, "driver profile")
    profile = profile_store.get_profile(driver_id)
    if profile is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Driver profile '{driver_id}' not found.",
        )
    return profile


@router.post("", response_model=DriverProfileDTO)
def save_profile(
    profile: DriverProfileDTO,
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> DriverProfileDTO:
    """Save or update the authenticated driver's physiological profile."""
    verify_resource_ownership(current_user, profile.driver_id, "driver profile")
    profile_store.save_profile(profile)
    return profile


@router.delete("/{driver_id}", response_model=dict)
def delete_profile(
    driver_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> dict:
    """Delete a driver profile owned by the authenticated user."""
    verify_resource_ownership(current_user, driver_id, "driver profile")
    deleted = profile_store.delete_profile(driver_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Driver profile '{driver_id}' not found.",
        )
    return {"status": "DELETED", "driver_id": driver_id}
