"""
DriveGuard Core Time Utilities.
Neutral standard-library helpers for timestamp validation and parsing.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional


def validate_iso_timestamp(v: Any) -> Optional[str]:
    """
    Validate that `v` is either None or a valid ISO 8601 date-and-time string.

    Requirements:
    1. None returns None.
    2. Non-string types raise TypeError.
    3. Must contain both date and time components (delimited by 'T' or ' ').
       Date-only strings (e.g. '2026-11-17') raise ValueError.
    4. Must parse cleanly via datetime.fromisoformat (supporting 'Z' and timezone offsets).
    5. Returns `v` unaltered (byte-for-byte exact string preservation).
    """
    if v is None:
        return None

    if not isinstance(v, str):
        raise TypeError(
            f"Timestamp must be an ISO 8601 string or None, got {type(v).__name__}: {v!r}"
        )

    # Date-only strings lack a time component ('T' or ' ')
    if "T" not in v and " " not in v:
        raise ValueError(
            f"Timestamp '{v}' must contain both date and time components (e.g. YYYY-MM-DDTHH:MM:SS)"
        )

    try:
        # Python 3.11 fromisoformat accepts arbitrary ISO 8601 formats (+00:00, Z, microseconds)
        datetime.fromisoformat(v.replace("Z", "+00:00"))
    except Exception as err:
        raise ValueError(f"Invalid ISO 8601 timestamp '{v}': {err}") from err

    return v
