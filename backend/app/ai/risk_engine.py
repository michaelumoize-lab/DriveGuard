"""
LEGACY / RESEARCH-ONLY COMPATIBILITY SHIM for `backend.app.ai.legacy.risk_engine`.

Production runtime MUST NOT import this module. Retained solely so offline
research scripts, dataset tools, and legacy benchmark tests continue to resolve.
"""

from backend.app.ai.legacy.risk_engine import *  # noqa: F401, F403
