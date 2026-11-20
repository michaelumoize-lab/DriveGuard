"""
Canonical ExperimentMode enum shared across FastAPI session/transport layers
and experimental benchmarks without importing legacy research classifiers.
"""

from __future__ import annotations

from enum import Enum


class ExperimentMode(str, Enum):
    """Experimental modes for benchmarking the eye-state analysis pipeline."""

    A_GEOMETRIC_ONLY = "EXPERIMENT_A_GEOMETRIC_ONLY"
    B_CNN_ALWAYS = "EXPERIMENT_B_CNN_ALWAYS"
    C_CONFIDENCE_GATED_CASCADE = "EXPERIMENT_C_CONFIDENCE_GATED_CASCADE"
