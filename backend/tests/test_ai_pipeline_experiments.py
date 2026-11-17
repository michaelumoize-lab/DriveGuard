"""
Tests for Phase 4: AI Pipeline Correctness, Unified Experiment Switching,
Provenance Diagnostics, and Research Validity (DG-05, DG-08, DG-09).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import List
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.ai.cascade_classifier import (
    ConfidenceGatedEyeCascade,
    ExperimentMode,
    MobileNetV3EyeVerifier,
)
from backend.app.ai.pipeline import SessionAIEngine
from backend.app.core.auth import create_backend_access_token
from backend.app.main import app
from backend.app.services.session_manager import session_manager
from backend.research.train_mobilenetv3_mrl import (
    MRLEyeSample,
    compute_roc_auc,
    get_split_statistics,
    split_subject_disjoint,
)


# -----------------------------------------------------------------------------
# 1. DG-09: Research Validity & ROC-AUC Tests
# -----------------------------------------------------------------------------

def test_compute_roc_auc_single_class_returns_none() -> None:
    """AUC must return None (undefined) on single-class or empty datasets, never false 1.0."""
    # Empty inputs
    assert compute_roc_auc([], []) is None

    # All positive (single class 1)
    assert compute_roc_auc([1, 1, 1, 1], [0.9, 0.8, 0.7, 0.95]) is None

    # All negative (single class 0)
    assert compute_roc_auc([0, 0, 0, 0], [0.1, 0.2, 0.15, 0.05]) is None


def test_compute_roc_auc_binary_class_accurate() -> None:
    """AUC must compute exact Wilcoxon-Mann-Whitney rank statistic on binary sets."""
    # Perfect separation
    y_true = [0, 0, 1, 1]
    y_prob = [0.1, 0.2, 0.8, 0.9]
    assert compute_roc_auc(y_true, y_prob) == 1.0

    # Inverted separation
    y_true_inv = [0, 0, 1, 1]
    y_prob_inv = [0.9, 0.8, 0.2, 0.1]
    assert compute_roc_auc(y_true_inv, y_prob_inv) == 0.0

    # Random chance separation
    y_true_mixed = [0, 1, 0, 1]
    y_prob_mixed = [0.5, 0.5, 0.5, 0.5]
    auc = compute_roc_auc(y_true_mixed, y_prob_mixed)
    assert auc is not None
    assert 0.0 <= auc <= 1.0


def test_split_subject_disjoint_strictly_isolates_participants() -> None:
    """Split must guarantee zero subject intersection between train, val, and test."""
    samples: List[MRLEyeSample] = []
    # Create 10 subjects with 20 samples each
    for subj_idx in range(1, 11):
        subj_id = f"s{subj_idx:04d}"
        for img_idx in range(20):
            samples.append(
                MRLEyeSample(
                    path=Path(f"{subj_id}_{img_idx}.png"),
                    subject_id=subj_id,
                    label_closed=img_idx % 2,
                    glasses=0,
                    reflections=0,
                    lighting=1,
                )
            )

    train_split, val_split, test_split = split_subject_disjoint(samples, train_ratio=0.7, val_ratio=0.15, seed=42)

    train_subjs = {s.subject_id for s in train_split}
    val_subjs = {s.subject_id for s in val_split}
    test_subjs = {s.subject_id for s in test_split}

    # Strict disjointness assertions
    assert len(train_subjs.intersection(val_subjs)) == 0
    assert len(train_subjs.intersection(test_subjs)) == 0
    assert len(val_subjs.intersection(test_subjs)) == 0

    # All splits non-empty
    assert len(train_subjs) > 0
    assert len(val_subjs) > 0
    assert len(test_subjs) > 0

    # Statistics reporting integrity
    stats = get_split_statistics(train_split, val_split, test_split)
    assert stats["train"]["num_subjects"] == len(train_subjs)
    assert stats["validation"]["num_subjects"] == len(val_subjs)
    assert stats["test"]["num_subjects"] == len(test_subjs)
    assert stats["train"]["open_count"] + stats["train"]["closed_count"] == len(train_split)


# -----------------------------------------------------------------------------
# 2. DG-08: Model Provenance & Verifier Diagnostics
# -----------------------------------------------------------------------------

def test_verifier_diagnostics_and_provenance() -> None:
    """Verifier must honestly report execution mode and model path without synthetic claims."""
    # Instantiated without an ONNX file (or default fallback)
    verifier = MobileNetV3EyeVerifier(onnx_model_path="nonexistent_path.onnx")
    diag = verifier.get_diagnostics()

    assert diag["is_pretrained_onnx"] is False
    assert diag["execution_mode"] == "ANALYTICAL_FALLBACK"
    assert diag["session_active"] is False

    cascade = ConfidenceGatedEyeCascade(mode=ExperimentMode.A_GEOMETRIC_ONLY, onnx_model_path="nonexistent.onnx")
    c_diag = cascade.get_diagnostics()
    assert c_diag["mode"] == ExperimentMode.A_GEOMETRIC_ONLY.value
    assert c_diag["verifier"]["execution_mode"] == "ANALYTICAL_FALLBACK"


# -----------------------------------------------------------------------------
# 3. DG-05: Experiment Switching Across Engine and REST
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_engine_synchronous_experiment_mode_transition() -> None:
    """Engine must switch modes synchronously and reject invalid mode transitions."""
    engine = SessionAIEngine(
        session_id="test_exp_engine",
        driver_id="driver_exp",
        experiment_mode=ExperimentMode.C_CONFIDENCE_GATED_CASCADE,
    )
    try:
        assert engine.experiment_mode == ExperimentMode.C_CONFIDENCE_GATED_CASCADE
        assert engine.cascade.mode == ExperimentMode.C_CONFIDENCE_GATED_CASCADE

        # Switch to Mode A (Geometric only)
        new_mode = await engine.submit_command("SWITCH_EXPERIMENT", mode="EXPERIMENT_A_GEOMETRIC_ONLY")
        assert new_mode == ExperimentMode.A_GEOMETRIC_ONLY
        assert engine.experiment_mode == ExperimentMode.A_GEOMETRIC_ONLY
        assert engine.cascade.mode == ExperimentMode.A_GEOMETRIC_ONLY

        # Switch to Mode B (CNN always)
        new_mode = await engine.submit_command("SWITCH_EXPERIMENT", mode="B_CNN_ALWAYS")
        assert new_mode == ExperimentMode.B_CNN_ALWAYS
        assert engine.experiment_mode == ExperimentMode.B_CNN_ALWAYS
        assert engine.cascade.mode == ExperimentMode.B_CNN_ALWAYS

        # Invalid mode must raise ValueError and preserve current mode
        with pytest.raises(ValueError):
            await engine.submit_command("SWITCH_EXPERIMENT", mode="INVALID_MODE_NAME")
        assert engine.experiment_mode == ExperimentMode.B_CNN_ALWAYS
    finally:
        engine.close()


def test_rest_api_experiment_mode_switch() -> None:
    """REST API must enforce authentication, ownership, and mode switching."""
    client = TestClient(app)
    user_alice = "user_alice_exp"
    user_bob = "user_bob_exp"

    token_alice = create_backend_access_token(user_alice, role="user", session_id="auth_sess_1")
    token_bob = create_backend_access_token(user_bob, role="user", session_id="auth_sess_2")
    headers_alice = {"Authorization": f"Bearer {token_alice}"}
    headers_bob = {"Authorization": f"Bearer {token_bob}"}

    # Alice creates a session
    sess_id = "sess_alice_exp_test"
    resp_create = client.post(
        "/api/v1/sessions",
        json={"session_id": sess_id, "experiment_mode": "EXPERIMENT_C_CONFIDENCE_GATED_CASCADE"},
        headers=headers_alice,
    )
    assert resp_create.status_code == 200

    # Bob attempts to switch Alice's session -> 403 Forbidden
    resp_bob_switch = client.post(
        f"/api/v1/sessions/{sess_id}/experiment",
        json={"experiment_mode": "EXPERIMENT_A_GEOMETRIC_ONLY"},
        headers=headers_bob,
    )
    assert resp_bob_switch.status_code == 403

    # Alice switches with invalid mode -> 400 Bad Request
    resp_invalid = client.post(
        f"/api/v1/sessions/{sess_id}/experiment",
        json={"experiment_mode": "NONEXISTENT_MODE"},
        headers=headers_alice,
    )
    assert resp_invalid.status_code == 400

    # Alice switches to Mode A -> 200 OK
    resp_valid = client.post(
        f"/api/v1/sessions/{sess_id}/experiment",
        json={"experiment_mode": "EXPERIMENT_A_GEOMETRIC_ONLY"},
        headers=headers_alice,
    )
    assert resp_valid.status_code == 200
    assert resp_valid.json()["experiment_mode"] == "EXPERIMENT_A_GEOMETRIC_ONLY"

    # Verify session summary reflects diagnostics and mode
    resp_details = client.get(f"/api/v1/sessions/{sess_id}", headers=headers_alice)
    assert resp_details.status_code == 200
    summary = resp_details.json()
    assert "cascade_diagnostics" in summary
    assert summary["cascade_diagnostics"]["mode"] == "EXPERIMENT_A_GEOMETRIC_ONLY"

    # Cleanup session
    client.delete(f"/api/v1/sessions/{sess_id}", headers=headers_alice)
