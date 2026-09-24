"""Offline safety checks for the bounded M13A Cloud capability probe."""

import json
from copy import deepcopy

import pytest

from src.eval.scripts.m13a_gate0_probe import CONTRACT_PATH, CallBudget, _validate_contract


def test_frozen_contract_matches_current_inputs() -> None:
    """Live checks must stop when the approved cohort or source inputs drift."""
    contract = json.loads(CONTRACT_PATH.read_text())

    _validate_contract(contract)

    altered = deepcopy(contract)
    altered["release_cohort"]["dataset_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Frozen Gate 0 input changed"):
        _validate_contract(altered)


def test_sdk_call_budget_blocks_next_request() -> None:
    """The probe cannot start a model call beyond its approved hard cap."""
    budget = CallBudget(limit=2)

    budget.reserve()
    budget.reserve()

    with pytest.raises(RuntimeError, match="model-call cap exhausted"):
        budget.reserve()
    assert budget.attempts == 2
