"""Property-based tests for OPA Policy Evaluation Determinism.

**Validates: Requirements 7.2, 7.4**

Property 12: OPA Policy Evaluation Determinism
- For any valid input conforming to rules.rego schema, evaluating the same
  input always produces the same PASS/FAIL result.
- FAIL always cites specific violated rule identifiers (non-empty rule_id).
- The violated_rules list is identical across repeated evaluations.
"""

import asyncio
import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from clinical_reasoning_fabric.beacon.opa_challenger_service import (
    OPAChallengerService,
    OPAEvaluator,
    PolicyEvaluationResult,
    PolicyViolation,
)


# =============================================================================
# Deterministic Mock OPA Evaluator
# =============================================================================


class DeterministicOPAEvaluator:
    """A deterministic OPA evaluator that consistently returns the same result
    for the same input by using a hash of the input to determine pass/fail.

    This ensures that the property test validates the OPAChallengerService's
    evaluate_policy method produces deterministic results when the underlying
    evaluator is deterministic.
    """

    # Well-known rule violations for deterministic FAIL cases
    VIOLATION_RULES = [
        {"rule_id": "RULE-001", "description": "Missing conservative therapy documentation"},
        {"rule_id": "RULE-002", "description": "Insufficient symptom duration"},
        {"rule_id": "RULE-003", "description": "No objective findings documented"},
        {"rule_id": "RULE-004", "description": "Specialist consult not completed"},
        {"rule_id": "RULE-005", "description": "Radiograph evidence not provided"},
    ]

    async def evaluate(self, policy_path: str, input_data: dict) -> dict:
        """Evaluate using a deterministic hash-based approach.

        The same input_data always produces the same result.
        Uses SHA-256 of the canonical JSON representation to determine outcome.
        """
        # Create a stable hash of the input
        canonical = json.dumps(input_data, sort_keys=True, default=str)
        digest = hashlib.sha256(canonical.encode()).hexdigest()

        # Use first byte of hash to determine pass/fail (deterministic)
        # If first hex digit is 0-7 (50% chance) -> PASS, 8-f -> FAIL
        first_nibble = int(digest[0], 16)
        passes = first_nibble < 8

        if passes:
            return {"result": True, "violations": []}
        else:
            # Use subsequent bytes to determine which rules are violated
            # This ensures the same input always produces the same violations
            num_violations = (int(digest[1], 16) % len(self.VIOLATION_RULES)) + 1
            violations = self.VIOLATION_RULES[:num_violations]
            return {"result": False, "violations": violations}


# =============================================================================
# Dummy KMS Client (not used in policy evaluation but required for init)
# =============================================================================


class DummyKMSClient:
    """Minimal KMS client stub for OPAChallengerService initialization."""

    async def verify_signature(self, content_hash: str, signature: str, key_id: str) -> bool:
        return True


# =============================================================================
# Hypothesis Strategies
# =============================================================================


# Strategy for generating valid decision context dicts conforming to rules.rego schema
@st.composite
def decision_context_strategy(draw):
    """Generate random decision contexts matching the schema used by
    _build_decision_context in OPAChallengerService.

    These are the kinds of dicts that evaluate_policy receives.
    """
    execution_id = draw(
        st.text(
            alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="-_"),
            min_size=1,
            max_size=50,
        )
    )
    decision = draw(st.sampled_from(["approve", "escalate", "deny", "pending"]))
    reason = draw(
        st.text(
            alphabet=st.characters(whitelist_categories=("L", "N", "Z")),
            min_size=1,
            max_size=200,
        )
    )
    lineage_trail_count = draw(st.integers(min_value=1, max_value=20))
    signature_count = draw(st.integers(min_value=1, max_value=10))
    has_execution_trace = draw(st.booleans())

    # Generate lineage entries
    lineage_entries = []
    for _ in range(draw(st.integers(min_value=1, max_value=5))):
        entry = {
            "conclusion": draw(
                st.text(
                    alphabet=st.characters(whitelist_categories=("L", "N", "Z")),
                    min_size=1,
                    max_size=100,
                )
            ),
            "evidence_id": draw(
                st.text(
                    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="-_"),
                    min_size=1,
                    max_size=30,
                )
            ),
            "retrieval_timestamp": draw(
                st.datetimes(
                    min_value=datetime(2020, 1, 1),
                    max_value=datetime(2030, 12, 31),
                )
            ).isoformat(),
        }
        lineage_entries.append(entry)

    return {
        "execution_id": execution_id,
        "decision": decision,
        "reason": reason,
        "lineage_trail_count": lineage_trail_count,
        "signature_count": signature_count,
        "has_execution_trace": has_execution_trace,
        "lineage_entries": lineage_entries,
    }


# =============================================================================
# Helpers
# =============================================================================


def _make_service() -> OPAChallengerService:
    """Create an OPAChallengerService with the deterministic OPA evaluator."""
    return OPAChallengerService(
        kms_client=DummyKMSClient(),
        opa_evaluator=DeterministicOPAEvaluator(),
    )


# =============================================================================
# Property 12: OPA Policy Evaluation Determinism
# =============================================================================


@pytest.mark.property
class TestOPAPolicyEvaluationDeterminism:
    """Property 12: OPA Policy Evaluation Determinism.

    **Validates: Requirements 7.2, 7.4**

    For any valid input conforming to rules.rego schema:
    1. Evaluating the same input always produces the same PASS/FAIL result.
    2. FAIL always cites specific violated rule identifiers (non-empty rule_id).
    3. The violated_rules list is identical across repeated evaluations.
    """

    @given(decision_context=decision_context_strategy())
    @settings(max_examples=200)
    @pytest.mark.asyncio
    async def test_determinism_same_input_same_result(self, decision_context):
        """Evaluating the same input twice produces identical PASS/FAIL outcomes.

        **Validates: Requirements 7.2**

        For any valid decision context, calling evaluate_policy with the same
        input must always produce the same `passed` boolean value.
        """
        service = _make_service()

        result1 = await service.evaluate_policy(decision_context)
        result2 = await service.evaluate_policy(decision_context)

        assert result1.passed == result2.passed, (
            f"Determinism violated: same input produced different outcomes. "
            f"First: passed={result1.passed}, Second: passed={result2.passed}. "
            f"Input: {decision_context}"
        )

    @given(decision_context=decision_context_strategy())
    @settings(max_examples=200)
    @pytest.mark.asyncio
    async def test_fail_always_cites_violated_rules(self, decision_context):
        """Every FAIL result has at least one violated rule with a non-empty rule_id.

        **Validates: Requirements 7.4**

        When OPA policy evaluation identifies a rule violation (passed=False),
        the result must cite at least one specific violated rule identifier.
        """
        service = _make_service()

        result = await service.evaluate_policy(decision_context)

        if not result.passed:
            assert len(result.violated_rules) > 0, (
                f"FAIL result must cite at least one violated rule, but got empty list. "
                f"Input: {decision_context}"
            )
            for violation in result.violated_rules:
                assert violation.rule_id is not None and len(violation.rule_id.strip()) > 0, (
                    f"Violated rule must have a non-empty rule_id, got: '{violation.rule_id}'"
                )
                assert violation.description is not None and len(violation.description.strip()) > 0, (
                    f"Violated rule must have a non-empty description, got: '{violation.description}'"
                )

    @given(decision_context=decision_context_strategy())
    @settings(max_examples=200)
    @pytest.mark.asyncio
    async def test_consistency_violated_rules_identical_across_evaluations(self, decision_context):
        """The violated_rules list is identical across repeated evaluations.

        **Validates: Requirements 7.2, 7.4**

        For the same input, not only must the pass/fail be the same, but
        the exact list of violated rules (rule_ids and descriptions) must
        be identical across evaluations.
        """
        service = _make_service()

        result1 = await service.evaluate_policy(decision_context)
        result2 = await service.evaluate_policy(decision_context)

        # Same number of violated rules
        assert len(result1.violated_rules) == len(result2.violated_rules), (
            f"Violated rules count differs: {len(result1.violated_rules)} vs "
            f"{len(result2.violated_rules)} for same input"
        )

        # Same rule_ids in same order
        rule_ids_1 = [v.rule_id for v in result1.violated_rules]
        rule_ids_2 = [v.rule_id for v in result2.violated_rules]
        assert rule_ids_1 == rule_ids_2, (
            f"Violated rule IDs differ across evaluations: {rule_ids_1} vs {rule_ids_2}"
        )

        # Same descriptions in same order
        descriptions_1 = [v.description for v in result1.violated_rules]
        descriptions_2 = [v.description for v in result2.violated_rules]
        assert descriptions_1 == descriptions_2, (
            f"Violated rule descriptions differ across evaluations: "
            f"{descriptions_1} vs {descriptions_2}"
        )

    @given(decision_context=decision_context_strategy())
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_pass_result_has_no_violated_rules(self, decision_context):
        """A PASS result has an empty violated_rules list.

        **Validates: Requirements 7.2**

        When policy evaluation passes, there should be no violated rules cited.
        """
        service = _make_service()

        result = await service.evaluate_policy(decision_context)

        if result.passed:
            assert len(result.violated_rules) == 0, (
                f"PASS result should have no violated rules, but got: "
                f"{[v.rule_id for v in result.violated_rules]}"
            )

    @given(decision_context=decision_context_strategy())
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_evaluated_at_timestamp_present(self, decision_context):
        """Every evaluation result has a valid evaluated_at timestamp.

        **Validates: Requirements 7.2**

        The evaluation result must always include a timestamp indicating
        when the evaluation occurred.
        """
        service = _make_service()

        result = await service.evaluate_policy(decision_context)

        assert result.evaluated_at is not None, (
            "PolicyEvaluationResult must have a non-null evaluated_at timestamp"
        )
        assert isinstance(result.evaluated_at, datetime), (
            f"evaluated_at must be a datetime, got {type(result.evaluated_at)}"
        )
