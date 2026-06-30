"""Unit tests for OPAChallengerService (BEACON Layer 5).

Tests KMS signature verification, OPA policy evaluation, decision orchestration,
infrastructure failure handling, and escalation behavior.

Validates:
    - verify_signatures() verifies all signatures within 30s; reports invalid/missing
    - evaluate_policy() evaluates against OPA rules within 10s; produces PASS/FAIL with rule IDs
    - verify_decision() orchestrates signature verification then policy evaluation
    - Invalid/missing signatures trigger FAIL with tamper alert identifying affected snippets
    - OPA violations trigger FAIL with violated rule identifiers and descriptions
    - Infrastructure failures (KMS unavailable, OPA unreachable) treated as FAIL with escalation
    - No shared mutable state with primary reasoning agent (stateless design)

Requirements referenced: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from clinical_reasoning_fabric.beacon.opa_challenger_service import (
    ChallengerResult,
    OPAChallengerService,
    PolicyEvaluationResult,
    PolicyViolation,
    SignatureVerificationResult,
)
from clinical_reasoning_fabric.models.core import (
    EvidenceBundle,
    KMSSignature,
    LineageEntry,
    TamperAlert,
    VerificationResult,
)
from clinical_reasoning_fabric.models.exceptions import KMSUnavailableError


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def kms_client():
    """Mock KMS client that verifies all signatures as valid by default."""
    client = AsyncMock()
    client.verify_signature = AsyncMock(return_value=True)
    return client


@pytest.fixture
def opa_evaluator():
    """Mock OPA evaluator that passes all policies by default."""
    evaluator = AsyncMock()
    evaluator.evaluate = AsyncMock(
        return_value={"result": True, "violations": []}
    )
    return evaluator


@pytest.fixture
def challenger(kms_client, opa_evaluator):
    """OPAChallengerService with mock dependencies."""
    return OPAChallengerService(
        kms_client=kms_client, opa_evaluator=opa_evaluator
    )


@pytest.fixture
def valid_signatures():
    """List of valid KMS signatures for testing."""
    return [
        KMSSignature(
            key_id="key-001",
            signature="c2lnbmF0dXJlMQ==",
            algorithm="RSASSA_PKCS1_V1_5_SHA_256",
            signed_at=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
        ),
        KMSSignature(
            key_id="key-002",
            signature="c2lnbmF0dXJlMg==",
            algorithm="RSASSA_PKCS1_V1_5_SHA_256",
            signed_at=datetime(2024, 1, 15, 10, 31, 0, tzinfo=timezone.utc),
        ),
    ]


@pytest.fixture
def valid_lineage_trail():
    """Valid lineage trail entries for testing."""
    return [
        LineageEntry(
            conclusion="Patient meets criteria for treatment",
            evidence_id="chunk-001",
            retrieval_timestamp=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
        ),
        LineageEntry(
            conclusion="Conservative therapy documented for 8 weeks",
            evidence_id="chunk-002",
            retrieval_timestamp=datetime(2024, 1, 15, 10, 1, 0, tzinfo=timezone.utc),
        ),
    ]


@pytest.fixture
def evidence_bundle(valid_signatures, valid_lineage_trail):
    """Valid evidence bundle for testing."""
    return EvidenceBundle(
        execution_id="exec-001",
        decision="approve",
        reason="All clinical criteria met",
        lineage_trail=valid_lineage_trail,
        original_document_signatures=valid_signatures,
    )


# =============================================================================
# Verify Signatures — Success Tests
# =============================================================================


class TestVerifySignaturesSuccess:
    """Tests for successful signature verification (Requirement 7.1)."""

    async def test_all_valid_signatures_returns_all_valid(
        self, challenger, valid_signatures
    ):
        """All valid signatures produce all_valid=True."""
        result = await challenger.verify_signatures(valid_signatures)

        assert isinstance(result, SignatureVerificationResult)
        assert result.all_valid is True
        assert len(result.valid) == 2
        assert len(result.invalid) == 0
        assert len(result.missing) == 0
        assert len(result.tamper_alerts) == 0

    async def test_single_valid_signature(self, challenger, valid_signatures):
        """Single valid signature is correctly identified."""
        result = await challenger.verify_signatures([valid_signatures[0]])

        assert result.all_valid is True
        assert len(result.valid) == 1

    async def test_verification_timestamp_is_set(self, challenger, valid_signatures):
        """verified_at timestamp is set on result."""
        result = await challenger.verify_signatures(valid_signatures)

        assert result.verified_at is not None
        assert isinstance(result.verified_at, datetime)


# =============================================================================
# Verify Signatures — Failure Tests
# =============================================================================


class TestVerifySignaturesFailure:
    """Tests for signature verification failures (Requirements 7.1, 7.3)."""

    async def test_invalid_signature_detected(self, kms_client, opa_evaluator):
        """Invalid signature produces invalid list entry and tamper alert."""
        # Second signature fails verification
        kms_client.verify_signature = AsyncMock(
            side_effect=[True, False]
        )
        challenger = OPAChallengerService(kms_client, opa_evaluator)
        sigs = [
            KMSSignature(
                key_id="key-001",
                signature="valid_sig",
                algorithm="RSASSA_PKCS1_V1_5_SHA_256",
                signed_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
            ),
            KMSSignature(
                key_id="key-002",
                signature="invalid_sig",
                algorithm="RSASSA_PKCS1_V1_5_SHA_256",
                signed_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
            ),
        ]

        result = await challenger.verify_signatures(sigs)

        assert result.all_valid is False
        assert len(result.valid) == 1
        assert len(result.invalid) == 1
        assert len(result.tamper_alerts) == 1
        assert "key-002" in result.invalid[0]

    async def test_missing_signature_detected(self, kms_client, opa_evaluator):
        """Signature that fails verification due to error is treated as missing."""
        kms_client.verify_signature = AsyncMock(
            side_effect=ValueError("Invalid signature format")
        )
        challenger = OPAChallengerService(kms_client, opa_evaluator)
        sigs = [
            KMSSignature(
                key_id="key-001",
                signature="corrupted_but_nonempty",
                algorithm="RSASSA_PKCS1_V1_5_SHA_256",
                signed_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
            ),
        ]

        result = await challenger.verify_signatures(sigs)

        assert result.all_valid is False
        # Verification error treated as missing/unverifiable
        assert len(result.missing) == 1
        assert len(result.tamper_alerts) == 1

    async def test_empty_signatures_list_reports_missing(
        self, challenger
    ):
        """Empty signatures list produces missing entry."""
        result = await challenger.verify_signatures([])

        assert result.all_valid is False
        assert len(result.missing) == 1
        assert len(result.tamper_alerts) == 1
        assert "no_signatures_provided" in result.missing[0]

    async def test_tamper_alert_includes_affected_snippet_info(
        self, kms_client, opa_evaluator
    ):
        """Tamper alert identifies the affected evidence snippet."""
        kms_client.verify_signature = AsyncMock(return_value=False)
        challenger = OPAChallengerService(kms_client, opa_evaluator)
        sigs = [
            KMSSignature(
                key_id="key-tampered",
                signature="bad_signature_data",
                algorithm="RSASSA_PKCS1_V1_5_SHA_256",
                signed_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
            ),
        ]

        result = await challenger.verify_signatures(sigs)

        assert len(result.tamper_alerts) == 1
        alert = result.tamper_alerts[0]
        assert isinstance(alert, TamperAlert)
        assert alert.expected_signature == "bad_signature_data"
        assert "key-tampered" in alert.document_id
        assert alert.detected_at is not None


# =============================================================================
# Verify Signatures — Infrastructure Failure Tests
# =============================================================================


class TestVerifySignaturesInfraFailure:
    """Tests for KMS unavailability (Requirement 7.6)."""

    async def test_kms_unavailable_raises_error(self, opa_evaluator):
        """KMS unavailability raises KMSUnavailableError."""
        kms_client = AsyncMock()
        kms_client.verify_signature = AsyncMock(
            side_effect=KMSUnavailableError(reason="KMS service unreachable")
        )
        challenger = OPAChallengerService(kms_client, opa_evaluator)
        sigs = [
            KMSSignature(
                key_id="key-001",
                signature="sig_data",
                algorithm="RSASSA_PKCS1_V1_5_SHA_256",
                signed_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
            ),
        ]

        with pytest.raises(KMSUnavailableError):
            await challenger.verify_signatures(sigs)

    async def test_kms_timeout_raises_timeout_error(self, opa_evaluator):
        """KMS operations exceeding 30s raise TimeoutError."""
        kms_client = AsyncMock()

        async def slow_verify(*args, **kwargs):
            await asyncio.sleep(35)
            return True

        kms_client.verify_signature = slow_verify
        challenger = OPAChallengerService(kms_client, opa_evaluator)
        # Override timeout for faster test
        challenger.SIGNATURE_TIMEOUT_SECONDS = 0.1

        sigs = [
            KMSSignature(
                key_id="key-001",
                signature="sig_data",
                algorithm="RSASSA_PKCS1_V1_5_SHA_256",
                signed_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
            ),
        ]

        with pytest.raises(asyncio.TimeoutError):
            await challenger.verify_signatures(sigs)


# =============================================================================
# Evaluate Policy — Success Tests
# =============================================================================


class TestEvaluatePolicySuccess:
    """Tests for successful OPA policy evaluation (Requirement 7.2)."""

    async def test_all_rules_pass_returns_passed_true(self, challenger):
        """When all OPA rules pass, result.passed is True."""
        decision_context = {
            "execution_id": "exec-001",
            "decision": "approve",
            "reason": "All criteria met",
        }

        result = await challenger.evaluate_policy(decision_context)

        assert isinstance(result, PolicyEvaluationResult)
        assert result.passed is True
        assert len(result.violated_rules) == 0

    async def test_evaluation_timestamp_is_set(self, challenger):
        """evaluated_at timestamp is populated."""
        result = await challenger.evaluate_policy({"decision": "approve"})

        assert result.evaluated_at is not None
        assert isinstance(result.evaluated_at, datetime)


# =============================================================================
# Evaluate Policy — Failure Tests
# =============================================================================


class TestEvaluatePolicyFailure:
    """Tests for OPA policy violations (Requirements 7.2, 7.4)."""

    async def test_policy_violation_returns_failed_with_rule_ids(
        self, kms_client
    ):
        """OPA violations produce FAIL with violated rule identifiers."""
        opa_evaluator = AsyncMock()
        opa_evaluator.evaluate = AsyncMock(
            return_value={
                "result": False,
                "violations": [
                    {
                        "rule_id": "conservative_therapy_met",
                        "description": "Conservative therapy duration < 6 weeks",
                    },
                    {
                        "rule_id": "objective_findings_met",
                        "description": "No objective findings documented",
                    },
                ],
            }
        )
        challenger = OPAChallengerService(kms_client, opa_evaluator)

        result = await challenger.evaluate_policy({"decision": "approve"})

        assert result.passed is False
        assert len(result.violated_rules) == 2
        assert result.violated_rules[0].rule_id == "conservative_therapy_met"
        assert result.violated_rules[1].rule_id == "objective_findings_met"
        assert "Conservative therapy" in result.violated_rules[0].description

    async def test_fail_without_explicit_violations_gets_generic_violation(
        self, kms_client
    ):
        """FAIL without explicit violation list gets a generic violation entry."""
        opa_evaluator = AsyncMock()
        opa_evaluator.evaluate = AsyncMock(
            return_value={"result": False, "violations": []}
        )
        challenger = OPAChallengerService(kms_client, opa_evaluator)

        result = await challenger.evaluate_policy({"decision": "approve"})

        assert result.passed is False
        assert len(result.violated_rules) == 1
        assert result.violated_rules[0].rule_id == "policy_check_failed"


# =============================================================================
# Evaluate Policy — Infrastructure Failure Tests
# =============================================================================


class TestEvaluatePolicyInfraFailure:
    """Tests for OPA unavailability (Requirement 7.6)."""

    async def test_opa_unavailable_raises_exception(self, kms_client):
        """OPA evaluator failure raises exception."""
        opa_evaluator = AsyncMock()
        opa_evaluator.evaluate = AsyncMock(
            side_effect=RuntimeError("rules.rego loading failure")
        )
        challenger = OPAChallengerService(kms_client, opa_evaluator)

        with pytest.raises(RuntimeError, match="rules.rego loading failure"):
            await challenger.evaluate_policy({"decision": "approve"})

    async def test_opa_timeout_raises_timeout_error(self, kms_client):
        """OPA evaluation exceeding 10s raises TimeoutError."""
        opa_evaluator = AsyncMock()

        async def slow_evaluate(*args, **kwargs):
            await asyncio.sleep(15)
            return {"result": True, "violations": []}

        opa_evaluator.evaluate = slow_evaluate
        challenger = OPAChallengerService(kms_client, opa_evaluator)
        # Override timeout for faster test
        challenger.POLICY_TIMEOUT_SECONDS = 0.1

        with pytest.raises(asyncio.TimeoutError):
            await challenger.evaluate_policy({"decision": "approve"})


# =============================================================================
# Verify Decision — Full Orchestration Tests
# =============================================================================


class TestVerifyDecisionSuccess:
    """Tests for successful verify_decision orchestration."""

    async def test_all_pass_returns_verification_pass(
        self, challenger, evidence_bundle
    ):
        """All signatures valid + all policies pass = VerificationResult.PASS."""
        result = await challenger.verify_decision(evidence_bundle)

        assert isinstance(result, ChallengerResult)
        assert result.verification_result == VerificationResult.PASS
        assert result.signature_result is not None
        assert result.signature_result.all_valid is True
        assert result.policy_result is not None
        assert result.policy_result.passed is True
        assert result.escalation_reason is None
        assert len(result.tamper_alerts) == 0
        assert len(result.violated_rules) == 0

    async def test_pass_result_has_completed_timestamp(
        self, challenger, evidence_bundle
    ):
        """Successful verification has a completed_at timestamp."""
        result = await challenger.verify_decision(evidence_bundle)

        assert result.completed_at is not None
        assert isinstance(result.completed_at, datetime)


# =============================================================================
# Verify Decision — Signature Failure Escalation Tests
# =============================================================================


class TestVerifyDecisionSignatureFailure:
    """Tests for verify_decision with signature failures (Requirement 7.3)."""

    async def test_invalid_signature_produces_fail_with_tamper_alert(
        self, opa_evaluator, evidence_bundle
    ):
        """Invalid signatures produce FAIL and tamper alerts."""
        kms_client = AsyncMock()
        kms_client.verify_signature = AsyncMock(
            side_effect=[True, False]
        )
        challenger = OPAChallengerService(kms_client, opa_evaluator)

        result = await challenger.verify_decision(evidence_bundle)

        assert result.verification_result == VerificationResult.FAIL
        assert len(result.tamper_alerts) == 1
        assert result.escalation_reason is not None
        assert "Tamper alert" in result.escalation_reason
        assert result.escalation_target == "medical_director"

    async def test_signature_failure_skips_policy_evaluation(
        self, opa_evaluator, evidence_bundle
    ):
        """When signatures fail, policy evaluation is not performed."""
        kms_client = AsyncMock()
        kms_client.verify_signature = AsyncMock(return_value=False)
        challenger = OPAChallengerService(kms_client, opa_evaluator)

        result = await challenger.verify_decision(evidence_bundle)

        assert result.verification_result == VerificationResult.FAIL
        assert result.policy_result is None
        opa_evaluator.evaluate.assert_not_called()

    async def test_escalation_identifies_affected_snippets(
        self, opa_evaluator, evidence_bundle
    ):
        """Escalation reason identifies which snippets have invalid signatures."""
        kms_client = AsyncMock()
        kms_client.verify_signature = AsyncMock(
            side_effect=[False, True]
        )
        challenger = OPAChallengerService(kms_client, opa_evaluator)

        result = await challenger.verify_decision(evidence_bundle)

        assert result.verification_result == VerificationResult.FAIL
        assert "key-001" in result.escalation_reason


# =============================================================================
# Verify Decision — Policy Violation Escalation Tests
# =============================================================================


class TestVerifyDecisionPolicyViolation:
    """Tests for verify_decision with OPA policy violations (Requirement 7.4)."""

    async def test_policy_violation_produces_fail_with_rule_ids(
        self, kms_client, evidence_bundle
    ):
        """OPA violations produce FAIL with violated rule IDs."""
        opa_evaluator = AsyncMock()
        opa_evaluator.evaluate = AsyncMock(
            return_value={
                "result": False,
                "violations": [
                    {
                        "rule_id": "conservative_therapy_met",
                        "description": "Therapy duration insufficient",
                    }
                ],
            }
        )
        challenger = OPAChallengerService(kms_client, opa_evaluator)

        result = await challenger.verify_decision(evidence_bundle)

        assert result.verification_result == VerificationResult.FAIL
        assert len(result.violated_rules) == 1
        assert result.violated_rules[0].rule_id == "conservative_therapy_met"
        assert result.escalation_reason is not None
        assert "conservative_therapy_met" in result.escalation_reason
        assert result.escalation_target == "medical_director"

    async def test_policy_violation_includes_signature_result(
        self, kms_client, evidence_bundle
    ):
        """Policy violation result includes successful signature verification."""
        opa_evaluator = AsyncMock()
        opa_evaluator.evaluate = AsyncMock(
            return_value={
                "result": False,
                "violations": [
                    {"rule_id": "rule-x", "description": "Violation X"}
                ],
            }
        )
        challenger = OPAChallengerService(kms_client, opa_evaluator)

        result = await challenger.verify_decision(evidence_bundle)

        assert result.signature_result is not None
        assert result.signature_result.all_valid is True
        assert result.policy_result is not None
        assert result.policy_result.passed is False


# =============================================================================
# Verify Decision — Infrastructure Failure Escalation Tests
# =============================================================================


class TestVerifyDecisionInfraFailure:
    """Tests for verify_decision with infrastructure failures (Requirement 7.6)."""

    async def test_kms_unavailable_returns_fail_with_escalation(
        self, opa_evaluator, evidence_bundle
    ):
        """KMS unavailability produces FAIL with escalation indication."""
        kms_client = AsyncMock()
        kms_client.verify_signature = AsyncMock(
            side_effect=KMSUnavailableError(reason="Service unreachable")
        )
        challenger = OPAChallengerService(kms_client, opa_evaluator)

        result = await challenger.verify_decision(evidence_bundle)

        assert result.verification_result == VerificationResult.FAIL
        assert result.escalation_reason is not None
        assert "could not be completed" in result.escalation_reason
        assert "KMS" in result.escalation_reason

    async def test_opa_unavailable_returns_fail_with_escalation(
        self, kms_client, evidence_bundle
    ):
        """OPA unavailability produces FAIL with escalation indication."""
        opa_evaluator = AsyncMock()
        opa_evaluator.evaluate = AsyncMock(
            side_effect=RuntimeError("rules.rego loading failure")
        )
        challenger = OPAChallengerService(kms_client, opa_evaluator)

        result = await challenger.verify_decision(evidence_bundle)

        assert result.verification_result == VerificationResult.FAIL
        assert result.escalation_reason is not None
        assert "could not be completed" in result.escalation_reason
        assert "rules.rego" in result.escalation_reason

    async def test_kms_failure_does_not_call_opa(
        self, opa_evaluator, evidence_bundle
    ):
        """KMS failure halts pipeline; OPA is never called."""
        kms_client = AsyncMock()
        kms_client.verify_signature = AsyncMock(
            side_effect=KMSUnavailableError(reason="down")
        )
        challenger = OPAChallengerService(kms_client, opa_evaluator)

        await challenger.verify_decision(evidence_bundle)

        opa_evaluator.evaluate.assert_not_called()

    async def test_generic_exception_treated_as_fail(
        self, opa_evaluator, evidence_bundle
    ):
        """Unexpected exceptions during verification treated as FAIL."""
        kms_client = AsyncMock()
        kms_client.verify_signature = AsyncMock(
            side_effect=ConnectionError("Network error")
        )
        challenger = OPAChallengerService(kms_client, opa_evaluator)

        result = await challenger.verify_decision(evidence_bundle)

        assert result.verification_result == VerificationResult.FAIL
        assert result.escalation_reason is not None


# =============================================================================
# Stateless Design Tests (Requirement 7.5)
# =============================================================================


class TestStatelessDesign:
    """Tests that OPAChallengerService has no shared mutable state."""

    async def test_no_instance_state_mutation_across_calls(
        self, challenger, evidence_bundle
    ):
        """Consecutive calls don't affect each other through instance state."""
        result1 = await challenger.verify_decision(evidence_bundle)
        result2 = await challenger.verify_decision(evidence_bundle)

        assert result1.verification_result == result2.verification_result
        assert result1.verification_result == VerificationResult.PASS

    async def test_different_bundles_produce_independent_results(
        self, kms_client, valid_lineage_trail
    ):
        """Different evidence bundles produce independent results."""
        opa_evaluator = AsyncMock()
        # First call passes, second call fails
        opa_evaluator.evaluate = AsyncMock(
            side_effect=[
                {"result": True, "violations": []},
                {
                    "result": False,
                    "violations": [
                        {"rule_id": "r1", "description": "Violation"}
                    ],
                },
            ]
        )
        challenger = OPAChallengerService(kms_client, opa_evaluator)

        bundle1 = EvidenceBundle(
            execution_id="exec-001",
            decision="approve",
            reason="Criteria met",
            lineage_trail=valid_lineage_trail,
            original_document_signatures=[
                KMSSignature(
                    key_id="k1",
                    signature="sig1",
                    signed_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
                )
            ],
        )
        bundle2 = EvidenceBundle(
            execution_id="exec-002",
            decision="approve",
            reason="Criteria met",
            lineage_trail=valid_lineage_trail,
            original_document_signatures=[
                KMSSignature(
                    key_id="k2",
                    signature="sig2",
                    signed_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
                )
            ],
        )

        result1 = await challenger.verify_decision(bundle1)
        result2 = await challenger.verify_decision(bundle2)

        assert result1.verification_result == VerificationResult.PASS
        assert result2.verification_result == VerificationResult.FAIL

    def test_no_shared_mutable_instance_attributes(self, challenger):
        """Service has no mutable collections or caches as instance attributes."""
        # The only instance attributes should be _kms_client and _opa_evaluator
        instance_vars = vars(challenger)
        assert "_kms_client" in instance_vars
        assert "_opa_evaluator" in instance_vars
        # No lists, dicts, or other mutable containers
        for key, value in instance_vars.items():
            if key.startswith("_"):
                continue
            assert not isinstance(value, (list, dict, set)), (
                f"Mutable state found: {key} = {type(value)}"
            )
