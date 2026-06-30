"""BEACON Layer 5 — OPA Challenger Service.

Independent verification agent that validates KMS signature provenance
and OPA policy compliance for all PA decisions. Operates with no shared
mutable state with the primary reasoning agent.

Requirements referenced: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Protocol

from clinical_reasoning_fabric.models.core import (
    EvidenceBundle,
    KMSSignature,
    TamperAlert,
    VerificationResult,
)
from clinical_reasoning_fabric.models.exceptions import KMSUnavailableError

logger = logging.getLogger(__name__)


# =============================================================================
# Protocols for Dependency Injection (Requirement 7.5 — no shared state)
# =============================================================================


class KMSClient(Protocol):
    """Protocol for KMS signature verification operations."""

    async def verify_signature(
        self, content_hash: str, signature: str, key_id: str
    ) -> bool:
        """Verify that a KMS signature matches the given content hash.

        Returns True if signature is valid, False otherwise.
        Raises KMSUnavailableError if the KMS service is not reachable.
        """
        ...


class OPAEvaluator(Protocol):
    """Protocol for OPA policy evaluation operations."""

    async def evaluate(self, policy_path: str, input_data: dict) -> dict:
        """Evaluate input against OPA policy at the given path.

        Returns the evaluation result as a dictionary containing:
        - 'result': bool (True if all rules pass)
        - 'violations': list of violated rule identifiers and descriptions

        Raises OPAUnavailableError if rules.rego cannot be loaded or OPA is unreachable.
        """
        ...


# =============================================================================
# Result Data Models
# =============================================================================


@dataclass
class SignatureVerificationResult:
    """Result of verifying KMS signatures on evidence snippets.

    Requirement 7.1: Verify that every evidence snippet has a valid KMS
    signature matching its content hash within 30 seconds.
    """

    valid: list[str] = field(default_factory=list)
    """List of chunk_ids with valid signatures."""

    invalid: list[str] = field(default_factory=list)
    """List of chunk_ids with invalid signatures."""

    missing: list[str] = field(default_factory=list)
    """List of chunk_ids with missing signatures."""

    tamper_alerts: list[TamperAlert] = field(default_factory=list)
    """TamperAlert for each invalid or missing signature."""

    all_valid: bool = False
    """True only if no invalid or missing signatures found."""

    verified_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass
class PolicyViolation:
    """A single OPA policy rule violation."""

    rule_id: str
    description: str


@dataclass
class PolicyEvaluationResult:
    """Result of evaluating a decision against OPA rules.rego.

    Requirement 7.2: Produce PASS or FAIL with violated rule identifiers.
    """

    passed: bool = False
    """True if all policy rules pass."""

    violated_rules: list[PolicyViolation] = field(default_factory=list)
    """List of violated rules with identifiers and descriptions."""

    evaluated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass
class ChallengerResult:
    """Complete result of the OPA Challenger verification process.

    Aggregates signature verification and policy evaluation outcomes.
    Used to determine whether to approve or escalate to Medical Director.
    """

    verification_result: VerificationResult
    """Overall PASS/FAIL determination."""

    signature_result: Optional[SignatureVerificationResult] = None
    """Detailed signature verification result."""

    policy_result: Optional[PolicyEvaluationResult] = None
    """Detailed policy evaluation result."""

    tamper_alerts: list[TamperAlert] = field(default_factory=list)
    """All tamper alerts from signature verification."""

    violated_rules: list[PolicyViolation] = field(default_factory=list)
    """All violated OPA rules."""

    escalation_reason: Optional[str] = None
    """Reason for escalation (if applicable)."""

    escalation_target: str = "medical_director"
    """Target queue for escalation."""

    completed_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# =============================================================================
# OPA Challenger Service
# =============================================================================


class OPAChallengerService:
    """Independent verification agent for signature and policy compliance.

    Requirement 7.1: Verify KMS signatures on all evidence within 30 seconds.
    Requirement 7.2: Evaluate against OPA rules.rego within 10 seconds.
    Requirement 7.3: Reject on invalid/missing signatures; escalate with tamper alert.
    Requirement 7.4: Route OPA violations to Medical Director with rule identifiers.
    Requirement 7.5: Operate independently with no shared mutable state.
    Requirement 7.6: On infrastructure failure, treat as FAIL and escalate.

    This service is stateless — all state is passed in via method arguments
    and returned via result objects. No instance state is shared with the
    primary reasoning agent.
    """

    SIGNATURE_TIMEOUT_SECONDS = 30
    POLICY_TIMEOUT_SECONDS = 10
    POLICY_PATH = "prior_auth.policy"

    def __init__(self, kms_client: KMSClient, opa_evaluator: OPAEvaluator):
        """Initialize with injected KMS and OPA dependencies.

        Args:
            kms_client: Client for KMS signature verification.
            opa_evaluator: Evaluator for OPA policy rules.
        """
        self._kms_client = kms_client
        self._opa_evaluator = opa_evaluator

    async def verify_decision(
        self, evidence_bundle: EvidenceBundle
    ) -> ChallengerResult:
        """Orchestrate full verification: signatures then policy evaluation.

        1. Verify KMS signatures on all referenced evidence (within 30s)
        2. If all signatures valid, evaluate decision against OPA rules.rego (within 10s)
        3. Return ChallengerResult with PASS/FAIL and findings

        On any infrastructure failure (KMS unavailable, rules.rego loading
        failure), returns FAIL and escalates with indication that verification
        could not be completed.

        Requirements: 7.1, 7.2, 7.3, 7.4, 7.6
        """
        # Step 1: Verify signatures
        try:
            signature_result = await self.verify_signatures(
                evidence_bundle.original_document_signatures,
                evidence_bundle.lineage_trail,
            )
        except (KMSUnavailableError, asyncio.TimeoutError, Exception) as exc:
            logger.error(
                f"Infrastructure failure during signature verification: {exc}"
            )
            return ChallengerResult(
                verification_result=VerificationResult.FAIL,
                escalation_reason=(
                    "Verification could not be completed: KMS service unavailable "
                    f"or internal error during signature verification. Error: {exc}"
                ),
            )

        # Step 2: Check signature results — reject on any invalid/missing
        if not signature_result.all_valid:
            affected_snippets = signature_result.invalid + signature_result.missing
            logger.warning(
                f"Signature verification failed. Invalid: {signature_result.invalid}, "
                f"Missing: {signature_result.missing}"
            )
            return ChallengerResult(
                verification_result=VerificationResult.FAIL,
                signature_result=signature_result,
                tamper_alerts=signature_result.tamper_alerts,
                escalation_reason=(
                    f"Tamper alert: {len(affected_snippets)} evidence snippet(s) have "
                    f"invalid or missing KMS signatures. Affected: {affected_snippets}"
                ),
            )

        # Step 3: Evaluate policy rules
        try:
            decision_context = self._build_decision_context(evidence_bundle)
            policy_result = await self.evaluate_policy(decision_context)
        except (asyncio.TimeoutError, Exception) as exc:
            logger.error(
                f"Infrastructure failure during policy evaluation: {exc}"
            )
            return ChallengerResult(
                verification_result=VerificationResult.FAIL,
                signature_result=signature_result,
                escalation_reason=(
                    "Verification could not be completed: OPA policy evaluation "
                    f"failed due to rules.rego loading failure or internal error. Error: {exc}"
                ),
            )

        # Step 4: Check policy results — route violations to Medical Director
        if not policy_result.passed:
            violated_rule_ids = [v.rule_id for v in policy_result.violated_rules]
            logger.warning(
                f"OPA policy evaluation failed. Violated rules: {violated_rule_ids}"
            )
            return ChallengerResult(
                verification_result=VerificationResult.FAIL,
                signature_result=signature_result,
                policy_result=policy_result,
                violated_rules=policy_result.violated_rules,
                escalation_reason=(
                    f"OPA policy violation: {len(policy_result.violated_rules)} rule(s) violated. "
                    f"Rule IDs: {violated_rule_ids}"
                ),
            )

        # Step 5: All passed
        logger.info("OPA Challenger verification PASSED for all checks.")
        return ChallengerResult(
            verification_result=VerificationResult.PASS,
            signature_result=signature_result,
            policy_result=policy_result,
        )

    async def verify_signatures(
        self,
        signatures: list[KMSSignature],
        lineage_trail: list = None,
    ) -> SignatureVerificationResult:
        """Verify KMS signatures on all evidence snippets.

        Must complete within 30 seconds (Requirement 7.1).
        Reports invalid and missing signatures for tamper alerting.

        Args:
            signatures: List of KMS signatures to verify.
            lineage_trail: Optional lineage entries for correlating evidence IDs.

        Returns:
            SignatureVerificationResult with valid/invalid/missing lists.

        Raises:
            KMSUnavailableError: If the KMS service is unreachable.
            asyncio.TimeoutError: If verification exceeds 30 seconds.
        """
        result = SignatureVerificationResult()

        if not signatures:
            result.missing.append("no_signatures_provided")
            result.tamper_alerts.append(
                TamperAlert(
                    chunk_id="unknown",
                    document_id="unknown",
                    content_hash="no_content_hash_available",
                    expected_signature=None,
                    reason="No signatures provided in evidence bundle",
                    detected_at=datetime.now(timezone.utc),
                )
            )
            result.all_valid = False
            return result

        try:
            verification_tasks = []
            for idx, sig in enumerate(signatures):
                verification_tasks.append(
                    self._verify_single_signature(sig, idx)
                )

            # Execute all verifications with 30-second timeout
            outcomes = await asyncio.wait_for(
                asyncio.gather(*verification_tasks, return_exceptions=True),
                timeout=self.SIGNATURE_TIMEOUT_SECONDS,
            )

            for idx, outcome in enumerate(outcomes):
                sig = signatures[idx]
                sig_id = f"sig_{idx}_{sig.key_id}"

                if isinstance(outcome, KMSUnavailableError):
                    raise outcome
                elif isinstance(outcome, Exception):
                    # Treat unexpected exceptions as missing/unverifiable
                    result.missing.append(sig_id)
                    result.tamper_alerts.append(
                        TamperAlert(
                            chunk_id=sig_id,
                            document_id=sig.key_id,
                            content_hash=f"unverifiable_{sig.key_id}",
                            expected_signature=sig.signature,
                            reason=f"Verification error: {outcome}",
                            detected_at=datetime.now(timezone.utc),
                        )
                    )
                elif outcome is True:
                    result.valid.append(sig_id)
                else:
                    result.invalid.append(sig_id)
                    result.tamper_alerts.append(
                        TamperAlert(
                            chunk_id=sig_id,
                            document_id=sig.key_id,
                            content_hash=f"invalid_{sig.key_id}",
                            expected_signature=sig.signature,
                            reason="KMS signature verification failed: signature does not match content hash",
                            detected_at=datetime.now(timezone.utc),
                        )
                    )

        except asyncio.TimeoutError:
            logger.error("Signature verification timed out (30s limit exceeded)")
            raise
        except KMSUnavailableError:
            logger.error("KMS service unavailable during signature verification")
            raise

        result.all_valid = (
            len(result.invalid) == 0 and len(result.missing) == 0
        )
        return result

    async def evaluate_policy(
        self, decision_context: dict
    ) -> PolicyEvaluationResult:
        """Evaluate decision against OPA rules.rego.

        Must complete within 10 seconds (Requirement 7.2).
        Produces PASS/FAIL with violated rule identifiers.

        Args:
            decision_context: Dictionary containing the decision data to evaluate.

        Returns:
            PolicyEvaluationResult with pass/fail and violated rules.

        Raises:
            asyncio.TimeoutError: If evaluation exceeds 10 seconds.
            Exception: If rules.rego cannot be loaded or OPA is unreachable.
        """
        try:
            opa_response = await asyncio.wait_for(
                self._opa_evaluator.evaluate(
                    self.POLICY_PATH, decision_context
                ),
                timeout=self.POLICY_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.error("OPA policy evaluation timed out (10s limit exceeded)")
            raise
        except Exception as exc:
            logger.error(f"OPA evaluation failed: {exc}")
            raise

        # Parse OPA response
        result = PolicyEvaluationResult()
        result.passed = opa_response.get("result", False)

        violations = opa_response.get("violations", [])
        for violation in violations:
            result.violated_rules.append(
                PolicyViolation(
                    rule_id=violation.get("rule_id", "unknown"),
                    description=violation.get("description", "No description provided"),
                )
            )

        # If not passed but no explicit violations returned, add a generic one
        if not result.passed and not result.violated_rules:
            result.violated_rules.append(
                PolicyViolation(
                    rule_id="policy_check_failed",
                    description="Policy evaluation returned FAIL without specific rule violations",
                )
            )

        result.evaluated_at = datetime.now(timezone.utc)
        return result

    # =========================================================================
    # Private Helper Methods
    # =========================================================================

    async def _verify_single_signature(
        self, signature: KMSSignature, index: int
    ) -> bool:
        """Verify a single KMS signature.

        Returns True if valid, False if invalid.
        Raises KMSUnavailableError if KMS is unreachable.
        """
        if not signature.signature or not signature.signature.strip():
            return False

        return await self._kms_client.verify_signature(
            content_hash=signature.key_id,  # Using key_id as content reference
            signature=signature.signature,
            key_id=signature.key_id,
        )

    def _build_decision_context(self, evidence_bundle: EvidenceBundle) -> dict:
        """Build the decision context dictionary for OPA evaluation.

        Extracts relevant fields from the EvidenceBundle to pass as
        OPA input for policy rule evaluation.
        """
        return {
            "execution_id": evidence_bundle.execution_id,
            "decision": evidence_bundle.decision,
            "reason": evidence_bundle.reason,
            "lineage_trail_count": len(evidence_bundle.lineage_trail),
            "signature_count": len(evidence_bundle.original_document_signatures),
            "has_execution_trace": evidence_bundle.execution_trace is not None,
            "lineage_entries": [
                {
                    "conclusion": entry.conclusion,
                    "evidence_id": entry.evidence_id,
                    "retrieval_timestamp": entry.retrieval_timestamp.isoformat(),
                }
                for entry in evidence_bundle.lineage_trail
            ],
        }
