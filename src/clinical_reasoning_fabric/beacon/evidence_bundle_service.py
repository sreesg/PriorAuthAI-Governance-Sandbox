"""
EvidenceBundleService — Evidence Bundle Producer (BEACON component).

Produces and validates Evidence Bundle output packages containing decision
rationale, cryptographic lineage, and original document signatures.

Requirements referenced: 10.1, 10.2, 10.3, 10.4, 10.5, 8.3

Key behaviors:
- produce_bundle(): produces an Evidence_Bundle conforming to output schema
  with execution_id, decision, reason, lineage_trail, and
  original_document_signatures.
- Validates all required schema fields are present, non-null, correct type.
- lineage_trail must have >= 1 entry, each with conclusion, evidence_id,
  retrieval_timestamp.
- original_document_signatures must have >= 1 signature.
- Attaches complete execution trace to bundle.
- On validation failure: raises BundleValidationError with missing/invalid
  fields, halts decision, logs bundle integrity error, escalates to MD.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional, Protocol, runtime_checkable

from clinical_reasoning_fabric.models.core import (
    EvidenceBundle,
    KMSSignature,
    LineageEntry,
    TraceEntry,
)
from clinical_reasoning_fabric.models.exceptions import BundleValidationError


# =============================================================================
# Constants
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# Schema Validator Protocol
# =============================================================================


@runtime_checkable
class SchemaValidator(Protocol):
    """Protocol for schema validation of Evidence Bundle fields.

    Implementations can provide additional domain-specific validation
    beyond the structural checks performed by EvidenceBundleService.
    """

    def validate_bundle(self, bundle: EvidenceBundle) -> list[str]:
        """Validate a bundle and return a list of validation error messages.

        Returns an empty list if validation passes, or a list of error
        descriptions if validation fails.
        """
        ...


# =============================================================================
# Default Schema Validator
# =============================================================================


class DefaultSchemaValidator:
    """Default schema validator performing structural checks on EvidenceBundle.

    Validates that all Pydantic model constraints are satisfied beyond
    the basic field-level checks done in produce_bundle().
    """

    def validate_bundle(self, bundle: EvidenceBundle) -> list[str]:
        """Validate bundle conforms to the output schema.

        Returns empty list on success, list of error descriptions on failure.
        """
        errors: list[str] = []

        # Validate each lineage entry has proper content
        for i, entry in enumerate(bundle.lineage_trail):
            if not entry.conclusion or not entry.conclusion.strip():
                errors.append(
                    f"lineage_trail[{i}].conclusion is empty or whitespace"
                )
            if not entry.evidence_id or not entry.evidence_id.strip():
                errors.append(
                    f"lineage_trail[{i}].evidence_id is empty or whitespace"
                )

        # Validate each signature has proper content
        for i, sig in enumerate(bundle.original_document_signatures):
            if not sig.key_id or not sig.key_id.strip():
                errors.append(
                    f"original_document_signatures[{i}].key_id is empty or whitespace"
                )
            if not sig.signature or not sig.signature.strip():
                errors.append(
                    f"original_document_signatures[{i}].signature is empty or whitespace"
                )

        return errors


# =============================================================================
# EvidenceBundleService
# =============================================================================


class EvidenceBundleService:
    """Produces and validates Evidence Bundle output packages.

    Requirement 10.1: Produce Evidence_Bundle conforming to defined output
    schema containing execution_id, decision, reason, lineage_trail, and
    original_document_signatures.

    Requirement 10.2: lineage_trail as ordered array of entries with
    conclusion, evidence_id, and retrieval_timestamp.

    Requirement 10.3: original_document_signatures array with KMS signature
    for every source document referenced.

    Requirement 10.4: Validate all required schema fields present, non-null,
    correct data type; lineage_trail >= 1 entry; signatures >= 1 signature.

    Requirement 10.5: On validation failure, halt decision, log bundle
    integrity error with missing/invalid fields, escalate to MD.

    Requirement 8.3: Attach execution trace to Evidence_Bundle.
    """

    def __init__(self, schema_validator: SchemaValidator | None = None) -> None:
        """Initialize EvidenceBundleService with an optional schema validator.

        Args:
            schema_validator: Optional custom schema validator. If None,
                uses DefaultSchemaValidator.
        """
        self._validator = schema_validator or DefaultSchemaValidator()

    @property
    def validator(self) -> SchemaValidator:
        """Access the schema validator."""
        return self._validator

    def produce_bundle(
        self,
        execution_id: str,
        decision: str,
        reason: str,
        lineage_trail: list[LineageEntry],
        document_signatures: list[KMSSignature],
        execution_trace: list[TraceEntry] | None = None,
    ) -> EvidenceBundle:
        """Produce an Evidence Bundle conforming to output schema.

        Validates all required fields are present, non-null, and of correct
        type. lineage_trail must have >= 1 entry, document_signatures must
        have >= 1 signature.

        Args:
            execution_id: Unique identifier for this execution.
            decision: The PA decision (approve/escalate).
            reason: Reasoning for the decision.
            lineage_trail: Ordered entries with conclusion, evidence_id,
                and retrieval_timestamp.
            document_signatures: KMS signatures for every referenced source
                document.
            execution_trace: Optional complete execution trace to attach.

        Returns:
            A validated EvidenceBundle instance.

        Raises:
            BundleValidationError: If schema validation fails. Contains
                lists of missing_fields and invalid_fields for audit logging
                and Medical Director escalation.
        """
        missing_fields: list[str] = []
        invalid_fields: list[str] = []

        # --- Validate execution_id ---
        if execution_id is None:
            missing_fields.append("execution_id")
        elif not isinstance(execution_id, str):
            invalid_fields.append("execution_id (must be a string)")
        elif not execution_id.strip():
            invalid_fields.append("execution_id (must be non-empty)")

        # --- Validate decision ---
        if decision is None:
            missing_fields.append("decision")
        elif not isinstance(decision, str):
            invalid_fields.append("decision (must be a string)")
        elif not decision.strip():
            invalid_fields.append("decision (must be non-empty)")

        # --- Validate reason ---
        if reason is None:
            missing_fields.append("reason")
        elif not isinstance(reason, str):
            invalid_fields.append("reason (must be a string)")
        elif not reason.strip():
            invalid_fields.append("reason (must be non-empty)")

        # --- Validate lineage_trail ---
        if lineage_trail is None:
            missing_fields.append("lineage_trail")
        elif not isinstance(lineage_trail, list):
            invalid_fields.append("lineage_trail (must be a list)")
        elif len(lineage_trail) < 1:
            invalid_fields.append("lineage_trail (must have >= 1 entry)")
        else:
            # Validate each lineage entry
            for i, entry in enumerate(lineage_trail):
                if not isinstance(entry, LineageEntry):
                    invalid_fields.append(
                        f"lineage_trail[{i}] (must be a LineageEntry)"
                    )
                    continue
                if not entry.conclusion or not entry.conclusion.strip():
                    invalid_fields.append(
                        f"lineage_trail[{i}].conclusion (must be non-empty)"
                    )
                if not entry.evidence_id or not entry.evidence_id.strip():
                    invalid_fields.append(
                        f"lineage_trail[{i}].evidence_id (must be non-empty)"
                    )
                if entry.retrieval_timestamp is None:
                    invalid_fields.append(
                        f"lineage_trail[{i}].retrieval_timestamp (must not be null)"
                    )
                elif not isinstance(entry.retrieval_timestamp, datetime):
                    invalid_fields.append(
                        f"lineage_trail[{i}].retrieval_timestamp (must be a datetime)"
                    )

        # --- Validate document_signatures ---
        if document_signatures is None:
            missing_fields.append("original_document_signatures")
        elif not isinstance(document_signatures, list):
            invalid_fields.append("original_document_signatures (must be a list)")
        elif len(document_signatures) < 1:
            invalid_fields.append(
                "original_document_signatures (must have >= 1 signature)"
            )
        else:
            # Validate each signature
            for i, sig in enumerate(document_signatures):
                if not isinstance(sig, KMSSignature):
                    invalid_fields.append(
                        f"original_document_signatures[{i}] (must be a KMSSignature)"
                    )

        # --- If any validation errors, halt and raise ---
        if missing_fields or invalid_fields:
            error_reason = self._build_error_reason(missing_fields, invalid_fields)
            logger.error(
                "Bundle integrity error: %s | missing_fields=%s | invalid_fields=%s",
                error_reason,
                missing_fields,
                invalid_fields,
            )
            raise BundleValidationError(
                reason=error_reason,
                missing_fields=missing_fields if missing_fields else None,
                invalid_fields=invalid_fields if invalid_fields else None,
            )

        # --- Construct the EvidenceBundle ---
        bundle = EvidenceBundle(
            execution_id=execution_id,
            decision=decision,
            reason=reason,
            lineage_trail=lineage_trail,
            original_document_signatures=document_signatures,
            execution_trace=execution_trace,
        )

        # --- Run additional schema validation ---
        additional_errors = self._validator.validate_bundle(bundle)
        if additional_errors:
            logger.error(
                "Bundle schema validation failed: %s", additional_errors
            )
            raise BundleValidationError(
                reason=f"Schema validation failed: {'; '.join(additional_errors)}",
                missing_fields=None,
                invalid_fields=additional_errors,
            )

        return bundle

    @staticmethod
    def _build_error_reason(
        missing_fields: list[str], invalid_fields: list[str]
    ) -> str:
        """Build a human-readable error reason from validation failures."""
        parts: list[str] = []
        if missing_fields:
            parts.append(f"Missing fields: {', '.join(missing_fields)}")
        if invalid_fields:
            parts.append(f"Invalid fields: {', '.join(invalid_fields)}")
        return "Evidence Bundle validation failed. " + "; ".join(parts)
