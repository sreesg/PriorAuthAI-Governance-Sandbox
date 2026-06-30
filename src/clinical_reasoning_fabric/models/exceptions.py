"""Custom exception hierarchy for the Clinical Reasoning Fabric.

All exceptions inherit from CRFError and capture structured context
(document_id, identity, timestamp, reason) for audit logging and
structured serialization.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Optional


@dataclass
class CRFError(Exception):
    """Base exception for all Clinical Reasoning Fabric errors.

    Captures structured context for audit logging and serialization.
    The timestamp is auto-populated as UTC ISO-8601 if not provided.
    """

    reason: str
    document_id: Optional[str] = None
    identity: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    details: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        # Initialize Exception with the reason message
        super().__init__(self.reason)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the exception to a dictionary for structured audit logging."""
        data = asdict(self)
        data["error_type"] = type(self).__name__
        return data

    def __str__(self) -> str:
        parts = [f"{type(self).__name__}: {self.reason}"]
        if self.document_id:
            parts.append(f"document_id={self.document_id}")
        if self.identity:
            parts.append(f"identity={self.identity}")
        return " | ".join(parts)


@dataclass
class IngestionError(CRFError):
    """Raised when document parsing fails due to corrupted or unsupported format.

    Requirement 1.6: If document parsing fails, reject the document with an error
    indicating the failure reason and log the failure to the audit trail.
    """

    pass


@dataclass
class PIIScrubError(CRFError):
    """Raised when PII scrubbing fails or cannot complete.

    Requirement 1.8: If PII scrubbing fails, halt document processing,
    prevent the document from being stored, and log the scrubbing failure.
    """

    pass


@dataclass
class KMSUnavailableError(CRFError):
    """Raised when KMS signing fails or the KMS service is unavailable.

    Requirement 1.7: If the KMS_Signer is unavailable or signing fails,
    halt ingestion for that document, discard any unsigned chunks, and
    log the signing failure to the audit trail.
    """

    pass


@dataclass
class UnauthorizedError(CRFError):
    """Raised when an identity lacks permissions for a requested operation.

    Requirement 5.3: Deny the request without exposing any clinical data
    in the response. Log unauthorized access attempt with requesting identity,
    requested operation, timestamp, and missing permission.

    IMPORTANT: This error must never contain clinical data in its fields.
    """

    operation: Optional[str] = None
    missing_permission: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize without exposing clinical data."""
        data = super().to_dict()
        if self.operation:
            data["operation"] = self.operation
        if self.missing_permission:
            data["missing_permission"] = self.missing_permission
        return data


@dataclass
class TraceRecordingError(CRFError):
    """Raised when trace recording fails, halting PA processing.

    Requirement 8.5: If trace recording fails, halt the PA request processing
    and return an error indicating trace failure rather than produce an
    unaudited decision.
    """

    request_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        if self.request_id:
            data["request_id"] = self.request_id
        return data


@dataclass
class MemberNotFoundError(CRFError):
    """Raised when a member is not found in the Causal Ontology Graph.

    Requirement 4.5 (implied by 5.3 context): If the member is not found
    in the Causal_Ontology_Graph, return an error indicating the member
    state is unavailable and halt processing.
    """

    member_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        if self.member_id:
            data["member_id"] = self.member_id
        return data


@dataclass
class BundleValidationError(CRFError):
    """Raised when an Evidence Bundle fails schema validation.

    Requirement 10.5: If the Evidence_Bundle fails schema validation, halt
    the decision, log a bundle integrity error identifying the missing or
    invalid fields, and escalate to Medical Director.
    """

    missing_fields: Optional[list[str]] = None
    invalid_fields: Optional[list[str]] = None

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        if self.missing_fields:
            data["missing_fields"] = self.missing_fields
        if self.invalid_fields:
            data["invalid_fields"] = self.invalid_fields
        return data


@dataclass
class UnmappableRecordError(CRFError):
    """Raised when a CDC source record cannot be mapped to a graph type.

    Requirement 12.6: If a source record cannot be mapped to a valid graph
    node or relationship type, skip the record, log the failure with the
    source record identifier and reason, and continue processing.
    """

    source_record_id: Optional[str] = None
    entity_type: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        if self.source_record_id:
            data["source_record_id"] = self.source_record_id
        if self.entity_type:
            data["entity_type"] = self.entity_type
        return data


@dataclass
class ToolValidationError(CRFError):
    """Raised when a tool request fails validation in the MCP Gateway.

    Requirement 6.3 (implied by 13.8 context): If the agent requests a tool
    not present in the approved catalog or with invalid parameters, reject
    the invocation and log the unauthorized tool request.
    """

    tool_name: Optional[str] = None
    validation_errors: Optional[list[str]] = None

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        if self.tool_name:
            data["tool_name"] = self.tool_name
        if self.validation_errors:
            data["validation_errors"] = self.validation_errors
        return data


@dataclass
class InvalidNamespaceError(CRFError):
    """Raised when a namespace format doesn't match the required pattern.

    Requirement 13.8: If a namespace identifier does not conform to the
    format of 1-128 alphanumeric, hyphen, or underscore characters, reject
    the request with an error and log the rejection.

    Valid pattern: ^[a-zA-Z0-9_-]{1,128}$
    """

    namespace: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        if self.namespace:
            data["namespace"] = self.namespace
        return data


@dataclass
class InferenceTimeoutError(CRFError):
    """Raised when inference exceeds the 15-second timeout per snippet.

    Requirement 14.1 (implied by 14.9 context): The Clinical_Inference_Engine
    SHALL analyze each retrieved snippet within 15 seconds per snippet.
    """

    snippet_id: Optional[str] = None
    timeout_seconds: float = 15.0

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        if self.snippet_id:
            data["snippet_id"] = self.snippet_id
        data["timeout_seconds"] = self.timeout_seconds
        return data
