"""Axisweave Service API - Versioned REST/gRPC standalone microservice interface.

Exposes document ingestion, hybrid retrieval, and provenance verification
as independent, use-case-agnostic operations with multi-tenant namespace isolation.
"""

from clinical_reasoning_fabric.api.auth_provider import (
    APIAuthProvider,
    APICredentials,
    AuthFailureLog,
)
from clinical_reasoning_fabric.api.axisweave_service_api import (
    AxisweaveServiceAPI,
    ChunkVerificationResult,
    IngestRequest,
    IngestResponse,
    RetrieveRequest,
    RetrieveResponse,
    VerifyRequest,
    VerifyResponse,
)
from clinical_reasoning_fabric.api.namespace import (
    NAMESPACE_PATTERN,
    Namespace,
    NamespaceRegistry,
    validate_namespace,
)

__all__ = [
    "APIAuthProvider",
    "APICredentials",
    "AuthFailureLog",
    "AxisweaveServiceAPI",
    "ChunkVerificationResult",
    "IngestRequest",
    "IngestResponse",
    "NAMESPACE_PATTERN",
    "Namespace",
    "NamespaceRegistry",
    "RetrieveRequest",
    "RetrieveResponse",
    "VerifyRequest",
    "VerifyResponse",
    "validate_namespace",
]
