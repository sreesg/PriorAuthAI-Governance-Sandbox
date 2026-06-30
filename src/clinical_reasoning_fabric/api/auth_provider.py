"""API authentication and tenant isolation for Axisweave Service API.

Provides API key validation, tenant_id extraction, and namespace access control
for multi-tenant isolation. On invalid/missing credentials, rejects requests
without exposing any stored document data and logs authentication failures.

Requirements referenced: 13.5, 13.9
- 13.5: Enforce tenant and namespace isolation; cross-namespace requires explicit grant.
- 13.9: Reject requests with invalid/missing credentials without exposing document data;
         log authentication failure with timestamp and source identifier.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from clinical_reasoning_fabric.api.namespace import NamespaceRegistry
from clinical_reasoning_fabric.models.exceptions import UnauthorizedError

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class APICredentials:
    """Credentials extracted from a validated API key.

    Attributes:
        api_key: The raw API key string provided by the caller.
        tenant_id: The tenant identifier associated with this API key.
        authorized_namespaces: List of namespace IDs this key is authorized to access.
    """

    api_key: str
    tenant_id: str
    authorized_namespaces: list[str] = field(default_factory=list)


@dataclass
class AuthFailureLog:
    """Structured log entry for authentication failures.

    Captures timestamp and source identifier as required by Requirement 13.9.
    """

    timestamp: str
    source_identifier: str
    reason: str
    api_key_prefix: Optional[str] = None


# =============================================================================
# API Auth Provider
# =============================================================================


class APIAuthProvider:
    """Authenticates API requests and enforces tenant/namespace isolation.

    Validates API keys, extracts tenant_id, and controls namespace access.
    Cross-namespace access requires an explicit grant in the API key scope.

    On authentication failure:
    - Rejects the request without exposing any stored document data
    - Logs the failure with timestamp and source identifier

    Requirement 13.5: Enforce tenant and namespace isolation.
    Requirement 13.9: Reject on invalid/missing credentials; log failure.
    """

    def __init__(
        self,
        namespace_registry: NamespaceRegistry,
        api_key_store: Optional[dict[str, APICredentials]] = None,
    ):
        """Initialize the auth provider.

        Args:
            namespace_registry: Registry for namespace ownership and grants.
            api_key_store: Mapping of API key strings to their associated credentials.
                In production this would be backed by a secure key vault or database.
                If None, an empty store is used.
        """
        self._namespace_registry = namespace_registry
        self._api_key_store: dict[str, APICredentials] = api_key_store or {}

    def register_api_key(
        self,
        api_key: str,
        tenant_id: str,
        authorized_namespaces: Optional[list[str]] = None,
    ) -> APICredentials:
        """Register a new API key with associated tenant and namespace authorizations.

        Args:
            api_key: The API key string to register.
            tenant_id: The tenant this key belongs to.
            authorized_namespaces: Explicit list of namespaces this key can access.
                If None, defaults to the tenant's own namespaces (determined at access time).

        Returns:
            The registered APICredentials instance.

        Raises:
            ValueError: If api_key is empty or already registered.
        """
        if not api_key or not api_key.strip():
            raise ValueError("API key must not be empty")

        if api_key in self._api_key_store:
            raise ValueError("API key is already registered")

        credentials = APICredentials(
            api_key=api_key,
            tenant_id=tenant_id,
            authorized_namespaces=authorized_namespaces or [],
        )
        self._api_key_store[api_key] = credentials
        return credentials

    def validate_credentials(self, api_key: str) -> APICredentials:
        """Validate an API key and extract associated tenant_id and authorized namespaces.

        Looks up the API key in the key store. On success, returns the associated
        APICredentials with tenant_id and authorized namespace scope.

        On failure (missing, empty, or invalid key):
        - Raises UnauthorizedError with no document data exposed
        - Logs authentication failure with timestamp and source identifier

        Args:
            api_key: The API key string to validate.

        Returns:
            APICredentials with tenant_id and authorized_namespaces.

        Raises:
            UnauthorizedError: If the API key is missing, empty, or not recognized.
        """
        # Check for missing/empty credentials
        if not api_key or not api_key.strip():
            self._log_auth_failure(
                source_identifier="unknown",
                reason="Missing or empty API key",
                api_key_prefix=None,
            )
            raise UnauthorizedError(
                reason="Authentication failed: missing or empty API key",
                identity=None,
                operation="api_access",
            )

        # Attempt to look up the API key
        credentials = self._api_key_store.get(api_key)

        if credentials is None:
            # Key not recognized — extract safe prefix for logging
            safe_prefix = api_key[:8] + "..." if len(api_key) > 8 else "***"
            self._log_auth_failure(
                source_identifier="unknown",
                reason="Invalid API key",
                api_key_prefix=safe_prefix,
            )
            raise UnauthorizedError(
                reason="Authentication failed: invalid API key",
                identity=None,
                operation="api_access",
            )

        return credentials

    def check_namespace_access(
        self, credentials: APICredentials, target_namespace: str
    ) -> bool:
        """Verify that the caller can access the target namespace.

        Access rules:
        1. If target_namespace is in the credentials' authorized_namespaces list,
           access is granted (explicit scope in token/API key).
        2. Otherwise, check the NamespaceRegistry to see if the tenant owns
           the namespace or has been granted cross-namespace access.
        3. If neither condition is met, deny access.

        Args:
            credentials: The authenticated caller's credentials.
            target_namespace: The namespace the caller wants to access.

        Returns:
            True if the caller has access to the target namespace.

        Raises:
            UnauthorizedError: If the caller does not have access to the namespace.
        """
        # Check 1: Explicit authorization in the API key scope
        if target_namespace in credentials.authorized_namespaces:
            return True

        # Check 2: Namespace registry ownership or cross-namespace grant
        if self._namespace_registry.has_access(credentials.tenant_id, target_namespace):
            return True

        # Access denied — log and raise
        self._log_auth_failure(
            source_identifier=credentials.tenant_id,
            reason=f"Namespace access denied: tenant '{credentials.tenant_id}' "
                   f"cannot access namespace '{target_namespace}'",
            api_key_prefix=credentials.api_key[:8] + "..."
            if len(credentials.api_key) > 8
            else "***",
        )
        raise UnauthorizedError(
            reason=f"Access denied: not authorized for namespace '{target_namespace}'",
            identity=credentials.tenant_id,
            operation="namespace_access",
        )

    def _log_auth_failure(
        self,
        source_identifier: str,
        reason: str,
        api_key_prefix: Optional[str] = None,
    ) -> None:
        """Log an authentication/authorization failure.

        Requirement 13.9: Log authentication failure including timestamp
        and source identifier to the audit trail.

        Args:
            source_identifier: Identifies the source of the failed request.
            reason: Human-readable reason for the failure.
            api_key_prefix: Safe prefix of the API key (for debugging, not full key).
        """
        failure_log = AuthFailureLog(
            timestamp=datetime.now(timezone.utc).isoformat(),
            source_identifier=source_identifier,
            reason=reason,
            api_key_prefix=api_key_prefix,
        )

        logger.warning(
            "Authentication failure | "
            f"timestamp={failure_log.timestamp} | "
            f"source={failure_log.source_identifier} | "
            f"reason={failure_log.reason}"
            + (f" | key_prefix={failure_log.api_key_prefix}" if failure_log.api_key_prefix else "")
        )
