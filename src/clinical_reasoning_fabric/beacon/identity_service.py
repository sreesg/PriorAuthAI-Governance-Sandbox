"""BEACON Layer 1 — Identity Service.

Handles authentication, RBAC enforcement, PII/PHI masking, and trace context
creation. Every agent action is associated with an authenticated identity.

Requirements referenced: 5.1, 5.2, 5.3, 5.4, 5.5
"""

from __future__ import annotations

import hashlib
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from clinical_reasoning_fabric.models.core import AuthResult, RBACPolicy
from clinical_reasoning_fabric.models.exceptions import UnauthorizedError

logger = logging.getLogger(__name__)


# =============================================================================
# Supporting Data Models
# =============================================================================


@dataclass
class Credentials:
    """Credentials submitted by a requesting identity.

    Supports API key or token-based authentication.
    """

    identity_id: str
    api_key: Optional[str] = None
    token: Optional[str] = None


@dataclass
class TraceContext:
    """Trace context associating every action with an authenticated identity.

    Requirement 5.4: Every agent action is attributable to a specific identity.
    """

    identity_id: str
    request_id: str
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class MaskingService:
    """Service for irreversible PHI/PII masking using SHA-256 hashing.

    Requirement 5.1: Replace all PII/PHI fields with irreversible masked tokens.
    """

    # PHI field name patterns that trigger masking
    phi_field_patterns: list[str] = field(default_factory=lambda: [
        r".*name.*",
        r".*first_name.*",
        r".*last_name.*",
        r".*patient_name.*",
        r".*ssn.*",
        r".*social_security.*",
        r".*dob.*",
        r".*date_of_birth.*",
        r".*birth_date.*",
        r".*mrn.*",
        r".*medical_record.*",
        r".*address.*",
        r".*street.*",
        r".*city.*",
        r".*zip.*",
        r".*zipcode.*",
        r".*postal.*",
        r".*phone.*",
        r".*telephone.*",
        r".*fax.*",
        r".*email.*",
        r".*e_mail.*",
        r".*health_plan.*",
        r".*account_number.*",
        r".*device_id.*",
        r".*ip_address.*",
        r".*biometric.*",
    ])

    def is_phi_field(self, field_name: str) -> bool:
        """Check if a field name matches a PHI pattern."""
        normalized = field_name.lower().strip()
        for pattern in self.phi_field_patterns:
            if re.match(pattern, normalized):
                return True
        return False

    def mask_value(self, value: Any) -> str:
        """Create an irreversible masked token from a value using SHA-256.

        The token is a prefix of the SHA-256 hash, making it irreversible
        while keeping it a consistent token for the same input.
        """
        value_str = str(value)
        hash_digest = hashlib.sha256(value_str.encode("utf-8")).hexdigest()
        return f"MASKED_{hash_digest[:16]}"


# =============================================================================
# Identity Service
# =============================================================================


class IdentityService:
    """Handles authentication, RBAC enforcement, and PII/PHI masking.

    Requirement 5.1: Replace all PII/PHI fields with irreversible masked tokens.
    Requirement 5.2: Authenticate requesting identity and verify RBAC policy.
    Requirement 5.3: Deny without exposing clinical data; log unauthorized attempts.
    Requirement 5.4: Associate every action with authenticated identity_id.
    Requirement 5.5: Reject on auth failure; log reason.
    """

    def __init__(self, rbac_policy: RBACPolicy, masking_service: MaskingService):
        self.rbac = rbac_policy
        self.masker = masking_service

    async def authenticate_and_authorize(
        self, credentials: Credentials, operation: str
    ) -> AuthResult:
        """Authenticate identity and verify RBAC permissions.

        Validates credentials (api_key or token must be present),
        looks up identity in RBAC policy, and checks if the assigned
        role permits the requested operation.

        Returns AuthResult with identity_id and granted permissions on success.
        Raises UnauthorizedError if identity lacks permissions.
        No clinical data is ever exposed in error responses.
        """
        # Step 1: Validate credentials presence
        if not credentials.api_key and not credentials.token:
            logger.warning(
                "Authentication failure: missing credentials | "
                f"timestamp={datetime.now(timezone.utc).isoformat()} | "
                f"reason=missing_credentials"
            )
            raise UnauthorizedError(
                reason="Authentication failed: missing credentials",
                identity=credentials.identity_id,
                operation=operation,
            )

        # Step 2: Validate credentials are not empty/invalid
        credential_value = credentials.api_key or credentials.token
        if not credential_value or not credential_value.strip():
            logger.warning(
                "Authentication failure: invalid credentials | "
                f"timestamp={datetime.now(timezone.utc).isoformat()} | "
                f"reason=invalid_credentials"
            )
            raise UnauthorizedError(
                reason="Authentication failed: invalid credentials",
                identity=credentials.identity_id,
                operation=operation,
            )

        # Step 3: Look up identity in RBAC policy
        identity_id = credentials.identity_id
        assigned_role = self.rbac.identity_role_assignments.get(identity_id)

        if assigned_role is None:
            logger.warning(
                "Authentication failure: identity not found in RBAC policy | "
                f"timestamp={datetime.now(timezone.utc).isoformat()} | "
                f"reason=identity_not_found | "
                f"identity={identity_id}"
            )
            raise UnauthorizedError(
                reason="Authentication failed: identity not recognized",
                identity=identity_id,
                operation=operation,
            )

        # Step 4: Check if assigned role exists in roles definition
        role_permissions = self.rbac.roles.get(assigned_role)
        if role_permissions is None:
            logger.warning(
                "Authorization failure: role not defined | "
                f"timestamp={datetime.now(timezone.utc).isoformat()} | "
                f"reason=role_not_defined | "
                f"identity={identity_id} | "
                f"role={assigned_role}"
            )
            raise UnauthorizedError(
                reason="Authorization failed: assigned role not defined",
                identity=identity_id,
                operation=operation,
                missing_permission=operation,
            )

        # Step 5: Check if role permits the requested operation
        if operation not in role_permissions:
            logger.warning(
                "Unauthorized access attempt | "
                f"identity={identity_id} | "
                f"operation={operation} | "
                f"timestamp={datetime.now(timezone.utc).isoformat()} | "
                f"missing_permission={operation}"
            )
            raise UnauthorizedError(
                reason="Authorization failed: insufficient permissions",
                identity=identity_id,
                operation=operation,
                missing_permission=operation,
            )

        # Success: return AuthResult
        logger.info(
            f"Authentication successful | identity={identity_id} | operation={operation}"
        )
        return AuthResult(
            identity_id=identity_id,
            granted_permissions=role_permissions,
            authenticated_at=datetime.now(timezone.utc),
            session_id=str(uuid.uuid4()),
        )

    def mask_phi(self, data: dict) -> dict:
        """Replace all PII/PHI fields with irreversible masked tokens.

        Requirement 5.1: Replace all PII and PHI fields in trace logs and
        observability outputs with irreversible masked tokens before
        they are persisted. No original identifiable value is recoverable.

        Processes the dictionary recursively, masking values for any
        key that matches PHI field patterns.
        """
        return self._mask_recursive(data)

    def _mask_recursive(self, data: Any) -> Any:
        """Recursively mask PHI fields in nested structures."""
        if isinstance(data, dict):
            masked = {}
            for key, value in data.items():
                if self.masker.is_phi_field(key):
                    masked[key] = self.masker.mask_value(value)
                elif isinstance(value, dict):
                    masked[key] = self._mask_recursive(value)
                elif isinstance(value, list):
                    masked[key] = [self._mask_recursive(item) for item in value]
                else:
                    masked[key] = value
            return masked
        elif isinstance(data, list):
            return [self._mask_recursive(item) for item in data]
        else:
            return data

    def create_trace_context(
        self, identity_id: str, request_id: str
    ) -> TraceContext:
        """Create a trace context associating all actions with the authenticated identity.

        Requirement 5.4: Every agent action is associated with the authenticated
        identity identifier in the execution trace so that each trace entry
        is attributable to a specific identity for audit purposes.
        """
        return TraceContext(
            identity_id=identity_id,
            request_id=request_id,
        )
