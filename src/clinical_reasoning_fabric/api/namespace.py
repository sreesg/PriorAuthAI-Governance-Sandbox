"""Namespace model and validation for Axisweave Service API.

Provides namespace lifecycle management and cross-namespace access grants
for multi-tenant isolation in the Axisweave Retrieval Stack.

Requirements referenced: 13.2, 13.8
- 13.2: Namespace identifier must be 1-128 alphanumeric, hyphen, or underscore characters.
- 13.8: Reject requests with invalid/missing namespace; log rejection to audit trail.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from clinical_reasoning_fabric.models.exceptions import InvalidNamespaceError


# Regex pattern for valid namespace identifiers: 1-128 alphanumeric, hyphen, underscore
NAMESPACE_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")


def validate_namespace(namespace: str) -> bool:
    """Validate that a namespace identifier conforms to the required format.

    Valid format: 1-128 characters, each being alphanumeric, hyphen, or underscore.
    Pattern: ^[a-zA-Z0-9_-]{1,128}$

    Args:
        namespace: The namespace identifier string to validate.

    Returns:
        True if the namespace is valid.

    Raises:
        InvalidNamespaceError: If the namespace is empty, exceeds 128 characters,
            or contains characters outside [a-zA-Z0-9_-].
    """
    if not namespace:
        raise InvalidNamespaceError(
            reason="Namespace identifier must not be empty",
            namespace=namespace,
        )

    if not NAMESPACE_PATTERN.match(namespace):
        if len(namespace) > 128:
            reason = (
                f"Namespace identifier exceeds maximum length of 128 characters "
                f"(got {len(namespace)} characters)"
            )
        else:
            reason = (
                "Namespace identifier contains invalid characters. "
                "Only alphanumeric characters, hyphens, and underscores are allowed "
                "(pattern: ^[a-zA-Z0-9_-]{1,128}$)"
            )
        raise InvalidNamespaceError(
            reason=reason,
            namespace=namespace,
        )

    return True


@dataclass
class Namespace:
    """Represents a tenant namespace for document isolation in the Axisweave service.

    Attributes:
        namespace_id: Unique identifier (1-128 alphanumeric, hyphen, underscore chars).
        owner_tenant_id: The tenant that owns this namespace.
        created_at: UTC timestamp when the namespace was created.
        cross_namespace_grants: List of namespace_ids this tenant can also access.
    """

    namespace_id: str
    owner_tenant_id: str
    created_at: datetime
    cross_namespace_grants: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate namespace_id format on creation."""
        validate_namespace(self.namespace_id)


class NamespaceRegistry:
    """Manages namespace lifecycle and cross-namespace access grants.

    Provides CRUD operations for namespaces and controls cross-namespace
    access grants for multi-tenant isolation.
    """

    def __init__(self) -> None:
        """Initialize the registry with an empty namespace store."""
        self._namespaces: dict[str, Namespace] = {}

    def register_namespace(self, namespace_id: str, owner_tenant_id: str) -> Namespace:
        """Register a new namespace owned by the specified tenant.

        Args:
            namespace_id: The namespace identifier (validated against format rules).
            owner_tenant_id: The tenant that will own this namespace.

        Returns:
            The newly created Namespace instance.

        Raises:
            InvalidNamespaceError: If namespace_id format is invalid.
            ValueError: If namespace_id is already registered.
        """
        validate_namespace(namespace_id)

        if namespace_id in self._namespaces:
            raise ValueError(
                f"Namespace '{namespace_id}' is already registered"
            )

        namespace = Namespace(
            namespace_id=namespace_id,
            owner_tenant_id=owner_tenant_id,
            created_at=datetime.now(timezone.utc),
            cross_namespace_grants=[],
        )
        self._namespaces[namespace_id] = namespace
        return namespace

    def get_namespace(self, namespace_id: str) -> Optional[Namespace]:
        """Retrieve a namespace by its identifier.

        Args:
            namespace_id: The namespace identifier to look up.

        Returns:
            The Namespace if found, None otherwise.
        """
        return self._namespaces.get(namespace_id)

    def grant_cross_namespace_access(
        self, owner_namespace: str, target_namespace: str
    ) -> None:
        """Grant cross-namespace access from the owner namespace to a target namespace.

        This allows the tenant owning `owner_namespace` to also access documents
        in `target_namespace`.

        Args:
            owner_namespace: The namespace whose owner is being granted access.
            target_namespace: The namespace being granted access to.

        Raises:
            ValueError: If owner_namespace is not registered or target_namespace
                is not registered, or if granting access to itself.
            InvalidNamespaceError: If either namespace_id format is invalid.
        """
        validate_namespace(owner_namespace)
        validate_namespace(target_namespace)

        if owner_namespace not in self._namespaces:
            raise ValueError(
                f"Owner namespace '{owner_namespace}' is not registered"
            )

        if target_namespace not in self._namespaces:
            raise ValueError(
                f"Target namespace '{target_namespace}' is not registered"
            )

        if owner_namespace == target_namespace:
            raise ValueError("Cannot grant cross-namespace access to the same namespace")

        namespace = self._namespaces[owner_namespace]
        if target_namespace not in namespace.cross_namespace_grants:
            namespace.cross_namespace_grants.append(target_namespace)

    def has_access(self, tenant_id: str, target_namespace: str) -> bool:
        """Check if a tenant has access to a target namespace.

        A tenant has access to a namespace if:
        1. They own the namespace, OR
        2. They own a namespace that has been granted cross-namespace access
           to the target namespace.

        Args:
            tenant_id: The tenant identifier to check access for.
            target_namespace: The namespace to check access to.

        Returns:
            True if the tenant has access to the target namespace.
        """
        # Check direct ownership
        target_ns = self._namespaces.get(target_namespace)
        if target_ns and target_ns.owner_tenant_id == tenant_id:
            return True

        # Check cross-namespace grants from any namespace owned by this tenant
        for ns in self._namespaces.values():
            if ns.owner_tenant_id == tenant_id:
                if target_namespace in ns.cross_namespace_grants:
                    return True

        return False

    def revoke_cross_namespace_access(
        self, owner_namespace: str, target_namespace: str
    ) -> None:
        """Revoke cross-namespace access from the owner namespace to a target namespace.

        Args:
            owner_namespace: The namespace whose grant is being revoked.
            target_namespace: The namespace whose access is being revoked.

        Raises:
            ValueError: If owner_namespace is not registered.
        """
        if owner_namespace not in self._namespaces:
            raise ValueError(
                f"Owner namespace '{owner_namespace}' is not registered"
            )

        namespace = self._namespaces[owner_namespace]
        if target_namespace in namespace.cross_namespace_grants:
            namespace.cross_namespace_grants.remove(target_namespace)

    def list_namespaces(self, owner_tenant_id: Optional[str] = None) -> list[Namespace]:
        """List all namespaces, optionally filtered by owner tenant.

        Args:
            owner_tenant_id: If provided, only return namespaces owned by this tenant.

        Returns:
            List of Namespace instances.
        """
        if owner_tenant_id is None:
            return list(self._namespaces.values())

        return [
            ns for ns in self._namespaces.values()
            if ns.owner_tenant_id == owner_tenant_id
        ]
