"""Unit tests for namespace model and validation.

Tests namespace format validation, Namespace dataclass creation,
and NamespaceRegistry CRUD operations including cross-namespace grants.

Validates:
    - validate_namespace() accepts valid alphanumeric/hyphen/underscore strings (1-128 chars)
    - validate_namespace() rejects empty strings
    - validate_namespace() rejects strings exceeding 128 characters
    - validate_namespace() rejects strings with special characters
    - Namespace dataclass validates on creation
    - NamespaceRegistry register/get/grant/has_access operations
    - Cross-namespace access granting and checking

Requirements referenced: 13.2, 13.8
"""

from datetime import datetime, timezone

import pytest

from clinical_reasoning_fabric.api.namespace import (
    NAMESPACE_PATTERN,
    Namespace,
    NamespaceRegistry,
    validate_namespace,
)
from clinical_reasoning_fabric.models.exceptions import InvalidNamespaceError


# =============================================================================
# validate_namespace() Tests
# =============================================================================


class TestValidateNamespace:
    """Tests for the validate_namespace() function."""

    def test_valid_simple_alphanumeric(self):
        """Simple alphanumeric strings are accepted."""
        assert validate_namespace("myNamespace") is True
        assert validate_namespace("namespace123") is True
        assert validate_namespace("ABC") is True

    def test_valid_with_hyphens(self):
        """Strings with hyphens are accepted."""
        assert validate_namespace("my-namespace") is True
        assert validate_namespace("a-b-c-d") is True

    def test_valid_with_underscores(self):
        """Strings with underscores are accepted."""
        assert validate_namespace("my_namespace") is True
        assert validate_namespace("a_b_c_d") is True

    def test_valid_mixed_chars(self):
        """Mixed alphanumeric, hyphen, and underscore strings are accepted."""
        assert validate_namespace("pa-workflow_v2") is True
        assert validate_namespace("HEDIS_Gap-Closure_2024") is True
        assert validate_namespace("fraud-detection_ns") is True

    def test_valid_single_character(self):
        """Single character namespace is accepted (minimum length 1)."""
        assert validate_namespace("a") is True
        assert validate_namespace("Z") is True
        assert validate_namespace("0") is True
        assert validate_namespace("-") is True
        assert validate_namespace("_") is True

    def test_valid_128_characters(self):
        """Exactly 128 character namespace is accepted (maximum length)."""
        ns = "a" * 128
        assert validate_namespace(ns) is True

    def test_reject_empty_string(self):
        """Empty string raises InvalidNamespaceError."""
        with pytest.raises(InvalidNamespaceError) as exc_info:
            validate_namespace("")

        assert "empty" in exc_info.value.reason.lower()
        assert exc_info.value.namespace == ""

    def test_reject_exceeds_128_characters(self):
        """String longer than 128 characters raises InvalidNamespaceError."""
        ns = "a" * 129
        with pytest.raises(InvalidNamespaceError) as exc_info:
            validate_namespace(ns)

        assert "128" in exc_info.value.reason
        assert exc_info.value.namespace == ns

    def test_reject_space_character(self):
        """Strings with spaces are rejected."""
        with pytest.raises(InvalidNamespaceError) as exc_info:
            validate_namespace("my namespace")

        assert "invalid characters" in exc_info.value.reason.lower()

    def test_reject_dot_character(self):
        """Strings with dots are rejected."""
        with pytest.raises(InvalidNamespaceError):
            validate_namespace("my.namespace")

    def test_reject_slash_character(self):
        """Strings with slashes are rejected."""
        with pytest.raises(InvalidNamespaceError):
            validate_namespace("my/namespace")

    def test_reject_at_sign(self):
        """Strings with @ are rejected."""
        with pytest.raises(InvalidNamespaceError):
            validate_namespace("user@namespace")

    def test_reject_exclamation_mark(self):
        """Strings with ! are rejected."""
        with pytest.raises(InvalidNamespaceError):
            validate_namespace("namespace!")

    def test_reject_unicode_characters(self):
        """Strings with unicode characters are rejected."""
        with pytest.raises(InvalidNamespaceError):
            validate_namespace("namespace_ñ")

    def test_reject_newline(self):
        """Strings with newlines are rejected."""
        with pytest.raises(InvalidNamespaceError):
            validate_namespace("name\nspace")

    def test_reject_tab(self):
        """Strings with tabs are rejected."""
        with pytest.raises(InvalidNamespaceError):
            validate_namespace("name\tspace")

    def test_error_includes_namespace_value(self):
        """InvalidNamespaceError includes the offending namespace."""
        bad_ns = "bad!namespace"
        with pytest.raises(InvalidNamespaceError) as exc_info:
            validate_namespace(bad_ns)

        assert exc_info.value.namespace == bad_ns


# =============================================================================
# Namespace Dataclass Tests
# =============================================================================


class TestNamespaceDataclass:
    """Tests for the Namespace dataclass."""

    def test_create_valid_namespace(self):
        """Namespace can be created with valid fields."""
        now = datetime.now(timezone.utc)
        ns = Namespace(
            namespace_id="pa-workflow",
            owner_tenant_id="tenant-001",
            created_at=now,
        )

        assert ns.namespace_id == "pa-workflow"
        assert ns.owner_tenant_id == "tenant-001"
        assert ns.created_at == now
        assert ns.cross_namespace_grants == []

    def test_create_with_cross_namespace_grants(self):
        """Namespace can be created with pre-existing grants."""
        ns = Namespace(
            namespace_id="primary-ns",
            owner_tenant_id="tenant-001",
            created_at=datetime.now(timezone.utc),
            cross_namespace_grants=["shared-ns", "partner-ns"],
        )

        assert ns.cross_namespace_grants == ["shared-ns", "partner-ns"]

    def test_create_rejects_invalid_namespace_id(self):
        """Namespace creation raises InvalidNamespaceError for invalid namespace_id."""
        with pytest.raises(InvalidNamespaceError):
            Namespace(
                namespace_id="bad namespace!",
                owner_tenant_id="tenant-001",
                created_at=datetime.now(timezone.utc),
            )

    def test_create_rejects_empty_namespace_id(self):
        """Namespace creation raises InvalidNamespaceError for empty namespace_id."""
        with pytest.raises(InvalidNamespaceError):
            Namespace(
                namespace_id="",
                owner_tenant_id="tenant-001",
                created_at=datetime.now(timezone.utc),
            )


# =============================================================================
# NamespaceRegistry Tests
# =============================================================================


class TestNamespaceRegistry:
    """Tests for NamespaceRegistry CRUD operations."""

    @pytest.fixture
    def registry(self):
        """Fresh NamespaceRegistry instance."""
        return NamespaceRegistry()

    def test_register_namespace(self, registry):
        """Register a new namespace successfully."""
        ns = registry.register_namespace("hedis-ns", "tenant-001")

        assert isinstance(ns, Namespace)
        assert ns.namespace_id == "hedis-ns"
        assert ns.owner_tenant_id == "tenant-001"
        assert ns.cross_namespace_grants == []
        assert isinstance(ns.created_at, datetime)

    def test_register_duplicate_raises_value_error(self, registry):
        """Registering a duplicate namespace raises ValueError."""
        registry.register_namespace("hedis-ns", "tenant-001")

        with pytest.raises(ValueError, match="already registered"):
            registry.register_namespace("hedis-ns", "tenant-002")

    def test_register_invalid_namespace_raises_error(self, registry):
        """Registering an invalid namespace raises InvalidNamespaceError."""
        with pytest.raises(InvalidNamespaceError):
            registry.register_namespace("bad namespace!", "tenant-001")

    def test_get_namespace_found(self, registry):
        """Get a registered namespace by ID."""
        registry.register_namespace("my-ns", "tenant-001")
        ns = registry.get_namespace("my-ns")

        assert ns is not None
        assert ns.namespace_id == "my-ns"

    def test_get_namespace_not_found(self, registry):
        """Get returns None for unregistered namespace."""
        ns = registry.get_namespace("nonexistent")
        assert ns is None

    def test_grant_cross_namespace_access(self, registry):
        """Grant cross-namespace access between two namespaces."""
        registry.register_namespace("ns-a", "tenant-001")
        registry.register_namespace("ns-b", "tenant-002")

        registry.grant_cross_namespace_access("ns-a", "ns-b")

        ns_a = registry.get_namespace("ns-a")
        assert "ns-b" in ns_a.cross_namespace_grants

    def test_grant_idempotent(self, registry):
        """Granting the same access twice doesn't duplicate entries."""
        registry.register_namespace("ns-a", "tenant-001")
        registry.register_namespace("ns-b", "tenant-002")

        registry.grant_cross_namespace_access("ns-a", "ns-b")
        registry.grant_cross_namespace_access("ns-a", "ns-b")

        ns_a = registry.get_namespace("ns-a")
        assert ns_a.cross_namespace_grants.count("ns-b") == 1

    def test_grant_to_self_raises_error(self, registry):
        """Granting cross-namespace access to itself raises ValueError."""
        registry.register_namespace("ns-a", "tenant-001")

        with pytest.raises(ValueError, match="same namespace"):
            registry.grant_cross_namespace_access("ns-a", "ns-a")

    def test_grant_unregistered_owner_raises_error(self, registry):
        """Granting from unregistered owner namespace raises ValueError."""
        registry.register_namespace("ns-b", "tenant-002")

        with pytest.raises(ValueError, match="not registered"):
            registry.grant_cross_namespace_access("unregistered", "ns-b")

    def test_grant_unregistered_target_raises_error(self, registry):
        """Granting to unregistered target namespace raises ValueError."""
        registry.register_namespace("ns-a", "tenant-001")

        with pytest.raises(ValueError, match="not registered"):
            registry.grant_cross_namespace_access("ns-a", "unregistered")

    def test_has_access_owner(self, registry):
        """Tenant has access to namespaces they own."""
        registry.register_namespace("ns-a", "tenant-001")

        assert registry.has_access("tenant-001", "ns-a") is True

    def test_has_access_no_ownership(self, registry):
        """Tenant does not have access to namespaces they don't own without grant."""
        registry.register_namespace("ns-a", "tenant-001")

        assert registry.has_access("tenant-002", "ns-a") is False

    def test_has_access_via_cross_namespace_grant(self, registry):
        """Tenant has access to granted cross-namespace targets."""
        registry.register_namespace("ns-a", "tenant-001")
        registry.register_namespace("ns-b", "tenant-002")
        registry.grant_cross_namespace_access("ns-a", "ns-b")

        # tenant-001 owns ns-a which has a grant to ns-b
        assert registry.has_access("tenant-001", "ns-b") is True

    def test_has_access_grant_is_directional(self, registry):
        """Cross-namespace grants are directional: A→B does not imply B→A."""
        registry.register_namespace("ns-a", "tenant-001")
        registry.register_namespace("ns-b", "tenant-002")
        registry.grant_cross_namespace_access("ns-a", "ns-b")

        # tenant-002 owns ns-b, but ns-b does NOT have a grant to ns-a
        assert registry.has_access("tenant-002", "ns-a") is False

    def test_has_access_unregistered_namespace(self, registry):
        """Access to an unregistered namespace returns False."""
        registry.register_namespace("ns-a", "tenant-001")
        assert registry.has_access("tenant-001", "nonexistent") is False

    def test_revoke_cross_namespace_access(self, registry):
        """Revoke a previously granted cross-namespace access."""
        registry.register_namespace("ns-a", "tenant-001")
        registry.register_namespace("ns-b", "tenant-002")
        registry.grant_cross_namespace_access("ns-a", "ns-b")

        registry.revoke_cross_namespace_access("ns-a", "ns-b")

        assert registry.has_access("tenant-001", "ns-b") is False
        ns_a = registry.get_namespace("ns-a")
        assert "ns-b" not in ns_a.cross_namespace_grants

    def test_revoke_nonexistent_grant_is_noop(self, registry):
        """Revoking a grant that doesn't exist is a no-op."""
        registry.register_namespace("ns-a", "tenant-001")
        registry.register_namespace("ns-b", "tenant-002")

        # No error raised even though no grant exists
        registry.revoke_cross_namespace_access("ns-a", "ns-b")

    def test_list_namespaces_all(self, registry):
        """List all registered namespaces."""
        registry.register_namespace("ns-a", "tenant-001")
        registry.register_namespace("ns-b", "tenant-002")
        registry.register_namespace("ns-c", "tenant-001")

        all_ns = registry.list_namespaces()
        assert len(all_ns) == 3

    def test_list_namespaces_by_owner(self, registry):
        """List namespaces filtered by owner tenant."""
        registry.register_namespace("ns-a", "tenant-001")
        registry.register_namespace("ns-b", "tenant-002")
        registry.register_namespace("ns-c", "tenant-001")

        tenant_ns = registry.list_namespaces(owner_tenant_id="tenant-001")
        assert len(tenant_ns) == 2
        assert all(ns.owner_tenant_id == "tenant-001" for ns in tenant_ns)

    def test_list_namespaces_empty_registry(self, registry):
        """Empty registry returns empty list."""
        assert registry.list_namespaces() == []
