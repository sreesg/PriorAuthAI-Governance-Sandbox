"""Unit tests for API versioning router and backwards compatibility.

Tests cover:
- Semantic version parsing and validation
- Request routing to correct handler version
- 410 Gone for unsupported versions
- Backwards compatibility enforcement (6 months minimum)
- Deprecation schedule validation
- APIVersionInfo dataclass construction

Requirements referenced: 13.6
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from clinical_reasoning_fabric.api.version_router import (
    APIVersionInfo,
    APIVersionRouter,
    InvalidVersionError,
    VersionGoneError,
    get_major_version,
    is_deprecation_valid,
    parse_semver,
)


# =============================================================================
# Tests for parse_semver
# =============================================================================


class TestParseSemver:
    """Tests for the parse_semver helper function."""

    def test_valid_version(self):
        assert parse_semver("1.0.0") == (1, 0, 0)

    def test_valid_version_higher_numbers(self):
        assert parse_semver("12.34.56") == (12, 34, 56)

    def test_valid_version_zeros(self):
        assert parse_semver("0.0.0") == (0, 0, 0)

    def test_invalid_format_missing_patch(self):
        assert parse_semver("1.0") is None

    def test_invalid_format_extra_segment(self):
        assert parse_semver("1.0.0.0") is None

    def test_invalid_format_letters(self):
        assert parse_semver("abc") is None

    def test_invalid_format_empty(self):
        assert parse_semver("") is None

    def test_invalid_leading_zeros(self):
        assert parse_semver("01.0.0") is None

    def test_invalid_negative(self):
        assert parse_semver("-1.0.0") is None

    def test_invalid_prerelease_suffix(self):
        # Strict semver core only — no pre-release suffixes
        assert parse_semver("1.0.0-alpha") is None

    def test_invalid_build_metadata(self):
        assert parse_semver("1.0.0+build") is None


# =============================================================================
# Tests for get_major_version
# =============================================================================


class TestGetMajorVersion:
    """Tests for extracting major version."""

    def test_extracts_major(self):
        assert get_major_version("2.1.3") == 2

    def test_invalid_returns_none(self):
        assert get_major_version("invalid") is None


# =============================================================================
# Tests for APIVersionInfo
# =============================================================================


class TestAPIVersionInfo:
    """Tests for the APIVersionInfo dataclass."""

    def test_valid_construction(self):
        info = APIVersionInfo(
            current_version="2.0.0",
            supported_versions=["1.0.0", "2.0.0"],
            deprecation_schedule={"1.0.0": "2025-12-31"},
        )
        assert info.current_version == "2.0.0"
        assert "1.0.0" in info.supported_versions
        assert "2.0.0" in info.supported_versions

    def test_current_version_auto_added_to_supported(self):
        info = APIVersionInfo(
            current_version="3.0.0",
            supported_versions=["2.0.0"],
        )
        assert "3.0.0" in info.supported_versions

    def test_invalid_current_version_raises(self):
        with pytest.raises(InvalidVersionError):
            APIVersionInfo(current_version="not-a-version")

    def test_invalid_supported_version_raises(self):
        with pytest.raises(InvalidVersionError):
            APIVersionInfo(
                current_version="1.0.0",
                supported_versions=["bad"],
            )


# =============================================================================
# Tests for APIVersionRouter — Version Routing
# =============================================================================


class TestAPIVersionRouterRouting:
    """Tests for version-based request routing."""

    def setup_method(self):
        """Set up a router with v1 and v2 handlers."""
        self.version_info = APIVersionInfo(
            current_version="2.0.0",
            supported_versions=["1.0.0", "2.0.0"],
            deprecation_schedule={"1.0.0": "2030-12-31"},  # Far future — still supported
        )
        self.router = APIVersionRouter(self.version_info)

        # Register mock handlers
        self.v1_ingest = MagicMock(name="v1_ingest_handler")
        self.v2_ingest = MagicMock(name="v2_ingest_handler")
        self.v1_retrieve = MagicMock(name="v1_retrieve_handler")
        self.v2_retrieve = MagicMock(name="v2_retrieve_handler")

        self.router.register_handler("1", "ingest", self.v1_ingest)
        self.router.register_handler("2", "ingest", self.v2_ingest)
        self.router.register_handler("1", "retrieve", self.v1_retrieve)
        self.router.register_handler("2", "retrieve", self.v2_retrieve)

    def test_routes_v1_request_to_v1_handler(self):
        handler = self.router.route_request("1.0.0", "ingest")
        assert handler is self.v1_ingest

    def test_routes_v2_request_to_v2_handler(self):
        handler = self.router.route_request("2.0.0", "ingest")
        assert handler is self.v2_ingest

    def test_routes_minor_version_to_major_handler(self):
        """Minor versions route to the same major version handler."""
        handler = self.router.route_request("1.2.0", "ingest")
        assert handler is self.v1_ingest

    def test_routes_patch_version_to_major_handler(self):
        """Patch versions route to the same major version handler."""
        handler = self.router.route_request("2.1.3", "retrieve")
        assert handler is self.v2_retrieve

    def test_routes_different_endpoints(self):
        handler = self.router.route_request("1.0.0", "retrieve")
        assert handler is self.v1_retrieve

    def test_invalid_version_raises_invalid_version_error(self):
        with pytest.raises(InvalidVersionError):
            self.router.route_request("bad-version", "ingest")

    def test_unregistered_endpoint_raises_key_error(self):
        with pytest.raises(KeyError):
            self.router.route_request("1.0.0", "nonexistent_endpoint")


# =============================================================================
# Tests for APIVersionRouter — Unsupported Versions (410 Gone)
# =============================================================================


class TestAPIVersionRouterUnsupported:
    """Tests for 410 Gone responses for unsupported versions."""

    def setup_method(self):
        """Set up a router where v1 is no longer supported."""
        self.version_info = APIVersionInfo(
            current_version="2.0.0",
            supported_versions=["2.0.0"],  # Only v2 supported
            deprecation_schedule={},
        )
        self.router = APIVersionRouter(self.version_info)
        self.router.register_handler("2", "ingest", MagicMock())

    def test_unsupported_version_raises_version_gone(self):
        with pytest.raises(VersionGoneError) as exc_info:
            self.router.route_request("1.0.0", "ingest")
        assert exc_info.value.requested_version == "1.0.0"
        assert "2.0.0" in exc_info.value.supported_versions

    def test_unsupported_minor_version_raises_version_gone(self):
        """A minor version within an unsupported major raises 410."""
        with pytest.raises(VersionGoneError):
            self.router.route_request("1.5.2", "ingest")

    def test_future_unsupported_version_raises_version_gone(self):
        """A version with major > current but not in supported list raises 410."""
        with pytest.raises(VersionGoneError):
            self.router.route_request("3.0.0", "ingest")

    def test_expired_deprecation_raises_version_gone(self):
        """A version whose deprecation date has passed raises 410."""
        version_info = APIVersionInfo(
            current_version="2.0.0",
            supported_versions=["1.0.0", "2.0.0"],
            deprecation_schedule={"1.0.0": "2020-01-01"},  # Past date
        )
        router = APIVersionRouter(version_info)
        router.register_handler("2", "ingest", MagicMock())

        with pytest.raises(VersionGoneError):
            router.route_request("1.0.0", "ingest")


# =============================================================================
# Tests for APIVersionRouter — Backward Compatibility Enforcement
# =============================================================================


class TestAPIVersionRouterBackwardCompatibility:
    """Tests for backward compatibility — prior major version for 6+ months."""

    def test_prior_version_supported_within_deprecation_window(self):
        """Prior major version is supported when deprecation date is in the future."""
        future_date = (datetime.now(timezone.utc) + timedelta(days=365)).strftime("%Y-%m-%d")
        version_info = APIVersionInfo(
            current_version="2.0.0",
            supported_versions=["1.0.0", "2.0.0"],
            deprecation_schedule={"1.0.0": future_date},
        )
        router = APIVersionRouter(version_info)
        assert router.is_version_supported("1.0.0") is True

    def test_prior_version_unsupported_after_deprecation(self):
        """Prior major version becomes unsupported after deprecation date passes."""
        past_date = "2020-01-01"
        version_info = APIVersionInfo(
            current_version="2.0.0",
            supported_versions=["1.0.0", "2.0.0"],
            deprecation_schedule={"1.0.0": past_date},
        )
        router = APIVersionRouter(version_info)
        assert router.is_version_supported("1.0.0") is False

    def test_validate_deprecation_schedule_compliant(self):
        """Deprecation 7 months after release is compliant (>= 6 months)."""
        version_info = APIVersionInfo(
            current_version="2.0.0",
            supported_versions=["1.0.0", "2.0.0"],
            deprecation_schedule={"1.0.0": "2025-08-01"},
        )
        router = APIVersionRouter(version_info)

        # Release date: Jan 1, 2025. Deprecation: Aug 1, 2025 → 7 months → compliant
        release_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
        results = router.validate_deprecation_schedule(release_date)
        assert results["1.0.0"] is True

    def test_validate_deprecation_schedule_non_compliant(self):
        """Deprecation 2 months after release is non-compliant (< 6 months)."""
        version_info = APIVersionInfo(
            current_version="2.0.0",
            supported_versions=["1.0.0", "2.0.0"],
            deprecation_schedule={"1.0.0": "2025-03-01"},
        )
        router = APIVersionRouter(version_info)

        # Release date: Jan 1, 2025. Deprecation: Mar 1, 2025 → 2 months → non-compliant
        release_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
        results = router.validate_deprecation_schedule(release_date)
        assert results["1.0.0"] is False

    def test_validate_deprecation_schedule_exactly_6_months(self):
        """Deprecation exactly 6 months after release is compliant."""
        version_info = APIVersionInfo(
            current_version="2.0.0",
            supported_versions=["1.0.0", "2.0.0"],
            deprecation_schedule={"1.0.0": "2025-07-01"},
        )
        router = APIVersionRouter(version_info)

        release_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
        results = router.validate_deprecation_schedule(release_date)
        assert results["1.0.0"] is True


# =============================================================================
# Tests for APIVersionRouter — is_version_supported
# =============================================================================


class TestIsVersionSupported:
    """Tests for is_version_supported method."""

    def setup_method(self):
        self.version_info = APIVersionInfo(
            current_version="2.0.0",
            supported_versions=["1.0.0", "2.0.0"],
            deprecation_schedule={},
        )
        self.router = APIVersionRouter(self.version_info)

    def test_current_version_supported(self):
        assert self.router.is_version_supported("2.0.0") is True

    def test_supported_prior_version(self):
        assert self.router.is_version_supported("1.0.0") is True

    def test_minor_version_of_supported_major(self):
        """Minor versions within a supported major are supported."""
        assert self.router.is_version_supported("1.5.0") is True
        assert self.router.is_version_supported("2.3.1") is True

    def test_unsupported_major_version(self):
        assert self.router.is_version_supported("3.0.0") is False

    def test_invalid_version_string(self):
        assert self.router.is_version_supported("invalid") is False

    def test_empty_version_string(self):
        assert self.router.is_version_supported("") is False


# =============================================================================
# Tests for APIVersionRouter — Utility Methods
# =============================================================================


class TestAPIVersionRouterUtilities:
    """Tests for utility methods on the router."""

    def test_get_supported_versions_filters_expired(self):
        version_info = APIVersionInfo(
            current_version="3.0.0",
            supported_versions=["1.0.0", "2.0.0", "3.0.0"],
            deprecation_schedule={
                "1.0.0": "2020-01-01",  # Expired
                "2.0.0": "2030-01-01",  # Still active
            },
        )
        router = APIVersionRouter(version_info)
        supported = router.get_supported_versions()

        assert "1.0.0" not in supported
        assert "2.0.0" in supported
        assert "3.0.0" in supported

    def test_get_deprecation_date_returns_date(self):
        version_info = APIVersionInfo(
            current_version="2.0.0",
            supported_versions=["1.0.0", "2.0.0"],
            deprecation_schedule={"1.0.0": "2025-12-31"},
        )
        router = APIVersionRouter(version_info)
        assert router.get_deprecation_date("1.0.0") == "2025-12-31"
        assert router.get_deprecation_date("1.5.2") == "2025-12-31"

    def test_get_deprecation_date_returns_none_for_non_deprecated(self):
        version_info = APIVersionInfo(
            current_version="2.0.0",
            supported_versions=["2.0.0"],
            deprecation_schedule={},
        )
        router = APIVersionRouter(version_info)
        assert router.get_deprecation_date("2.0.0") is None

    def test_is_version_deprecated_active(self):
        """A version with a future deprecation date is deprecated but supported."""
        future_date = (datetime.now(timezone.utc) + timedelta(days=180)).strftime("%Y-%m-%d")
        version_info = APIVersionInfo(
            current_version="2.0.0",
            supported_versions=["1.0.0", "2.0.0"],
            deprecation_schedule={"1.0.0": future_date},
        )
        router = APIVersionRouter(version_info)
        assert router.is_version_deprecated("1.0.0") is True
        assert router.is_version_deprecated("2.0.0") is False

    def test_is_version_deprecated_expired(self):
        """A version with a past deprecation date is no longer deprecated (it's gone)."""
        version_info = APIVersionInfo(
            current_version="2.0.0",
            supported_versions=["1.0.0", "2.0.0"],
            deprecation_schedule={"1.0.0": "2020-01-01"},
        )
        router = APIVersionRouter(version_info)
        # Past deprecation = no longer "deprecated" (it's just gone)
        assert router.is_version_deprecated("1.0.0") is False


# =============================================================================
# Tests for is_deprecation_valid helper
# =============================================================================


class TestIsDeprecationValid:
    """Tests for the is_deprecation_valid helper function."""

    def test_sufficient_gap(self):
        release = datetime(2025, 1, 1, tzinfo=timezone.utc)
        assert is_deprecation_valid("2025-07-01", release) is True

    def test_insufficient_gap(self):
        release = datetime(2025, 1, 1, tzinfo=timezone.utc)
        assert is_deprecation_valid("2025-03-01", release) is False

    def test_exact_six_months(self):
        release = datetime(2025, 1, 1, tzinfo=timezone.utc)
        assert is_deprecation_valid("2025-07-01", release) is True

    def test_invalid_date_string(self):
        release = datetime(2025, 1, 1, tzinfo=timezone.utc)
        assert is_deprecation_valid("not-a-date", release) is False
