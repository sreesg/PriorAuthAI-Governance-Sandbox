"""API versioning router with semantic versioning and backwards compatibility.

Routes requests to the correct handler version based on request version header.
Maintains backwards compatibility for prior major version for minimum 6 months
after a new major version is released. Returns 410 Gone for unsupported versions.

Requirements referenced: 13.6
- 13.6: Maintain a versioned API contract with semantic versioning where breaking
         changes increment the major version and non-breaking additions increment
         the minor version; support prior major version for minimum 6 months after
         a new major version is released.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

SEMVER_PATTERN = re.compile(
    r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)$"
)

MIN_DEPRECATION_MONTHS = 6


# =============================================================================
# Exceptions
# =============================================================================


class VersionGoneError(Exception):
    """Raised when a requested API version is no longer supported (HTTP 410 Gone).

    Attributes:
        requested_version: The version requested by the client.
        supported_versions: List of currently supported versions.
        message: Human-readable description of the error.
    """

    def __init__(
        self,
        requested_version: str,
        supported_versions: list[str],
        message: Optional[str] = None,
    ):
        self.requested_version = requested_version
        self.supported_versions = supported_versions
        self.message = message or (
            f"API version '{requested_version}' is no longer supported. "
            f"Supported versions: {supported_versions}"
        )
        super().__init__(self.message)


class InvalidVersionError(Exception):
    """Raised when a version string does not conform to semantic versioning.

    Attributes:
        version_string: The invalid version string provided.
        message: Human-readable description of the error.
    """

    def __init__(self, version_string: str, message: Optional[str] = None):
        self.version_string = version_string
        self.message = message or (
            f"Invalid version format: '{version_string}'. "
            f"Expected semantic versioning format: major.minor.patch"
        )
        super().__init__(self.message)


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class APIVersionInfo:
    """Describes the versioning state of the API.

    Attributes:
        current_version: The current (latest) API version (e.g., "2.0.0").
        supported_versions: List of all currently supported version strings.
        deprecation_schedule: Mapping of version to deprecation date (ISO format).
            Example: {"1.0.0": "2025-12-31"} means version 1.0.0 is deprecated
            and support will be removed after 2025-12-31.
    """

    current_version: str
    supported_versions: list[str] = field(default_factory=list)
    deprecation_schedule: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the version info on construction."""
        if not parse_semver(self.current_version):
            raise InvalidVersionError(self.current_version)
        for v in self.supported_versions:
            if not parse_semver(v):
                raise InvalidVersionError(v)
        # Ensure current_version is in supported_versions
        if self.current_version not in self.supported_versions:
            self.supported_versions.append(self.current_version)


# =============================================================================
# Helper Functions
# =============================================================================


def parse_semver(version: str) -> Optional[tuple[int, int, int]]:
    """Parse a semantic version string into (major, minor, patch) tuple.

    Args:
        version: A string in the format "major.minor.patch".

    Returns:
        Tuple of (major, minor, patch) integers if valid, None otherwise.
    """
    match = SEMVER_PATTERN.match(version)
    if not match:
        return None
    return (
        int(match.group("major")),
        int(match.group("minor")),
        int(match.group("patch")),
    )


def get_major_version(version: str) -> Optional[int]:
    """Extract the major version number from a semver string.

    Args:
        version: A semantic version string.

    Returns:
        The major version as an integer, or None if the string is invalid.
    """
    parsed = parse_semver(version)
    return parsed[0] if parsed else None


def is_deprecation_valid(
    deprecation_date_str: str, release_date: datetime, min_months: int = MIN_DEPRECATION_MONTHS
) -> bool:
    """Check if a deprecation date provides at least min_months of support after release.

    Args:
        deprecation_date_str: ISO date string for when support ends (e.g., "2025-12-31").
        release_date: When the new major version was released.
        min_months: Minimum number of months required (default 6).

    Returns:
        True if the deprecation date is at least min_months after the release date.
    """
    try:
        deprecation_date = datetime.fromisoformat(deprecation_date_str)
        if deprecation_date.tzinfo is None:
            deprecation_date = deprecation_date.replace(tzinfo=timezone.utc)
        if release_date.tzinfo is None:
            release_date = release_date.replace(tzinfo=timezone.utc)

        # Calculate months difference
        month_diff = (
            (deprecation_date.year - release_date.year) * 12
            + (deprecation_date.month - release_date.month)
        )
        return month_diff >= min_months
    except (ValueError, TypeError):
        return False


# =============================================================================
# API Version Router
# =============================================================================


class APIVersionRouter:
    """Routes API requests to the correct handler version based on version header.

    Maintains backwards compatibility for the prior major version for a minimum
    of 6 months after a new major version is released. Returns HTTP 410 Gone
    for unsupported versions.

    The router maps (version_major, endpoint) pairs to handler callables.
    Minor/patch versions are resolved within their major version family.

    Usage:
        version_info = APIVersionInfo(
            current_version="2.0.0",
            supported_versions=["1.0.0", "2.0.0"],
            deprecation_schedule={"1.0.0": "2025-12-31"},
        )
        router = APIVersionRouter(version_info)

        # Register handlers for different versions
        router.register_handler("1", "ingest", v1_ingest_handler)
        router.register_handler("2", "ingest", v2_ingest_handler)

        # Route a request
        handler = router.route_request("1.2.0", "ingest")

    Requirement 13.6:
        - Semantic versioning: breaking changes → major, additions → minor.
        - Prior major version supported for minimum 6 months after new major release.
    """

    def __init__(self, version_info: APIVersionInfo):
        """Initialize the version router.

        Args:
            version_info: API versioning state including current version,
                supported versions, and deprecation schedule.

        Raises:
            InvalidVersionError: If version_info contains invalid version strings.
        """
        self.info = version_info
        # Map of (major_version_str, endpoint) → handler callable
        self._handlers: dict[tuple[str, str], Callable[..., Any]] = {}

    def register_handler(
        self, major_version: str, endpoint: str, handler: Callable[..., Any]
    ) -> None:
        """Register a handler for a specific major version and endpoint.

        Args:
            major_version: The major version string (e.g., "1", "2").
            endpoint: The endpoint name (e.g., "ingest", "retrieve", "verify").
            handler: The callable that handles requests for this version/endpoint.
        """
        self._handlers[(major_version, endpoint)] = handler
        logger.info(
            "Registered handler for v%s/%s", major_version, endpoint
        )

    def route_request(self, request_version: str, endpoint: str) -> Callable[..., Any]:
        """Route a request to the appropriate handler version.

        Resolves the request version to the correct major version handler.
        Minor and patch versions are handled by the same major version handler
        (backwards compatible within a major version).

        Args:
            request_version: The API version from the request header (e.g., "1.2.0").
            endpoint: The requested endpoint name (e.g., "ingest").

        Returns:
            The handler callable for the resolved version and endpoint.

        Raises:
            InvalidVersionError: If request_version is not valid semver.
            VersionGoneError: If the requested version is no longer supported (410 Gone).
            KeyError: If no handler is registered for the resolved version/endpoint.
        """
        # Parse and validate the request version
        parsed = parse_semver(request_version)
        if parsed is None:
            raise InvalidVersionError(request_version)

        major, minor, patch = parsed

        # Check if the version is supported
        if not self.is_version_supported(request_version):
            raise VersionGoneError(
                requested_version=request_version,
                supported_versions=self.info.supported_versions,
            )

        # Resolve to major version handler
        major_str = str(major)
        handler_key = (major_str, endpoint)

        if handler_key not in self._handlers:
            raise KeyError(
                f"No handler registered for version v{major_str}, "
                f"endpoint '{endpoint}'"
            )

        logger.debug(
            "Routing request version=%s endpoint=%s → handler v%s/%s",
            request_version,
            endpoint,
            major_str,
            endpoint,
        )

        return self._handlers[handler_key]

    def is_version_supported(self, version: str) -> bool:
        """Check if a version is currently supported.

        A version is supported if:
        1. Its major version matches any major version in the supported_versions list.
        2. If the version is in the deprecation schedule, the deprecation date has
           not yet passed.

        The prior major version remains supported for a minimum of 6 months after
        the new major version is released.

        Args:
            version: The semantic version string to check.

        Returns:
            True if the version is still supported, False otherwise.
        """
        parsed = parse_semver(version)
        if parsed is None:
            return False

        major = parsed[0]

        # Get all supported major versions
        supported_majors = set()
        for supported_v in self.info.supported_versions:
            sv_parsed = parse_semver(supported_v)
            if sv_parsed:
                supported_majors.add(sv_parsed[0])

        # Check if the major version is in the supported set
        if major not in supported_majors:
            return False

        # Check deprecation schedule — if deprecated and past the date, unsupported
        for deprecated_version, deprecation_date_str in self.info.deprecation_schedule.items():
            dep_parsed = parse_semver(deprecated_version)
            if dep_parsed and dep_parsed[0] == major:
                try:
                    deprecation_date = datetime.fromisoformat(deprecation_date_str)
                    if deprecation_date.tzinfo is None:
                        deprecation_date = deprecation_date.replace(tzinfo=timezone.utc)
                    now = datetime.now(timezone.utc)
                    if now > deprecation_date:
                        return False
                except (ValueError, TypeError):
                    # Invalid date format — treat as still supported (fail open)
                    pass

        return True

    def get_supported_versions(self) -> list[str]:
        """Return the list of currently supported API versions.

        Filters out any versions whose deprecation date has passed.

        Returns:
            List of supported version strings.
        """
        return [
            v for v in self.info.supported_versions
            if self.is_version_supported(v)
        ]

    def get_deprecation_date(self, version: str) -> Optional[str]:
        """Get the scheduled deprecation date for a version.

        Args:
            version: The version to check.

        Returns:
            ISO date string of deprecation, or None if not scheduled for deprecation.
        """
        parsed = parse_semver(version)
        if parsed is None:
            return None

        major = parsed[0]

        for deprecated_version, date_str in self.info.deprecation_schedule.items():
            dep_parsed = parse_semver(deprecated_version)
            if dep_parsed and dep_parsed[0] == major:
                return date_str

        return None

    def is_version_deprecated(self, version: str) -> bool:
        """Check if a version is deprecated (scheduled for removal) but still supported.

        A version is deprecated if it appears in the deprecation_schedule but
        the deprecation date has not yet passed.

        Args:
            version: The version to check.

        Returns:
            True if the version is deprecated but still supported.
        """
        parsed = parse_semver(version)
        if parsed is None:
            return False

        major = parsed[0]

        for deprecated_version, deprecation_date_str in self.info.deprecation_schedule.items():
            dep_parsed = parse_semver(deprecated_version)
            if dep_parsed and dep_parsed[0] == major:
                # It's in the deprecation schedule — check if still active
                try:
                    deprecation_date = datetime.fromisoformat(deprecation_date_str)
                    if deprecation_date.tzinfo is None:
                        deprecation_date = deprecation_date.replace(tzinfo=timezone.utc)
                    now = datetime.now(timezone.utc)
                    # Deprecated but not yet gone
                    return now <= deprecation_date
                except (ValueError, TypeError):
                    return False

        return False

    def validate_deprecation_schedule(
        self, new_major_release_date: datetime
    ) -> dict[str, bool]:
        """Validate that all deprecation dates provide at least 6 months of support.

        Ensures compliance with Requirement 13.6: prior major version must be
        supported for minimum 6 months after new major version release.

        Args:
            new_major_release_date: The date the new major version was released.

        Returns:
            Dict mapping deprecated versions to whether their deprecation date
            provides sufficient support time (True = compliant).
        """
        results: dict[str, bool] = {}
        for version, date_str in self.info.deprecation_schedule.items():
            results[version] = is_deprecation_valid(
                date_str, new_major_release_date, MIN_DEPRECATION_MONTHS
            )
        return results
