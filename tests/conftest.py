"""Shared test configuration and fixtures for Clinical Reasoning Fabric tests."""

import os

from hypothesis import settings, HealthCheck

# Register Hypothesis profiles
settings.register_profile(
    "default",
    max_examples=100,
    deadline=10000,
    suppress_health_check=[HealthCheck.too_slow],
)

settings.register_profile(
    "ci",
    max_examples=500,
    deadline=30000,
    suppress_health_check=[HealthCheck.too_slow],
)

settings.register_profile(
    "dev",
    max_examples=20,
    deadline=5000,
    suppress_health_check=[HealthCheck.too_slow],
)

# Load the profile from HYPOTHESIS_PROFILE env var, defaulting to "default"
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "default"))
