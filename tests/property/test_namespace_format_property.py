"""Property-based tests for Namespace Format Validation.

**Validates: Requirements 13.2, 13.8**

Property 22: Namespace Format Validation
- Any string of 1-128 alphanumeric/hyphen/underscore characters is accepted.
- Empty strings, strings >128 characters, or strings containing other characters are rejected.
- Uses Hypothesis strategies for comprehensive input space coverage.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from clinical_reasoning_fabric.api.namespace import validate_namespace
from clinical_reasoning_fabric.models.exceptions import InvalidNamespaceError


# --- Valid namespace alphabet: lowercase, digits, hyphen, underscore ---
# Note: The pattern also accepts uppercase (A-Z), so we include it in tests.
VALID_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
VALID_ALPHABET_LOWER = "abcdefghijklmnopqrstuvwxyz0123456789-_"


# --- Strategies ---

def valid_namespace_strategy() -> st.SearchStrategy[str]:
    """Generate valid namespace strings: 1-128 chars from [a-zA-Z0-9_-]."""
    return st.text(
        alphabet=VALID_ALPHABET,
        min_size=1,
        max_size=128,
    )


def valid_namespace_lowercase_strategy() -> st.SearchStrategy[str]:
    """Generate valid namespace strings using only lowercase alphabet."""
    return st.text(
        alphabet=VALID_ALPHABET_LOWER,
        min_size=1,
        max_size=128,
    )


def too_long_namespace_strategy() -> st.SearchStrategy[str]:
    """Generate namespace strings that exceed 128 characters."""
    return st.text(
        alphabet=VALID_ALPHABET,
        min_size=129,
        max_size=300,
    )


def invalid_char_namespace_strategy() -> st.SearchStrategy[str]:
    """Generate non-empty strings containing at least one invalid character."""
    # Characters not in the valid set
    invalid_chars = " !@#$%^&*()+=[]{}|\\;:'\",.<>?/`~\t\n"

    @st.composite
    def build_invalid(draw):
        # Start with valid prefix (possibly empty)
        prefix = draw(st.text(alphabet=VALID_ALPHABET, min_size=0, max_size=10))
        # At least one invalid character
        bad_char = draw(st.sampled_from(list(invalid_chars)))
        # Optional valid suffix
        suffix = draw(st.text(alphabet=VALID_ALPHABET, min_size=0, max_size=10))
        result = prefix + bad_char + suffix
        # Ensure total length is <= 128 to isolate the "invalid char" reason
        return result[:128] if len(result) > 128 else result

    return build_invalid()


# --- Property Tests ---


@pytest.mark.property
class TestNamespaceFormatValidation:
    """Property 22: Namespace Format Validation.

    **Validates: Requirements 13.2, 13.8**

    Tests that:
    - Any string of 1-128 alphanumeric/hyphen/underscore characters is accepted.
    - Empty strings are rejected with InvalidNamespaceError.
    - Strings exceeding 128 characters are rejected with InvalidNamespaceError.
    - Strings containing characters outside [a-zA-Z0-9_-] are rejected.
    """

    @given(namespace=valid_namespace_strategy())
    def test_valid_namespace_always_accepted(self, namespace: str):
        """Any string of 1-128 alphanumeric/hyphen/underscore chars is accepted."""
        result = validate_namespace(namespace)
        assert result is True, (
            f"Valid namespace '{namespace}' (len={len(namespace)}) was rejected"
        )

    @given(namespace=valid_namespace_lowercase_strategy())
    def test_valid_lowercase_namespace_accepted(self, namespace: str):
        """Valid lowercase-only namespaces are accepted."""
        result = validate_namespace(namespace)
        assert result is True, (
            f"Valid lowercase namespace '{namespace}' was rejected"
        )

    @given(namespace=too_long_namespace_strategy())
    def test_too_long_namespace_always_rejected(self, namespace: str):
        """Strings exceeding 128 characters are always rejected."""
        with pytest.raises(InvalidNamespaceError) as exc_info:
            validate_namespace(namespace)
        assert exc_info.value.namespace == namespace
        assert "128" in exc_info.value.reason

    def test_empty_namespace_always_rejected(self):
        """Empty string is rejected with InvalidNamespaceError."""
        with pytest.raises(InvalidNamespaceError) as exc_info:
            validate_namespace("")
        assert "empty" in exc_info.value.reason.lower()

    @given(namespace=invalid_char_namespace_strategy())
    def test_invalid_characters_always_rejected(self, namespace: str):
        """Strings with characters outside [a-zA-Z0-9_-] are rejected."""
        with pytest.raises(InvalidNamespaceError):
            validate_namespace(namespace)

    @given(length=st.integers(min_value=1, max_value=128))
    def test_any_valid_length_accepted(self, length: int):
        """Namespaces of any length from 1 to 128 (with valid chars) are accepted."""
        namespace = "a" * length
        result = validate_namespace(namespace)
        assert result is True, (
            f"Namespace of length {length} with valid chars was rejected"
        )

    @given(
        namespace=st.text(
            alphabet=VALID_ALPHABET,
            min_size=1,
            max_size=128,
        )
    )
    def test_boundary_128_accepted(self, namespace: str):
        """Namespaces at exactly the maximum boundary (128 chars) are accepted."""
        # Pad or trim to exactly 128
        ns_128 = (namespace * ((128 // len(namespace)) + 1))[:128]
        result = validate_namespace(ns_128)
        assert result is True, (
            f"Namespace at exactly 128 chars was rejected"
        )

    def test_exactly_129_chars_rejected(self):
        """A namespace of exactly 129 valid characters is rejected."""
        namespace = "a" * 129
        with pytest.raises(InvalidNamespaceError) as exc_info:
            validate_namespace(namespace)
        assert "128" in exc_info.value.reason

    @given(data=st.data())
    def test_arbitrary_text_classified_correctly(self, data):
        """For any arbitrary string, it is either accepted (valid) or rejected (invalid)."""
        text = data.draw(st.text(min_size=0, max_size=200))

        import re
        pattern = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")
        expected_valid = bool(pattern.match(text))

        if expected_valid:
            result = validate_namespace(text)
            assert result is True, (
                f"Text '{text}' matches pattern but was rejected"
            )
        else:
            with pytest.raises(InvalidNamespaceError):
                validate_namespace(text)
