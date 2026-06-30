"""Property-based tests for Cryptographic Provenance Round-Trip.

**Validates: Requirements 1.3, 1.4**

Property 2: Cryptographic Provenance Round-Trip
- For any text string, hashing and signing then verifying produces a valid result.
- Same text always produces the same SHA-256 hash (determinism).
- Hash output is always a 64-character lowercase hex string.
- Different texts produce different hashes (with high probability).
- Signing the same hash always produces a non-empty signature.
"""

import asyncio
import base64
import hashlib
import re
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from hypothesis import given, assume, settings
from hypothesis import strategies as st

from clinical_reasoning_fabric.ingestion.document_ingestion_service import (
    DocumentIngestionService,
)
from clinical_reasoning_fabric.ingestion.pii_scrubber import PIIScrubber
from clinical_reasoning_fabric.models.core import KMSSignature


# --- Mock KMS client that simulates deterministic signing ---


class MockKMSClient:
    """A mock KMS client that produces deterministic signatures based on input.

    Uses HMAC-SHA256 with a fixed key to simulate KMS signing behavior,
    ensuring the same input always produces the same signature.
    """

    KEY_ID = "arn:aws:kms:us-east-1:123456789012:key/mock-key-id"
    FIXED_SECRET = b"mock-kms-signing-secret-for-tests"

    def sign(self, KeyId: str, Message: bytes, MessageType: str, SigningAlgorithm: str) -> dict:
        """Simulate KMS sign operation deterministically."""
        import hmac

        # Produce a deterministic signature using HMAC
        signature_bytes = hmac.new(
            self.FIXED_SECRET, Message, hashlib.sha256
        ).digest()

        return {
            "KeyId": self.KEY_ID,
            "Signature": signature_bytes,
            "SigningAlgorithm": SigningAlgorithm,
        }


def _create_service() -> DocumentIngestionService:
    """Create a DocumentIngestionService with mock dependencies for testing."""
    mock_kms = MockKMSClient()
    mock_qdrant = MagicMock()
    pii_scrubber = PIIScrubber()
    mock_chunker = MagicMock()

    return DocumentIngestionService(
        kms_client=mock_kms,
        qdrant_client=mock_qdrant,
        pii_scrubber=pii_scrubber,
        chunker=mock_chunker,
        kms_key_id=MockKMSClient.KEY_ID,
    )


def _run_async(coro):
    """Helper to run async function synchronously in tests."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# --- Hypothesis strategies ---


def arbitrary_text_strategy() -> st.SearchStrategy[str]:
    """Generate arbitrary non-empty text strings for hashing."""
    return st.text(min_size=1, max_size=5000)


def clinical_text_strategy() -> st.SearchStrategy[str]:
    """Generate realistic clinical text strings."""
    prefixes = [
        "Patient presents with",
        "Assessment indicates",
        "Diagnosis confirmed:",
        "Treatment plan includes",
        "Lab results show",
        "Imaging findings reveal",
        "Clinical note:",
        "History of present illness:",
    ]
    conditions = [
        "chronic obstructive pulmonary disease",
        "type 2 diabetes mellitus",
        "congestive heart failure NYHA class III",
        "metastatic non-small cell lung cancer",
        "bilateral knee osteoarthritis",
        "lumbar spinal stenosis",
        "atrial fibrillation with rapid ventricular response",
        "major depressive disorder recurrent",
    ]
    return st.builds(
        lambda p, c: f"{p} {c}",
        st.sampled_from(prefixes),
        st.sampled_from(conditions),
    )


# --- Property tests ---


@pytest.mark.property
class TestCryptographicProvenanceRoundTrip:
    """Property 2: Cryptographic Provenance Round-Trip.

    **Validates: Requirements 1.3, 1.4**

    Tests that hashing and signing operations maintain cryptographic integrity:
    - Determinism: same text always yields same hash
    - Format: hash is always 64 lowercase hex characters
    - Round-trip: compute hash then verify it matches re-computation
    - Uniqueness: different texts produce different hashes
    - Signing consistency: signing same hash produces a non-empty signature
    """

    service = _create_service()

    @given(text=arbitrary_text_strategy())
    @settings(max_examples=100)
    def test_hash_determinism(self, text: str):
        """Same text always produces the same SHA-256 hash.

        **Validates: Requirements 1.3**

        For any input text, calling compute_hash multiple times must return
        the identical hash value every time.
        """
        hash1 = _run_async(self.service.compute_hash(text))
        hash2 = _run_async(self.service.compute_hash(text))
        assert hash1 == hash2, (
            f"Hash is not deterministic: '{hash1}' != '{hash2}' for same input"
        )

    @given(text=arbitrary_text_strategy())
    @settings(max_examples=100)
    def test_hash_format_always_64_hex(self, text: str):
        """Hash output is always exactly 64 lowercase hexadecimal characters.

        **Validates: Requirements 1.3**

        SHA-256 produces 256 bits = 32 bytes = 64 hex characters.
        """
        content_hash = _run_async(self.service.compute_hash(text))

        # Must be exactly 64 characters
        assert len(content_hash) == 64, (
            f"Hash length is {len(content_hash)}, expected 64"
        )
        # Must contain only valid hex characters (lowercase)
        assert re.fullmatch(r"[0-9a-f]{64}", content_hash), (
            f"Hash '{content_hash}' is not a valid 64-char lowercase hex string"
        )

    @given(text=arbitrary_text_strategy())
    @settings(max_examples=100)
    def test_hash_round_trip_verification(self, text: str):
        """For any text, computing hash then re-computing matches.

        **Validates: Requirements 1.3**

        This verifies the round-trip property: hash(text) == hash(text),
        and that the hash matches direct hashlib computation.
        """
        # Compute via service
        service_hash = _run_async(self.service.compute_hash(text))

        # Compute directly with hashlib for verification
        direct_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        assert service_hash == direct_hash, (
            f"Service hash '{service_hash}' doesn't match direct computation '{direct_hash}'"
        )

    @given(text1=arbitrary_text_strategy(), text2=arbitrary_text_strategy())
    @settings(max_examples=100)
    def test_hash_uniqueness_different_texts(self, text1: str, text2: str):
        """Different texts produce different hashes (collision resistance).

        **Validates: Requirements 1.3**

        SHA-256 is collision-resistant, so distinct inputs should produce
        distinct outputs with overwhelming probability.
        """
        assume(text1 != text2)

        hash1 = _run_async(self.service.compute_hash(text1))
        hash2 = _run_async(self.service.compute_hash(text2))

        assert hash1 != hash2, (
            f"Hash collision detected: both inputs produced '{hash1}'"
        )

    @given(text=arbitrary_text_strategy())
    @settings(max_examples=100)
    def test_signing_produces_non_empty_signature(self, text: str):
        """Signing any hash always produces a non-empty signature.

        **Validates: Requirements 1.4**

        For any text, compute its hash and then sign it. The resulting
        KMSSignature must have a non-empty signature field, a key_id,
        algorithm, and timestamp.
        """
        content_hash = _run_async(self.service.compute_hash(text))
        kms_signature = _run_async(self.service.sign_hash(content_hash))

        # Signature must be a valid KMSSignature
        assert isinstance(kms_signature, KMSSignature)
        assert kms_signature.signature, "Signature must not be empty"
        assert len(kms_signature.signature) > 0, "Signature must have content"
        assert kms_signature.key_id, "Key ID must not be empty"
        assert kms_signature.algorithm == "RSASSA_PKCS1_V1_5_SHA_256"
        assert kms_signature.signed_at is not None

    @given(text=arbitrary_text_strategy())
    @settings(max_examples=100)
    def test_signing_consistency_same_hash(self, text: str):
        """Signing the same hash consistently produces a valid base64 signature.

        **Validates: Requirements 1.4**

        For any text, the signing operation on the same hash should produce
        a consistent, decodable base64 signature.
        """
        content_hash = _run_async(self.service.compute_hash(text))
        sig1 = _run_async(self.service.sign_hash(content_hash))
        sig2 = _run_async(self.service.sign_hash(content_hash))

        # Both signatures should be valid base64
        try:
            decoded1 = base64.b64decode(sig1.signature)
            decoded2 = base64.b64decode(sig2.signature)
        except Exception as e:
            pytest.fail(f"Signature is not valid base64: {e}")

        assert len(decoded1) > 0, "Decoded signature must not be empty"
        assert len(decoded2) > 0, "Decoded signature must not be empty"

        # Same hash should produce same deterministic signature
        assert sig1.signature == sig2.signature, (
            "Signing the same hash should produce the same signature"
        )

    @given(text=clinical_text_strategy())
    @settings(max_examples=50)
    def test_full_provenance_chain(self, text: str):
        """Full provenance chain: text → hash → sign produces valid, linked artifacts.

        **Validates: Requirements 1.3, 1.4**

        For any clinical text, the complete chain of hash then sign produces:
        - A valid 64-char hex hash
        - A non-empty base64 signature
        - Signature timestamp is a valid datetime
        - The chain is reproducible (same text → same hash → same signature)
        """
        # First pass
        hash1 = _run_async(self.service.compute_hash(text))
        sig1 = _run_async(self.service.sign_hash(hash1))

        # Second pass - verify reproducibility
        hash2 = _run_async(self.service.compute_hash(text))
        sig2 = _run_async(self.service.sign_hash(hash2))

        # Hash must be consistent
        assert hash1 == hash2

        # Hash format
        assert re.fullmatch(r"[0-9a-f]{64}", hash1)

        # Signature must be valid
        assert sig1.signature == sig2.signature
        assert sig1.key_id == sig2.key_id
        assert sig1.algorithm == sig2.algorithm

        # Verify base64 decodability
        decoded = base64.b64decode(sig1.signature)
        assert len(decoded) > 0
