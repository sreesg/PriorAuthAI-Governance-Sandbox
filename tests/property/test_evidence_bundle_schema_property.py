"""Property-based tests for Evidence Bundle Schema Conformance.

**Validates: Requirements 10.1, 10.2, 10.3, 10.4, 8.3**

Property 16: Evidence Bundle Schema Conformance
- Any produced bundle has all required fields present and non-null
- lineage_trail has >= 1 entry, each with non-empty conclusion, evidence_id,
  and valid retrieval_timestamp
- original_document_signatures has >= 1 valid KMS signature with non-empty
  key_id and signature
- execution_trace is present when provided
- Bundle is a valid Pydantic model (re-validation succeeds)
"""

from datetime import datetime, timezone, timedelta

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from clinical_reasoning_fabric.beacon.evidence_bundle_service import (
    EvidenceBundleService,
)
from clinical_reasoning_fabric.models.core import (
    EvidenceBundle,
    KMSSignature,
    LineageEntry,
    TraceCategory,
    TraceEntry,
)


# =============================================================================
# Hypothesis Strategies
# =============================================================================

# Non-empty text strategy for string fields
non_empty_text = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_",
    min_size=1,
    max_size=50,
)

# Strategy for valid datetime values
valid_datetime = st.datetimes(
    min_value=datetime(2000, 1, 1),
    max_value=datetime(2030, 12, 31),
    timezones=st.just(timezone.utc),
)


@st.composite
def lineage_entry_strategy(draw):
    """Generate a valid LineageEntry with non-empty fields and valid timestamp."""
    conclusion = draw(non_empty_text)
    evidence_id = draw(non_empty_text)
    retrieval_timestamp = draw(valid_datetime)
    return LineageEntry(
        conclusion=conclusion,
        evidence_id=evidence_id,
        retrieval_timestamp=retrieval_timestamp,
    )


@st.composite
def lineage_trail_strategy(draw):
    """Generate a list of 1-10 valid LineageEntry instances."""
    num_entries = draw(st.integers(min_value=1, max_value=10))
    entries = [draw(lineage_entry_strategy()) for _ in range(num_entries)]
    return entries


@st.composite
def kms_signature_strategy(draw):
    """Generate a valid KMSSignature with non-empty key_id and signature."""
    key_id = draw(non_empty_text)
    signature = draw(non_empty_text)
    algorithm = draw(st.sampled_from([
        "RSASSA_PKCS1_V1_5_SHA_256",
        "RSASSA_PSS_SHA_256",
        "ECDSA_SHA_256",
    ]))
    signed_at = draw(valid_datetime)
    return KMSSignature(
        key_id=key_id,
        signature=signature,
        algorithm=algorithm,
        signed_at=signed_at,
    )


@st.composite
def document_signatures_strategy(draw):
    """Generate a list of 1-5 valid KMSSignature instances."""
    num_sigs = draw(st.integers(min_value=1, max_value=5))
    sigs = [draw(kms_signature_strategy()) for _ in range(num_sigs)]
    return sigs


@st.composite
def trace_entry_strategy(draw, seq_num: int = 0, request_id: str = "req-001"):
    """Generate a valid TraceEntry."""
    category = draw(st.sampled_from(list(TraceCategory)))
    identity_id = draw(non_empty_text)
    # Generate a valid UTC ISO-8601 timestamp with ms precision
    dt = draw(valid_datetime)
    timestamp = dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"
    return TraceEntry(
        sequence_number=seq_num,
        timestamp=timestamp,
        request_id=request_id,
        identity_id=identity_id,
        category=category,
    )


@st.composite
def execution_trace_strategy(draw):
    """Generate a valid execution trace with 1-5 ordered entries."""
    num_entries = draw(st.integers(min_value=1, max_value=5))
    request_id = draw(non_empty_text)
    entries = []
    for i in range(num_entries):
        entry = draw(trace_entry_strategy(seq_num=i, request_id=request_id))
        entries.append(entry)
    return entries


@st.composite
def evidence_bundle_inputs_strategy(draw):
    """Generate valid inputs for produce_bundle."""
    execution_id = draw(non_empty_text)
    decision = draw(st.sampled_from(["approve", "escalate"]))
    reason = draw(non_empty_text)
    lineage_trail = draw(lineage_trail_strategy())
    document_signatures = draw(document_signatures_strategy())
    # Optionally include execution trace
    include_trace = draw(st.booleans())
    execution_trace = draw(execution_trace_strategy()) if include_trace else None
    return {
        "execution_id": execution_id,
        "decision": decision,
        "reason": reason,
        "lineage_trail": lineage_trail,
        "document_signatures": document_signatures,
        "execution_trace": execution_trace,
    }


# =============================================================================
# Property 16: Evidence Bundle Schema Conformance
# =============================================================================


@pytest.mark.property
class TestEvidenceBundleSchemaConformance:
    """Property 16: Evidence Bundle Schema Conformance.

    **Validates: Requirements 10.1, 10.2, 10.3, 10.4, 8.3**

    For any valid combination of inputs to produce_bundle:
    - Produced bundle has all required fields present and non-null
    - lineage_trail has >= 1 entry, each with non-empty conclusion,
      evidence_id, and valid retrieval_timestamp
    - original_document_signatures has >= 1 signature with non-empty
      key_id and signature
    - execution_trace is present when provided
    - Bundle is a valid Pydantic model (re-validation succeeds)
    """

    @given(inputs=evidence_bundle_inputs_strategy())
    @settings(max_examples=200)
    def test_all_required_fields_present_and_non_null(self, inputs):
        """Produced bundle has all required fields present and non-null.

        **Validates: Requirements 10.1**

        The Evidence_Bundle must conform to the defined output schema
        containing execution_id, decision, reason, lineage_trail, and
        original_document_signatures — all non-null.
        """
        service = EvidenceBundleService()
        bundle = service.produce_bundle(**inputs)

        # All required fields are present and non-null
        assert bundle.execution_id is not None, "execution_id must be non-null"
        assert bundle.decision is not None, "decision must be non-null"
        assert bundle.reason is not None, "reason must be non-null"
        assert bundle.lineage_trail is not None, "lineage_trail must be non-null"
        assert bundle.original_document_signatures is not None, (
            "original_document_signatures must be non-null"
        )

        # All required fields are non-empty strings
        assert len(bundle.execution_id) > 0, "execution_id must be non-empty"
        assert len(bundle.decision) > 0, "decision must be non-empty"
        assert len(bundle.reason) > 0, "reason must be non-empty"

    @given(inputs=evidence_bundle_inputs_strategy())
    @settings(max_examples=200)
    def test_lineage_trail_has_valid_entries(self, inputs):
        """lineage_trail has >= 1 entry, each with valid fields.

        **Validates: Requirements 10.2**

        The lineage_trail is an ordered array of entries, where each entry
        contains a non-empty conclusion statement, evidence_id, and valid
        retrieval_timestamp.
        """
        service = EvidenceBundleService()
        bundle = service.produce_bundle(**inputs)

        # lineage_trail has >= 1 entry
        assert len(bundle.lineage_trail) >= 1, (
            "lineage_trail must have at least 1 entry"
        )

        # Each entry has required fields
        for i, entry in enumerate(bundle.lineage_trail):
            assert entry.conclusion is not None and len(entry.conclusion.strip()) > 0, (
                f"lineage_trail[{i}].conclusion must be non-empty"
            )
            assert entry.evidence_id is not None and len(entry.evidence_id.strip()) > 0, (
                f"lineage_trail[{i}].evidence_id must be non-empty"
            )
            assert entry.retrieval_timestamp is not None, (
                f"lineage_trail[{i}].retrieval_timestamp must not be null"
            )
            assert isinstance(entry.retrieval_timestamp, datetime), (
                f"lineage_trail[{i}].retrieval_timestamp must be a datetime"
            )

    @given(inputs=evidence_bundle_inputs_strategy())
    @settings(max_examples=200)
    def test_signatures_have_valid_entries(self, inputs):
        """original_document_signatures has >= 1 signature with valid fields.

        **Validates: Requirements 10.3**

        The original_document_signatures array contains KMS signatures for
        every source document, each with non-empty key_id and signature.
        """
        service = EvidenceBundleService()
        bundle = service.produce_bundle(**inputs)

        # signatures has >= 1 entry
        assert len(bundle.original_document_signatures) >= 1, (
            "original_document_signatures must have at least 1 signature"
        )

        # Each signature has required fields
        for i, sig in enumerate(bundle.original_document_signatures):
            assert sig.key_id is not None and len(sig.key_id.strip()) > 0, (
                f"original_document_signatures[{i}].key_id must be non-empty"
            )
            assert sig.signature is not None and len(sig.signature.strip()) > 0, (
                f"original_document_signatures[{i}].signature must be non-empty"
            )
            assert sig.algorithm is not None and len(sig.algorithm.strip()) > 0, (
                f"original_document_signatures[{i}].algorithm must be non-empty"
            )
            assert sig.signed_at is not None, (
                f"original_document_signatures[{i}].signed_at must not be null"
            )
            assert isinstance(sig.signed_at, datetime), (
                f"original_document_signatures[{i}].signed_at must be a datetime"
            )

    @given(inputs=evidence_bundle_inputs_strategy())
    @settings(max_examples=200)
    def test_execution_trace_present_when_provided(self, inputs):
        """execution_trace is present in bundle when provided as input.

        **Validates: Requirements 8.3**

        When a PA decision is produced, the execution trace containing all
        entries from request initiation through decision production is
        attached to the Evidence_Bundle.
        """
        service = EvidenceBundleService()
        bundle = service.produce_bundle(**inputs)

        if inputs["execution_trace"] is not None:
            # Trace was provided — must be attached to bundle
            assert bundle.execution_trace is not None, (
                "execution_trace must be present in bundle when provided"
            )
            assert len(bundle.execution_trace) == len(inputs["execution_trace"]), (
                "execution_trace length must match provided trace"
            )
            # Verify trace entries have valid structure
            for i, entry in enumerate(bundle.execution_trace):
                assert entry.sequence_number == i, (
                    f"execution_trace[{i}].sequence_number must be {i}"
                )
                assert entry.request_id is not None and len(entry.request_id) > 0, (
                    f"execution_trace[{i}].request_id must be non-empty"
                )
                assert entry.identity_id is not None and len(entry.identity_id) > 0, (
                    f"execution_trace[{i}].identity_id must be non-empty"
                )
                assert isinstance(entry.category, TraceCategory), (
                    f"execution_trace[{i}].category must be a valid TraceCategory"
                )

    @given(inputs=evidence_bundle_inputs_strategy())
    @settings(max_examples=200)
    def test_bundle_is_valid_pydantic_model(self, inputs):
        """Bundle is a valid Pydantic model — re-validation succeeds.

        **Validates: Requirements 10.4**

        The Evidence_Bundle must pass schema validation: all required fields
        present, non-null, correct data type, lineage_trail >= 1 entry,
        signatures >= 1 signature.
        """
        service = EvidenceBundleService()
        bundle = service.produce_bundle(**inputs)

        # Bundle must be an instance of EvidenceBundle
        assert isinstance(bundle, EvidenceBundle), (
            "produce_bundle must return an EvidenceBundle instance"
        )

        # Re-validate by creating a new instance from model dump
        # This confirms the bundle conforms to the Pydantic schema
        bundle_dict = bundle.model_dump()
        try:
            revalidated = EvidenceBundle.model_validate(bundle_dict)
        except ValidationError as e:
            pytest.fail(
                f"Bundle re-validation failed: {e}. "
                f"Bundle dict: {bundle_dict}"
            )

        # Revalidated bundle should be equivalent
        assert revalidated.execution_id == bundle.execution_id
        assert revalidated.decision == bundle.decision
        assert revalidated.reason == bundle.reason
        assert len(revalidated.lineage_trail) == len(bundle.lineage_trail)
        assert len(revalidated.original_document_signatures) == len(
            bundle.original_document_signatures
        )
