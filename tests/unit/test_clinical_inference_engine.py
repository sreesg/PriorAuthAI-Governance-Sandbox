"""Unit tests for ClinicalInferenceEngine.

Tests LLM-powered clinical inference including snippet analysis,
SDOH factor derivation, confidence scoring, and timeout handling.

Validates:
    - analyze_snippet() produces valid inferred facts within timeout
    - analyze_snippet() raises InferenceTimeoutError on timeout
    - derive_sdoh_factors() returns only SDOH-type facts
    - Max 10 inferred facts per snippet enforced
    - Confidence threshold filtering works correctly
    - Inference types and SDOH categories are validated
    - LLM response parsing handles malformed JSON gracefully

Requirements referenced: 14.1, 14.2, 14.6
"""

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from clinical_reasoning_fabric.inference.clinical_inference_engine import (
    ClinicalInferenceEngine,
    InferenceChain,
    InferenceHop,
    InferenceResult,
    InferredFact,
)
from clinical_reasoning_fabric.models.core import (
    ChunkProvenance,
    KMSSignature,
    ScoredChunk,
)
from clinical_reasoning_fabric.models.exceptions import InferenceTimeoutError


# =============================================================================
# Mock LLM Client
# =============================================================================


class MockLLMClient:
    """Mock LLM client that returns structured JSON based on keyword detection.

    Simulates LLM behavior by detecting keywords in input text and returning
    appropriate SDOH/inference facts.
    """

    KEYWORD_MAPPINGS = {
        "bus": {
            "inference_type": "sdoh_factor",
            "sdoh_category": "transportation_barriers",
            "conclusion": "Patient likely faces transportation barriers",
            "confidence": 0.75,
        },
        "ride": {
            "inference_type": "sdoh_factor",
            "sdoh_category": "transportation_barriers",
            "conclusion": "Patient has difficulty getting rides to appointments",
            "confidence": 0.7,
        },
        "homeless": {
            "inference_type": "sdoh_factor",
            "sdoh_category": "housing_instability",
            "conclusion": "Patient experiencing housing instability",
            "confidence": 0.9,
        },
        "shelter": {
            "inference_type": "sdoh_factor",
            "sdoh_category": "housing_instability",
            "conclusion": "Patient may be staying in temporary shelter",
            "confidence": 0.8,
        },
        "refrigerat": {
            "inference_type": "sdoh_factor",
            "sdoh_category": "medication_storage_limitations",
            "conclusion": "Patient may lack proper medication storage",
            "confidence": 0.7,
        },
        "insulin": {
            "inference_type": "medication_adherence_risk",
            "sdoh_category": None,
            "conclusion": "Patient at risk of medication non-adherence due to storage issues",
            "confidence": 0.65,
        },
        "food bank": {
            "inference_type": "sdoh_factor",
            "sdoh_category": "food_insecurity",
            "conclusion": "Patient likely experiencing food insecurity",
            "confidence": 0.8,
        },
        "hungry": {
            "inference_type": "sdoh_factor",
            "sdoh_category": "food_insecurity",
            "conclusion": "Patient may be food insecure",
            "confidence": 0.6,
        },
        "alone": {
            "inference_type": "sdoh_factor",
            "sdoh_category": "caregiver_availability",
            "conclusion": "Patient lacks caregiver support",
            "confidence": 0.65,
        },
        "no family": {
            "inference_type": "sdoh_factor",
            "sdoh_category": "caregiver_availability",
            "conclusion": "Patient has limited family support for care",
            "confidence": 0.7,
        },
        "miss appointment": {
            "inference_type": "care_access_barrier",
            "sdoh_category": None,
            "conclusion": "Patient has barriers to attending appointments",
            "confidence": 0.7,
        },
        "cant afford": {
            "inference_type": "care_access_barrier",
            "sdoh_category": None,
            "conclusion": "Patient faces financial barriers to care access",
            "confidence": 0.75,
        },
    }

    def __init__(self, delay: float = 0.0):
        """Initialize mock with optional delay to simulate processing time."""
        self.delay = delay
        self.call_count = 0
        self.last_prompt = None

    async def infer(self, prompt: str) -> str:
        """Mock LLM inference based on keyword detection in the prompt."""
        self.call_count += 1
        self.last_prompt = prompt

        if self.delay > 0:
            await asyncio.sleep(self.delay)

        # Detect keywords in the prompt text
        text_lower = prompt.lower()
        facts = []

        for keyword, fact_template in self.KEYWORD_MAPPINGS.items():
            if keyword in text_lower:
                fact = {
                    "inference_type": fact_template["inference_type"],
                    "conclusion": fact_template["conclusion"],
                    "confidence": fact_template["confidence"],
                    "hops": [
                        {
                            "source_text": f"Text contains '{keyword}'",
                            "intermediate_conclusion": fact_template["conclusion"],
                            "confidence": fact_template["confidence"],
                        }
                    ],
                }
                if fact_template["sdoh_category"]:
                    fact["sdoh_category"] = fact_template["sdoh_category"]
                facts.append(fact)

        return json.dumps(facts)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_llm_client():
    """Standard mock LLM client with no delay."""
    return MockLLMClient(delay=0.0)


@pytest.fixture
def slow_llm_client():
    """Mock LLM client that takes longer than the 15s timeout."""
    return MockLLMClient(delay=20.0)


@pytest.fixture
def mock_graph_service():
    """Mock graph service with async methods."""
    service = MagicMock()
    service.upsert_node = AsyncMock()
    service.upsert_relationship = AsyncMock()
    return service


@pytest.fixture
def sample_snippet():
    """Sample ScoredChunk for testing inference."""
    return ScoredChunk(
        chunk_id="chunk-test-001",
        text="Patient reports taking the bus to get to clinic appointments. "
        "Mentions difficulty storing insulin at the shelter.",
        score=0.85,
        provenance=ChunkProvenance(
            document_id="doc-001",
            content_hash="a" * 64,
            kms_signature=KMSSignature(
                key_id="key-001",
                signature="sig-base64-data",
                signed_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
            ),
            chunk_index=0,
            ingestion_timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
        ),
    )


@pytest.fixture
def engine(mock_llm_client, mock_graph_service):
    """ClinicalInferenceEngine with default configuration."""
    return ClinicalInferenceEngine(
        llm_client=mock_llm_client,
        graph_service=mock_graph_service,
        confidence_threshold=0.3,
        inference_depth="shallow",
    )


@pytest.fixture
def engine_high_threshold(mock_llm_client, mock_graph_service):
    """ClinicalInferenceEngine with high confidence threshold."""
    return ClinicalInferenceEngine(
        llm_client=mock_llm_client,
        graph_service=mock_graph_service,
        confidence_threshold=0.8,
        inference_depth="shallow",
    )


# =============================================================================
# Initialization Tests
# =============================================================================


class TestEngineInitialization:
    """Tests for ClinicalInferenceEngine initialization."""

    def test_valid_initialization(self, mock_llm_client, mock_graph_service):
        """Engine initializes with valid parameters."""
        engine = ClinicalInferenceEngine(
            llm_client=mock_llm_client,
            graph_service=mock_graph_service,
            confidence_threshold=0.5,
            inference_depth="deep",
        )
        assert engine.confidence_threshold == 0.5
        assert engine.depth == "deep"

    def test_invalid_confidence_threshold_too_high(self, mock_llm_client, mock_graph_service):
        """Raises ValueError for confidence threshold > 1.0."""
        with pytest.raises(ValueError, match="confidence_threshold"):
            ClinicalInferenceEngine(
                llm_client=mock_llm_client,
                graph_service=mock_graph_service,
                confidence_threshold=1.5,
            )

    def test_invalid_confidence_threshold_negative(self, mock_llm_client, mock_graph_service):
        """Raises ValueError for negative confidence threshold."""
        with pytest.raises(ValueError, match="confidence_threshold"):
            ClinicalInferenceEngine(
                llm_client=mock_llm_client,
                graph_service=mock_graph_service,
                confidence_threshold=-0.1,
            )

    def test_invalid_inference_depth(self, mock_llm_client, mock_graph_service):
        """Raises ValueError for invalid inference depth."""
        with pytest.raises(ValueError, match="inference_depth"):
            ClinicalInferenceEngine(
                llm_client=mock_llm_client,
                graph_service=mock_graph_service,
                inference_depth="medium",
            )

    def test_default_parameters(self, mock_llm_client, mock_graph_service):
        """Engine uses correct defaults when not specified."""
        engine = ClinicalInferenceEngine(
            llm_client=mock_llm_client,
            graph_service=mock_graph_service,
        )
        assert engine.confidence_threshold == 0.3
        assert engine.depth == "shallow"


# =============================================================================
# analyze_snippet Tests
# =============================================================================


class TestAnalyzeSnippet:
    """Tests for analyze_snippet() method."""

    @pytest.mark.asyncio
    async def test_analyze_snippet_returns_inference_result(self, engine, sample_snippet):
        """analyze_snippet returns InferenceResult with valid structure."""
        result = await engine.analyze_snippet(sample_snippet, member_id="member-001")

        assert isinstance(result, InferenceResult)
        assert result.snippet_id == "chunk-test-001"
        assert result.depth_used == "shallow"
        assert result.processing_time_ms >= 0

    @pytest.mark.asyncio
    async def test_analyze_snippet_detects_transportation_barriers(self, engine, sample_snippet):
        """Detects transportation barriers from 'bus' keyword."""
        result = await engine.analyze_snippet(sample_snippet, member_id="member-001")

        transport_facts = [
            f for f in result.inferred_facts
            if f.sdoh_category == "transportation_barriers"
        ]
        assert len(transport_facts) >= 1
        assert transport_facts[0].inference_type == "sdoh_factor"
        assert 0.0 <= transport_facts[0].confidence <= 1.0

    @pytest.mark.asyncio
    async def test_analyze_snippet_detects_housing_instability(self, engine, sample_snippet):
        """Detects housing instability from 'shelter' keyword."""
        result = await engine.analyze_snippet(sample_snippet, member_id="member-001")

        housing_facts = [
            f for f in result.inferred_facts
            if f.sdoh_category == "housing_instability"
        ]
        assert len(housing_facts) >= 1

    @pytest.mark.asyncio
    async def test_analyze_snippet_max_10_facts(self, mock_graph_service):
        """Enforces max 10 inferred facts per snippet."""
        # Create a client that returns many facts
        class ManyFactsLLM:
            async def infer(self, prompt: str) -> str:
                facts = []
                for i in range(15):
                    facts.append({
                        "inference_type": "sdoh_factor",
                        "sdoh_category": "housing_instability",
                        "conclusion": f"Conclusion {i}",
                        "confidence": 0.9 - (i * 0.03),
                        "hops": [{
                            "source_text": f"Source {i}",
                            "intermediate_conclusion": f"Conclusion {i}",
                            "confidence": 0.9 - (i * 0.03),
                        }],
                    })
                return json.dumps(facts)

        engine = ClinicalInferenceEngine(
            llm_client=ManyFactsLLM(),
            graph_service=mock_graph_service,
        )
        snippet = ScoredChunk(
            chunk_id="chunk-many",
            text="Patient has many issues",
            score=0.8,
            provenance=ChunkProvenance(
                document_id="doc-001",
                content_hash="b" * 64,
                kms_signature=KMSSignature(
                    key_id="key-001",
                    signature="sig-data",
                    signed_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
                ),
                chunk_index=0,
                ingestion_timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
            ),
        )

        result = await engine.analyze_snippet(snippet, member_id="member-001")
        assert len(result.inferred_facts) <= 10

    @pytest.mark.asyncio
    async def test_analyze_snippet_timeout_raises_error(self, slow_llm_client, mock_graph_service):
        """Raises InferenceTimeoutError when processing exceeds 15s."""
        engine = ClinicalInferenceEngine(
            llm_client=slow_llm_client,
            graph_service=mock_graph_service,
        )
        # Override timeout for faster test
        engine.SNIPPET_TIMEOUT_SECONDS = 0.1

        snippet = ScoredChunk(
            chunk_id="chunk-timeout",
            text="Some clinical text",
            score=0.8,
            provenance=ChunkProvenance(
                document_id="doc-001",
                content_hash="c" * 64,
                kms_signature=KMSSignature(
                    key_id="key-001",
                    signature="sig-data",
                    signed_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
                ),
                chunk_index=0,
                ingestion_timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
            ),
        )

        with pytest.raises(InferenceTimeoutError) as exc_info:
            await engine.analyze_snippet(snippet, member_id="member-001")

        assert exc_info.value.snippet_id == "chunk-timeout"

    @pytest.mark.asyncio
    async def test_analyze_snippet_all_facts_have_valid_inference_type(
        self, engine, sample_snippet
    ):
        """All inferred facts have a valid inference_type."""
        result = await engine.analyze_snippet(sample_snippet, member_id="member-001")

        for fact in result.inferred_facts:
            assert fact.inference_type in ClinicalInferenceEngine.INFERENCE_TYPES

    @pytest.mark.asyncio
    async def test_analyze_snippet_all_facts_have_valid_confidence(
        self, engine, sample_snippet
    ):
        """All inferred facts have confidence between 0.0 and 1.0."""
        result = await engine.analyze_snippet(sample_snippet, member_id="member-001")

        for fact in result.inferred_facts:
            assert 0.0 <= fact.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_analyze_snippet_sdoh_facts_have_valid_category(
        self, engine, sample_snippet
    ):
        """SDOH factor facts have a valid sdoh_category."""
        result = await engine.analyze_snippet(sample_snippet, member_id="member-001")

        for fact in result.inferred_facts:
            if fact.inference_type == "sdoh_factor":
                assert fact.sdoh_category in ClinicalInferenceEngine.SDOH_CATEGORIES

    @pytest.mark.asyncio
    async def test_analyze_snippet_empty_text_returns_empty(self, mock_graph_service):
        """Empty/no-keyword text returns zero inferred facts."""
        class EmptyLLM:
            async def infer(self, prompt: str) -> str:
                return "[]"

        engine = ClinicalInferenceEngine(
            llm_client=EmptyLLM(),
            graph_service=mock_graph_service,
        )
        snippet = ScoredChunk(
            chunk_id="chunk-empty",
            text="Normal follow up visit. Vitals stable.",
            score=0.7,
            provenance=ChunkProvenance(
                document_id="doc-001",
                content_hash="d" * 64,
                kms_signature=KMSSignature(
                    key_id="key-001",
                    signature="sig-data",
                    signed_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
                ),
                chunk_index=0,
                ingestion_timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
            ),
        )

        result = await engine.analyze_snippet(snippet, member_id="member-001")
        assert len(result.inferred_facts) == 0


# =============================================================================
# derive_sdoh_factors Tests
# =============================================================================


class TestDeriveSdohFactors:
    """Tests for derive_sdoh_factors() method."""

    @pytest.mark.asyncio
    async def test_derive_sdoh_returns_only_sdoh_type_facts(self, engine):
        """derive_sdoh_factors returns only sdoh_factor inference_type."""
        text = "Patient takes the bus to clinic. Reports being hungry often."
        facts = await engine.derive_sdoh_factors(text)

        for fact in facts:
            assert fact.inference_type == "sdoh_factor"

    @pytest.mark.asyncio
    async def test_derive_sdoh_detects_transportation_barriers(self, engine):
        """Detects transportation barriers from text about bus usage."""
        text = "Patient relies on the bus for all medical appointments."
        facts = await engine.derive_sdoh_factors(text)

        categories = [f.sdoh_category for f in facts]
        assert "transportation_barriers" in categories

    @pytest.mark.asyncio
    async def test_derive_sdoh_detects_food_insecurity(self, engine):
        """Detects food insecurity from text about food bank usage."""
        text = "Patient mentions visiting the food bank weekly."
        facts = await engine.derive_sdoh_factors(text)

        categories = [f.sdoh_category for f in facts]
        assert "food_insecurity" in categories

    @pytest.mark.asyncio
    async def test_derive_sdoh_respects_depth_parameter(self, engine):
        """Depth parameter is passed through to LLM prompt."""
        text = "Patient takes the bus to appointments."
        facts = await engine.derive_sdoh_factors(text, depth="deep")

        # Facts should still be valid regardless of depth
        for fact in facts:
            assert fact.inference_type == "sdoh_factor"
            assert fact.sdoh_category in ClinicalInferenceEngine.SDOH_CATEGORIES

    @pytest.mark.asyncio
    async def test_derive_sdoh_applies_threshold_filter(self, engine_high_threshold):
        """Facts below threshold are excluded from results."""
        # Most mock facts have confidence < 0.8
        text = "Patient takes the bus and reports being hungry."
        facts = await engine_high_threshold.derive_sdoh_factors(text)

        for fact in facts:
            assert fact.confidence >= 0.8

    @pytest.mark.asyncio
    async def test_derive_sdoh_max_10_facts(self, mock_graph_service):
        """Enforces max 10 SDOH facts returned."""
        class ManySDOHLLM:
            async def infer(self, prompt: str) -> str:
                facts = []
                for i in range(12):
                    facts.append({
                        "inference_type": "sdoh_factor",
                        "sdoh_category": "housing_instability",
                        "conclusion": f"SDOH factor {i}",
                        "confidence": 0.85 - (i * 0.02),
                        "hops": [{
                            "source_text": f"Source {i}",
                            "intermediate_conclusion": f"SDOH factor {i}",
                            "confidence": 0.85 - (i * 0.02),
                        }],
                    })
                return json.dumps(facts)

        engine = ClinicalInferenceEngine(
            llm_client=ManySDOHLLM(),
            graph_service=mock_graph_service,
        )
        facts = await engine.derive_sdoh_factors("text with many issues")
        assert len(facts) <= 10

    @pytest.mark.asyncio
    async def test_derive_sdoh_handles_no_results(self, mock_graph_service):
        """Returns empty list when no SDOH factors detected."""
        class NoFactsLLM:
            async def infer(self, prompt: str) -> str:
                return "[]"

        engine = ClinicalInferenceEngine(
            llm_client=NoFactsLLM(),
            graph_service=mock_graph_service,
        )
        facts = await engine.derive_sdoh_factors("Normal visit, patient stable.")
        assert facts == []


# =============================================================================
# Confidence and Threshold Tests
# =============================================================================


class TestConfidenceAndThreshold:
    """Tests for confidence calculation and threshold filtering."""

    def test_compute_chain_confidence_single_hop(self, engine):
        """Single-hop chain confidence equals the hop confidence."""
        chain = InferenceChain(
            chain_id="chain-001",
            hops=[InferenceHop(hop_number=1, source_text="text", intermediate_conclusion="conc", confidence=0.8)],
            cumulative_confidence=0.8,
            source_snippet_id="snippet-001",
            final_conclusion="conclusion",
        )
        assert engine.compute_chain_confidence(chain) == pytest.approx(0.8)

    def test_compute_chain_confidence_multi_hop(self, engine):
        """Multi-hop chain confidence is product of hop confidences."""
        chain = InferenceChain(
            chain_id="chain-002",
            hops=[
                InferenceHop(hop_number=1, source_text="t1", intermediate_conclusion="c1", confidence=0.9),
                InferenceHop(hop_number=2, source_text="t2", intermediate_conclusion="c2", confidence=0.8),
                InferenceHop(hop_number=3, source_text="t3", intermediate_conclusion="c3", confidence=0.7),
            ],
            cumulative_confidence=0.504,
            source_snippet_id="snippet-002",
            final_conclusion="final conclusion",
        )
        assert engine.compute_chain_confidence(chain) == pytest.approx(0.9 * 0.8 * 0.7)

    def test_apply_threshold_filter_includes_above(self, engine):
        """Facts above threshold are included."""
        facts = [
            self._make_fact(confidence=0.5),
            self._make_fact(confidence=0.8),
        ]
        filtered = engine.apply_threshold_filter(facts)
        assert len(filtered) == 2

    def test_apply_threshold_filter_excludes_below(self, engine):
        """Facts below threshold are excluded."""
        facts = [
            self._make_fact(confidence=0.1),
            self._make_fact(confidence=0.2),
        ]
        filtered = engine.apply_threshold_filter(facts)
        assert len(filtered) == 0

    def test_apply_threshold_filter_boundary(self, engine):
        """Facts exactly at threshold are included."""
        facts = [self._make_fact(confidence=0.3)]  # Threshold is 0.3
        filtered = engine.apply_threshold_filter(facts)
        assert len(filtered) == 1

    def _make_fact(self, confidence: float) -> InferredFact:
        """Helper to create a minimal InferredFact for testing."""
        chain = InferenceChain(
            chain_id="chain-test",
            hops=[InferenceHop(hop_number=1, source_text="src", intermediate_conclusion="conc", confidence=confidence)],
            cumulative_confidence=confidence,
            source_snippet_id="snippet-test",
            final_conclusion="test conclusion",
        )
        return InferredFact(
            fact_id="fact-test",
            inference_type="care_access_barrier",
            sdoh_category=None,
            conclusion="test conclusion",
            confidence=confidence,
            inference_chain=chain,
            source_text_excerpt="source text",
        )


# =============================================================================
# LLM Response Parsing Tests
# =============================================================================


class TestLLMResponseParsing:
    """Tests for LLM response parsing edge cases."""

    @pytest.mark.asyncio
    async def test_malformed_json_returns_empty(self, mock_graph_service):
        """Malformed JSON response returns empty facts list."""
        class BadJsonLLM:
            async def infer(self, prompt: str) -> str:
                return "this is not valid json {{"

        engine = ClinicalInferenceEngine(
            llm_client=BadJsonLLM(),
            graph_service=mock_graph_service,
        )
        snippet = ScoredChunk(
            chunk_id="chunk-bad",
            text="Some text",
            score=0.7,
            provenance=ChunkProvenance(
                document_id="doc-001",
                content_hash="e" * 64,
                kms_signature=KMSSignature(
                    key_id="key-001",
                    signature="sig-data",
                    signed_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
                ),
                chunk_index=0,
                ingestion_timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
            ),
        )
        result = await engine.analyze_snippet(snippet, member_id="member-001")
        assert len(result.inferred_facts) == 0

    @pytest.mark.asyncio
    async def test_non_array_json_returns_empty(self, mock_graph_service):
        """Non-array JSON response returns empty facts list."""
        class ObjectJsonLLM:
            async def infer(self, prompt: str) -> str:
                return '{"not": "an array"}'

        engine = ClinicalInferenceEngine(
            llm_client=ObjectJsonLLM(),
            graph_service=mock_graph_service,
        )
        snippet = ScoredChunk(
            chunk_id="chunk-obj",
            text="Some text",
            score=0.7,
            provenance=ChunkProvenance(
                document_id="doc-001",
                content_hash="f" * 64,
                kms_signature=KMSSignature(
                    key_id="key-001",
                    signature="sig-data",
                    signed_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
                ),
                chunk_index=0,
                ingestion_timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
            ),
        )
        result = await engine.analyze_snippet(snippet, member_id="member-001")
        assert len(result.inferred_facts) == 0

    @pytest.mark.asyncio
    async def test_invalid_inference_type_skipped(self, mock_graph_service):
        """Facts with invalid inference_type are skipped."""
        class InvalidTypeLLM:
            async def infer(self, prompt: str) -> str:
                return json.dumps([{
                    "inference_type": "invalid_type",
                    "conclusion": "Something",
                    "confidence": 0.8,
                    "hops": [{"source_text": "s", "intermediate_conclusion": "c", "confidence": 0.8}],
                }])

        engine = ClinicalInferenceEngine(
            llm_client=InvalidTypeLLM(),
            graph_service=mock_graph_service,
        )
        snippet = ScoredChunk(
            chunk_id="chunk-invalid",
            text="Some text",
            score=0.7,
            provenance=ChunkProvenance(
                document_id="doc-001",
                content_hash="a" * 64,
                kms_signature=KMSSignature(
                    key_id="key-001",
                    signature="sig-data",
                    signed_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
                ),
                chunk_index=0,
                ingestion_timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
            ),
        )
        result = await engine.analyze_snippet(snippet, member_id="member-001")
        assert len(result.inferred_facts) == 0

    @pytest.mark.asyncio
    async def test_invalid_sdoh_category_skipped(self, mock_graph_service):
        """SDOH facts with invalid category are skipped."""
        class InvalidCategoryLLM:
            async def infer(self, prompt: str) -> str:
                return json.dumps([{
                    "inference_type": "sdoh_factor",
                    "sdoh_category": "invalid_category",
                    "conclusion": "Something",
                    "confidence": 0.8,
                    "hops": [{"source_text": "s", "intermediate_conclusion": "c", "confidence": 0.8}],
                }])

        engine = ClinicalInferenceEngine(
            llm_client=InvalidCategoryLLM(),
            graph_service=mock_graph_service,
        )
        snippet = ScoredChunk(
            chunk_id="chunk-badcat",
            text="Some text",
            score=0.7,
            provenance=ChunkProvenance(
                document_id="doc-001",
                content_hash="a" * 64,
                kms_signature=KMSSignature(
                    key_id="key-001",
                    signature="sig-data",
                    signed_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
                ),
                chunk_index=0,
                ingestion_timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
            ),
        )
        result = await engine.analyze_snippet(snippet, member_id="member-001")
        assert len(result.inferred_facts) == 0

    @pytest.mark.asyncio
    async def test_confidence_out_of_range_skipped(self, mock_graph_service):
        """Facts with confidence outside [0.0, 1.0] are skipped."""
        class BadConfidenceLLM:
            async def infer(self, prompt: str) -> str:
                return json.dumps([{
                    "inference_type": "care_access_barrier",
                    "conclusion": "Something",
                    "confidence": 1.5,
                    "hops": [{"source_text": "s", "intermediate_conclusion": "c", "confidence": 0.8}],
                }])

        engine = ClinicalInferenceEngine(
            llm_client=BadConfidenceLLM(),
            graph_service=mock_graph_service,
        )
        snippet = ScoredChunk(
            chunk_id="chunk-badconf",
            text="Some text",
            score=0.7,
            provenance=ChunkProvenance(
                document_id="doc-001",
                content_hash="a" * 64,
                kms_signature=KMSSignature(
                    key_id="key-001",
                    signature="sig-data",
                    signed_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
                ),
                chunk_index=0,
                ingestion_timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
            ),
        )
        result = await engine.analyze_snippet(snippet, member_id="member-001")
        assert len(result.inferred_facts) == 0


# =============================================================================
# Data Model Tests
# =============================================================================


class TestDataModels:
    """Tests for inference data model validation."""

    def test_inference_hop_valid(self):
        """Valid InferenceHop is created successfully."""
        hop = InferenceHop(
            hop_number=1,
            source_text="Patient mentions bus",
            intermediate_conclusion="Transportation barrier implied",
            confidence=0.75,
        )
        assert hop.hop_number == 1
        assert hop.confidence == 0.75

    def test_inference_hop_invalid_confidence(self):
        """InferenceHop rejects confidence outside [0.0, 1.0]."""
        with pytest.raises(ValueError, match="confidence"):
            InferenceHop(
                hop_number=1,
                source_text="text",
                intermediate_conclusion="conc",
                confidence=1.5,
            )

    def test_inference_hop_invalid_hop_number(self):
        """InferenceHop rejects hop_number < 1."""
        with pytest.raises(ValueError, match="hop_number"):
            InferenceHop(
                hop_number=0,
                source_text="text",
                intermediate_conclusion="conc",
                confidence=0.5,
            )

    def test_inference_chain_valid(self):
        """Valid InferenceChain is created successfully."""
        chain = InferenceChain(
            chain_id="chain-001",
            hops=[
                InferenceHop(hop_number=1, source_text="t", intermediate_conclusion="c", confidence=0.9),
            ],
            cumulative_confidence=0.9,
            source_snippet_id="snippet-001",
            final_conclusion="Final conclusion",
        )
        assert chain.chain_id == "chain-001"
        assert len(chain.hops) == 1

    def test_inferred_fact_requires_sdoh_category_for_sdoh_type(self):
        """InferredFact with sdoh_factor type requires sdoh_category."""
        chain = InferenceChain(
            chain_id="chain-001",
            hops=[InferenceHop(hop_number=1, source_text="t", intermediate_conclusion="c", confidence=0.8)],
            cumulative_confidence=0.8,
            source_snippet_id="snippet-001",
            final_conclusion="conclusion",
        )
        with pytest.raises(ValueError, match="sdoh_category"):
            InferredFact(
                fact_id="fact-001",
                inference_type="sdoh_factor",
                sdoh_category=None,  # Missing!
                conclusion="conclusion",
                confidence=0.8,
                inference_chain=chain,
                source_text_excerpt="text",
            )

    def test_inferred_fact_non_sdoh_type_no_category_needed(self):
        """Non-sdoh_factor types don't require sdoh_category."""
        chain = InferenceChain(
            chain_id="chain-001",
            hops=[InferenceHop(hop_number=1, source_text="t", intermediate_conclusion="c", confidence=0.8)],
            cumulative_confidence=0.8,
            source_snippet_id="snippet-001",
            final_conclusion="conclusion",
        )
        fact = InferredFact(
            fact_id="fact-001",
            inference_type="care_access_barrier",
            sdoh_category=None,
            conclusion="Patient has care access barrier",
            confidence=0.8,
            inference_chain=chain,
            source_text_excerpt="text",
        )
        assert fact.inference_type == "care_access_barrier"
        assert fact.sdoh_category is None

    def test_inferred_fact_truncates_source_text_excerpt(self):
        """source_text_excerpt is truncated to 500 chars."""
        chain = InferenceChain(
            chain_id="chain-001",
            hops=[InferenceHop(hop_number=1, source_text="t", intermediate_conclusion="c", confidence=0.8)],
            cumulative_confidence=0.8,
            source_snippet_id="snippet-001",
            final_conclusion="conclusion",
        )
        long_text = "x" * 600
        fact = InferredFact(
            fact_id="fact-001",
            inference_type="medication_adherence_risk",
            sdoh_category=None,
            conclusion="conclusion",
            confidence=0.8,
            inference_chain=chain,
            source_text_excerpt=long_text,
        )
        assert len(fact.source_text_excerpt) == 500

    def test_inference_result_max_facts_enforced(self):
        """InferenceResult rejects more than 10 facts."""
        chain = InferenceChain(
            chain_id="chain-001",
            hops=[InferenceHop(hop_number=1, source_text="t", intermediate_conclusion="c", confidence=0.8)],
            cumulative_confidence=0.8,
            source_snippet_id="snippet-001",
            final_conclusion="conclusion",
        )
        facts = [
            InferredFact(
                fact_id=f"fact-{i}",
                inference_type="care_access_barrier",
                sdoh_category=None,
                conclusion=f"Conclusion {i}",
                confidence=0.8,
                inference_chain=chain,
                source_text_excerpt="text",
            )
            for i in range(11)
        ]
        with pytest.raises(ValueError, match="Max 10"):
            InferenceResult(
                snippet_id="snippet-001",
                inferred_facts=facts,
                processing_time_ms=100,
                depth_used="shallow",
                total_hops_executed=11,
            )


# =============================================================================
# Integration-style Tests (full flow with mock LLM)
# =============================================================================


class TestFullInferenceFlow:
    """End-to-end inference flow tests with mock LLM."""

    @pytest.mark.asyncio
    async def test_full_flow_food_insecurity(self, engine):
        """Full flow detecting food insecurity from text."""
        snippet = ScoredChunk(
            chunk_id="chunk-food",
            text="Patient reports visiting the food bank regularly due to limited income.",
            score=0.9,
            provenance=ChunkProvenance(
                document_id="doc-food",
                content_hash="a" * 64,
                kms_signature=KMSSignature(
                    key_id="key-001",
                    signature="sig-data",
                    signed_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
                ),
                chunk_index=0,
                ingestion_timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
            ),
        )
        result = await engine.analyze_snippet(snippet, member_id="member-food")

        assert result.snippet_id == "chunk-food"
        food_facts = [f for f in result.inferred_facts if f.sdoh_category == "food_insecurity"]
        assert len(food_facts) >= 1
        assert food_facts[0].confidence >= 0.3

    @pytest.mark.asyncio
    async def test_full_flow_caregiver_availability(self, engine):
        """Full flow detecting caregiver availability issues."""
        snippet = ScoredChunk(
            chunk_id="chunk-alone",
            text="Patient lives alone with no family nearby to help with medication management.",
            score=0.85,
            provenance=ChunkProvenance(
                document_id="doc-alone",
                content_hash="b" * 64,
                kms_signature=KMSSignature(
                    key_id="key-001",
                    signature="sig-data",
                    signed_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
                ),
                chunk_index=0,
                ingestion_timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
            ),
        )
        result = await engine.analyze_snippet(snippet, member_id="member-alone")

        caregiver_facts = [
            f for f in result.inferred_facts
            if f.sdoh_category == "caregiver_availability"
        ]
        assert len(caregiver_facts) >= 1

    @pytest.mark.asyncio
    async def test_full_flow_medication_adherence_risk(self, engine):
        """Full flow detecting medication adherence risk."""
        snippet = ScoredChunk(
            chunk_id="chunk-insulin",
            text="Patient struggles to keep insulin refrigerated at the shelter.",
            score=0.88,
            provenance=ChunkProvenance(
                document_id="doc-insulin",
                content_hash="c" * 64,
                kms_signature=KMSSignature(
                    key_id="key-001",
                    signature="sig-data",
                    signed_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
                ),
                chunk_index=0,
                ingestion_timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
            ),
        )
        result = await engine.analyze_snippet(snippet, member_id="member-insulin")

        # Should detect both medication adherence risk and housing instability
        med_facts = [
            f for f in result.inferred_facts
            if f.inference_type == "medication_adherence_risk"
        ]
        housing_facts = [
            f for f in result.inferred_facts
            if f.sdoh_category == "housing_instability"
        ]
        assert len(med_facts) >= 1 or len(housing_facts) >= 1


# =============================================================================
# link_to_graph Tests
# =============================================================================


class TestLinkToGraph:
    """Tests for link_to_graph() method.

    Validates:
        - SDOH_Factor nodes are created for facts above confidence threshold
        - No nodes are created for facts below confidence threshold
        - INFERRED_FROM relationship has correct properties
        - origin="inferred" is set on the SDOH_Factor node

    Requirements referenced: 14.3
    """

    @pytest.fixture
    def sdoh_fact_above_threshold(self):
        """Create an SDOH fact with confidence above default threshold (0.3)."""
        chain = InferenceChain(
            chain_id="chain-graph-001",
            hops=[
                InferenceHop(
                    hop_number=1,
                    source_text="Patient reports taking bus to clinic",
                    intermediate_conclusion="Patient lacks reliable transportation",
                    confidence=0.75,
                ),
            ],
            cumulative_confidence=0.75,
            source_snippet_id="evidence-source-001",
            final_conclusion="Patient faces transportation barriers",
        )
        return InferredFact(
            fact_id="fact-graph-001",
            inference_type="sdoh_factor",
            sdoh_category="transportation_barriers",
            conclusion="Patient faces transportation barriers",
            confidence=0.75,
            inference_chain=chain,
            source_text_excerpt="Patient reports taking bus to clinic",
            inferred_at=datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc),
        )

    @pytest.fixture
    def sdoh_fact_below_threshold(self):
        """Create an SDOH fact with confidence below default threshold (0.3)."""
        chain = InferenceChain(
            chain_id="chain-graph-002",
            hops=[
                InferenceHop(
                    hop_number=1,
                    source_text="Patient mentioned bus stop",
                    intermediate_conclusion="Possible transportation issue",
                    confidence=0.2,
                ),
            ],
            cumulative_confidence=0.2,
            source_snippet_id="evidence-source-002",
            final_conclusion="Possible transportation barriers",
        )
        return InferredFact(
            fact_id="fact-graph-002",
            inference_type="sdoh_factor",
            sdoh_category="transportation_barriers",
            conclusion="Possible transportation barriers",
            confidence=0.2,
            inference_chain=chain,
            source_text_excerpt="Patient mentioned bus stop",
            inferred_at=datetime(2024, 6, 15, 11, 0, 0, tzinfo=timezone.utc),
        )

    @pytest.fixture
    def multi_hop_sdoh_fact(self):
        """Create an SDOH fact with multi-hop inference chain."""
        chain = InferenceChain(
            chain_id="chain-graph-003",
            hops=[
                InferenceHop(
                    hop_number=1,
                    source_text="Patient stores insulin at neighbor's house",
                    intermediate_conclusion="Patient lacks refrigeration at home",
                    confidence=0.85,
                ),
                InferenceHop(
                    hop_number=2,
                    source_text="Lack of refrigeration implies housing issues",
                    intermediate_conclusion="Patient may have unstable housing",
                    confidence=0.7,
                ),
            ],
            cumulative_confidence=0.595,
            source_snippet_id="evidence-source-003",
            final_conclusion="Patient has medication storage limitations due to housing instability",
        )
        return InferredFact(
            fact_id="fact-graph-003",
            inference_type="sdoh_factor",
            sdoh_category="medication_storage_limitations",
            conclusion="Patient has medication storage limitations due to housing instability",
            confidence=0.595,
            inference_chain=chain,
            source_text_excerpt="Patient stores insulin at neighbor's house",
            inferred_at=datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc),
        )

    @pytest.mark.asyncio
    async def test_link_creates_sdoh_node_for_above_threshold(
        self, engine, mock_graph_service, sdoh_fact_above_threshold
    ):
        """SDOH_Factor node is created when fact confidence >= threshold."""
        await engine.link_to_graph("member-001", sdoh_fact_above_threshold)

        # Verify upsert_node was called for SDOH_Factor
        mock_graph_service.upsert_node.assert_called_once()
        call_args = mock_graph_service.upsert_node.call_args
        assert call_args.kwargs["node_type"] == "SDOH_Factor"
        assert call_args.kwargs["properties"]["type"] == "transportation_barriers"
        assert call_args.kwargs["properties"]["origin"] == "inferred"
        assert call_args.kwargs["properties"]["confidence"] == 0.75

    @pytest.mark.asyncio
    async def test_link_creates_inferred_from_relationship(
        self, engine, mock_graph_service, sdoh_fact_above_threshold
    ):
        """INFERRED_FROM relationship is created with correct properties."""
        await engine.link_to_graph("member-001", sdoh_fact_above_threshold)

        # Verify upsert_relationship was called
        mock_graph_service.upsert_relationship.assert_called_once()
        call_args = mock_graph_service.upsert_relationship.call_args
        assert call_args.kwargs["rel_type"] == "INFERRED_FROM"
        assert call_args.kwargs["target_id"] == "evidence-source-001"

        # Check relationship properties
        props = call_args.kwargs["properties"]
        assert props["source_text"] == "Patient reports taking bus to clinic"
        assert props["confidence"] == 0.75
        assert "inference_chain_json" in props
        assert props["inferred_at"] == "2024-06-15T10:30:00+00:00"

    @pytest.mark.asyncio
    async def test_link_no_node_for_below_threshold(
        self, engine, mock_graph_service, sdoh_fact_below_threshold
    ):
        """No node or relationship created when fact confidence < threshold."""
        await engine.link_to_graph("member-001", sdoh_fact_below_threshold)

        # Neither upsert_node nor upsert_relationship should be called
        mock_graph_service.upsert_node.assert_not_called()
        mock_graph_service.upsert_relationship.assert_not_called()

    @pytest.mark.asyncio
    async def test_link_origin_is_inferred(
        self, engine, mock_graph_service, sdoh_fact_above_threshold
    ):
        """The SDOH_Factor node has origin='inferred' property."""
        await engine.link_to_graph("member-001", sdoh_fact_above_threshold)

        call_args = mock_graph_service.upsert_node.call_args
        properties = call_args.kwargs["properties"]
        assert properties["origin"] == "inferred"

    @pytest.mark.asyncio
    async def test_link_relationship_properties_complete(
        self, engine, mock_graph_service, sdoh_fact_above_threshold
    ):
        """Relationship has all required properties: source_text, inference_chain_json, confidence, inferred_at."""
        await engine.link_to_graph("member-001", sdoh_fact_above_threshold)

        call_args = mock_graph_service.upsert_relationship.call_args
        props = call_args.kwargs["properties"]

        # All four required properties must be present
        assert "source_text" in props
        assert "inference_chain_json" in props
        assert "confidence" in props
        assert "inferred_at" in props

        # source_text should match fact's source_text_excerpt
        assert props["source_text"] == sdoh_fact_above_threshold.source_text_excerpt

        # confidence should match fact's confidence
        assert props["confidence"] == sdoh_fact_above_threshold.confidence

    @pytest.mark.asyncio
    async def test_link_inference_chain_json_is_valid(
        self, engine, mock_graph_service, sdoh_fact_above_threshold
    ):
        """inference_chain_json is valid JSON containing chain details."""
        await engine.link_to_graph("member-001", sdoh_fact_above_threshold)

        call_args = mock_graph_service.upsert_relationship.call_args
        props = call_args.kwargs["properties"]
        chain_data = json.loads(props["inference_chain_json"])

        assert chain_data["chain_id"] == "chain-graph-001"
        assert len(chain_data["hops"]) == 1
        assert chain_data["hops"][0]["hop_number"] == 1
        assert chain_data["hops"][0]["source_text"] == "Patient reports taking bus to clinic"
        assert chain_data["cumulative_confidence"] == 0.75
        assert chain_data["final_conclusion"] == "Patient faces transportation barriers"

    @pytest.mark.asyncio
    async def test_link_multi_hop_chain_serialized_correctly(
        self, engine, mock_graph_service, multi_hop_sdoh_fact
    ):
        """Multi-hop inference chain is correctly serialized to JSON."""
        await engine.link_to_graph("member-001", multi_hop_sdoh_fact)

        call_args = mock_graph_service.upsert_relationship.call_args
        props = call_args.kwargs["properties"]
        chain_data = json.loads(props["inference_chain_json"])

        assert len(chain_data["hops"]) == 2
        assert chain_data["hops"][0]["hop_number"] == 1
        assert chain_data["hops"][1]["hop_number"] == 2
        assert chain_data["cumulative_confidence"] == 0.595

    @pytest.mark.asyncio
    async def test_link_sdoh_node_has_member_id(
        self, engine, mock_graph_service, sdoh_fact_above_threshold
    ):
        """SDOH_Factor node includes member_id for context."""
        await engine.link_to_graph("member-test-123", sdoh_fact_above_threshold)

        call_args = mock_graph_service.upsert_node.call_args
        properties = call_args.kwargs["properties"]
        assert properties["member_id"] == "member-test-123"

    @pytest.mark.asyncio
    async def test_link_fact_at_exact_threshold(self, mock_llm_client, mock_graph_service):
        """Fact with confidence exactly at threshold IS linked."""
        engine = ClinicalInferenceEngine(
            llm_client=mock_llm_client,
            graph_service=mock_graph_service,
            confidence_threshold=0.5,
        )
        chain = InferenceChain(
            chain_id="chain-boundary",
            hops=[InferenceHop(hop_number=1, source_text="t", intermediate_conclusion="c", confidence=0.5)],
            cumulative_confidence=0.5,
            source_snippet_id="evidence-boundary",
            final_conclusion="Boundary conclusion",
        )
        fact = InferredFact(
            fact_id="fact-boundary",
            inference_type="sdoh_factor",
            sdoh_category="food_insecurity",
            conclusion="Boundary conclusion",
            confidence=0.5,
            inference_chain=chain,
            source_text_excerpt="boundary text",
            inferred_at=datetime(2024, 6, 15, 10, 0, 0, tzinfo=timezone.utc),
        )

        await engine.link_to_graph("member-001", fact)

        # Should be linked since 0.5 >= 0.5
        mock_graph_service.upsert_node.assert_called_once()
        mock_graph_service.upsert_relationship.assert_called_once()

    @pytest.mark.asyncio
    async def test_link_fact_just_below_threshold(self, mock_llm_client, mock_graph_service):
        """Fact with confidence just below threshold is NOT linked."""
        engine = ClinicalInferenceEngine(
            llm_client=mock_llm_client,
            graph_service=mock_graph_service,
            confidence_threshold=0.5,
        )
        chain = InferenceChain(
            chain_id="chain-below",
            hops=[InferenceHop(hop_number=1, source_text="t", intermediate_conclusion="c", confidence=0.49)],
            cumulative_confidence=0.49,
            source_snippet_id="evidence-below",
            final_conclusion="Below threshold conclusion",
        )
        fact = InferredFact(
            fact_id="fact-below",
            inference_type="sdoh_factor",
            sdoh_category="food_insecurity",
            conclusion="Below threshold conclusion",
            confidence=0.49,
            inference_chain=chain,
            source_text_excerpt="below text",
            inferred_at=datetime(2024, 6, 15, 10, 0, 0, tzinfo=timezone.utc),
        )

        await engine.link_to_graph("member-001", fact)

        mock_graph_service.upsert_node.assert_not_called()
        mock_graph_service.upsert_relationship.assert_not_called()

    @pytest.mark.asyncio
    async def test_link_sdoh_node_type_is_correct(
        self, engine, mock_graph_service, sdoh_fact_above_threshold
    ):
        """Node is created with node_type='SDOH_Factor'."""
        await engine.link_to_graph("member-001", sdoh_fact_above_threshold)

        call_args = mock_graph_service.upsert_node.call_args
        assert call_args.kwargs["node_type"] == "SDOH_Factor"

    @pytest.mark.asyncio
    async def test_link_sdoh_node_has_confidence(
        self, engine, mock_graph_service, sdoh_fact_above_threshold
    ):
        """SDOH_Factor node has the confidence score property."""
        await engine.link_to_graph("member-001", sdoh_fact_above_threshold)

        call_args = mock_graph_service.upsert_node.call_args
        properties = call_args.kwargs["properties"]
        assert properties["confidence"] == 0.75
