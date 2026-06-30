"""
Unit tests for CausalOntologyGraphService.

Tests the graph service logic including active-state filtering,
node upsert with provenance, relationship upsert, and evidence querying.

Requirements validated: 3.1, 3.2, 3.4, 3.5
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from clinical_reasoning_fabric.graph.causal_ontology_graph_service import (
    ACTIVE_STATE_QUERY_TIMEOUT_SECONDS,
    CausalOntologyGraphService,
    INACTIVE_STATUSES,
    UNIQUENESS_CONSTRAINTS,
    VALID_NODE_TYPES,
    VALID_RELATIONSHIP_TYPES,
)
from clinical_reasoning_fabric.models.core import MemberActiveState
from clinical_reasoning_fabric.models.exceptions import MemberNotFoundError


# =============================================================================
# Fixtures
# =============================================================================


class MockResult:
    """Mock for Neo4j result cursor."""

    def __init__(self, records: list[dict[str, Any]] | None = None, single_value=None):
        self._records = records or []
        self._single_value = single_value

    async def single(self):
        return self._single_value

    async def data(self):
        return self._records


class MockSession:
    """Mock for Neo4j async session."""

    def __init__(self):
        self.run = AsyncMock()
        self._calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class MockDriver:
    """Mock for Neo4j async driver."""

    def __init__(self, session: MockSession = None):
        self._session = session or MockSession()

    def session(self):
        return self._session


@pytest.fixture
def mock_session():
    return MockSession()


@pytest.fixture
def mock_driver(mock_session):
    return MockDriver(mock_session)


@pytest.fixture
def service(mock_driver):
    return CausalOntologyGraphService(mock_driver)


# =============================================================================
# Tests: Constants and Configuration
# =============================================================================


class TestConstants:
    """Test that module constants are defined correctly."""

    def test_inactive_statuses_contains_all_required(self):
        """Requirement 3.5: resolved, discontinued, closed, superseded are inactive."""
        assert "resolved" in INACTIVE_STATUSES
        assert "discontinued" in INACTIVE_STATUSES
        assert "closed" in INACTIVE_STATUSES
        assert "superseded" in INACTIVE_STATUSES

    def test_valid_node_types(self):
        """Requirement 3.1: All specified node types are defined."""
        assert "Member" in VALID_NODE_TYPES
        assert "Event" in VALID_NODE_TYPES
        assert "PolicyRule" in VALID_NODE_TYPES
        assert "SDOH_Factor" in VALID_NODE_TYPES
        assert "EvidenceSource" in VALID_NODE_TYPES

    def test_valid_relationship_types(self):
        """Requirement 3.2: All specified relationship types are defined."""
        assert "HAS_CONDITION" in VALID_RELATIONSHIP_TYPES
        assert "IS_PRESCRIBED" in VALID_RELATIONSHIP_TYPES
        assert "TRIGGERED_BY" in VALID_RELATIONSHIP_TYPES
        assert "GOVERNED_BY" in VALID_RELATIONSHIP_TYPES
        assert "EVIDENCED_BY" in VALID_RELATIONSHIP_TYPES

    def test_uniqueness_constraints_defined(self):
        """Requirement 3.1: Uniqueness constraints for all node types."""
        assert len(UNIQUENESS_CONSTRAINTS) == 5
        # Check each node type has a constraint
        constraint_text = " ".join(UNIQUENESS_CONSTRAINTS)
        assert "Member" in constraint_text
        assert "Event" in constraint_text
        assert "PolicyRule" in constraint_text
        assert "SDOH_Factor" in constraint_text
        assert "EvidenceSource" in constraint_text

    def test_active_state_timeout(self):
        """Requirement 3.4: Query must complete within 2 seconds."""
        assert ACTIVE_STATE_QUERY_TIMEOUT_SECONDS == 2.0


# =============================================================================
# Tests: get_member_active_state
# =============================================================================


class TestGetMemberActiveState:
    """Tests for the get_member_active_state method."""

    @pytest.mark.asyncio
    async def test_member_not_found_raises_error(self, service, mock_session):
        """Requirement 4.5: Raises MemberNotFoundError for non-existent members."""
        # Mock the member check to return None (not found)
        mock_session.run.return_value = MockResult(single_value=None)

        with pytest.raises(MemberNotFoundError) as exc_info:
            await service.get_member_active_state("non-existent-member")

        assert exc_info.value.member_id == "non-existent-member"

    @pytest.mark.asyncio
    async def test_returns_member_active_state_model(self, service, mock_session):
        """Requirement 3.4: Returns MemberActiveState with all fields."""
        # Mock member exists
        member_result = MockResult(single_value={"m": {"member_id": "M001"}})
        # Mock empty results for diagnoses, prescriptions, sdoh, policies
        empty_result = MockResult(records=[])

        mock_session.run.side_effect = [
            member_result,
            empty_result,  # diagnoses
            empty_result,  # prescriptions
            empty_result,  # sdoh factors
            empty_result,  # governing policies
        ]

        result = await service.get_member_active_state("M001")

        assert isinstance(result, MemberActiveState)
        assert result.member_id == "M001"
        assert result.active_diagnoses == []
        assert result.active_prescriptions == []
        assert result.sdoh_factors == []
        assert result.governing_policies == []
        assert result.last_updated is not None

    @pytest.mark.asyncio
    async def test_returns_active_diagnoses(self, service, mock_session):
        """Requirement 3.4: Returns current active diagnoses."""
        member_result = MockResult(single_value={"m": {"member_id": "M001"}})
        diagnoses_result = MockResult(
            records=[
                {"e": {"event_id": "E001", "condition_code": "J45.0", "status": "active"}},
                {"e": {"event_id": "E002", "condition_code": "E11.9", "status": "active"}},
            ]
        )
        empty_result = MockResult(records=[])

        mock_session.run.side_effect = [
            member_result,
            diagnoses_result,
            empty_result,  # prescriptions
            empty_result,  # sdoh factors
            empty_result,  # governing policies
        ]

        result = await service.get_member_active_state("M001")

        assert len(result.active_diagnoses) == 2
        assert result.active_diagnoses[0]["condition_code"] == "J45.0"

    @pytest.mark.asyncio
    async def test_timeout_raises_error(self, service, mock_session):
        """Requirement 3.4: Must complete within 2 seconds."""
        # Make the query take longer than the timeout
        async def slow_query(*args, **kwargs):
            await asyncio.sleep(3)
            return MockResult(single_value={"m": {"member_id": "M001"}})

        mock_session.run.side_effect = slow_query

        with pytest.raises(asyncio.TimeoutError):
            await service.get_member_active_state("M001")


# =============================================================================
# Tests: upsert_node
# =============================================================================


class TestUpsertNode:
    """Tests for the upsert_node method."""

    @pytest.mark.asyncio
    async def test_upsert_member_node(self, service, mock_session):
        """Requirement 3.1: Can upsert a Member node."""
        mock_session.run.return_value = MockResult()

        await service.upsert_node(
            node_type="Member",
            node_id="M001",
            properties={"name": "John Doe", "dob": "1980-01-01"},
        )

        mock_session.run.assert_called_once()
        call_args = mock_session.run.call_args
        assert "MERGE" in call_args[0][0]
        assert "Member" in call_args[0][0]
        assert call_args[1]["node_id"] == "M001"

    @pytest.mark.asyncio
    async def test_upsert_with_execution_id_provenance(self, service, mock_session):
        """Requirement 11.4: Records execution_id as provenance."""
        mock_session.run.return_value = MockResult()

        await service.upsert_node(
            node_type="Event",
            node_id="E001",
            properties={"condition_code": "J45.0"},
            execution_id="exec-123",
        )

        call_args = mock_session.run.call_args
        props = call_args[1]["properties"]
        assert props["_last_execution_id"] == "exec-123"
        assert "_last_updated_at" in props

    @pytest.mark.asyncio
    async def test_upsert_without_execution_id(self, service, mock_session):
        """No provenance metadata when execution_id is not provided."""
        mock_session.run.return_value = MockResult()

        await service.upsert_node(
            node_type="PolicyRule",
            node_id="POL-001",
            properties={"name": "Rule 1"},
        )

        call_args = mock_session.run.call_args
        props = call_args[1]["properties"]
        assert "_last_execution_id" not in props

    @pytest.mark.asyncio
    async def test_invalid_node_type_raises_error(self, service):
        """Raises ValueError for invalid node type."""
        with pytest.raises(ValueError, match="Invalid node_type"):
            await service.upsert_node(
                node_type="InvalidType",
                node_id="X001",
                properties={},
            )

    @pytest.mark.asyncio
    async def test_all_valid_node_types_accepted(self, service, mock_session):
        """All defined node types are accepted without error."""
        mock_session.run.return_value = MockResult()

        for node_type in VALID_NODE_TYPES:
            await service.upsert_node(
                node_type=node_type,
                node_id=f"test-{node_type}",
                properties={"test": True},
            )


# =============================================================================
# Tests: upsert_relationship
# =============================================================================


class TestUpsertRelationship:
    """Tests for the upsert_relationship method."""

    @pytest.mark.asyncio
    async def test_upsert_has_condition_relationship(self, service, mock_session):
        """Requirement 3.2: Can create HAS_CONDITION relationship."""
        mock_session.run.return_value = MockResult()

        await service.upsert_relationship(
            source_id="M001",
            target_id="E001",
            rel_type="HAS_CONDITION",
            properties={"diagnosed_at": "2024-01-01"},
        )

        mock_session.run.assert_called_once()
        call_args = mock_session.run.call_args
        query = call_args[0][0]
        assert "MERGE" in query
        assert "HAS_CONDITION" in query
        assert call_args[1]["source_id"] == "M001"
        assert call_args[1]["target_id"] == "E001"

    @pytest.mark.asyncio
    async def test_upsert_relationship_adds_timestamp(self, service, mock_session):
        """Relationship properties include _updated_at timestamp."""
        mock_session.run.return_value = MockResult()

        await service.upsert_relationship(
            source_id="E001",
            target_id="POL-001",
            rel_type="GOVERNED_BY",
            properties={"reason": "policy applies"},
        )

        call_args = mock_session.run.call_args
        props = call_args[1]["properties"]
        assert "_updated_at" in props
        assert "reason" in props

    @pytest.mark.asyncio
    async def test_invalid_relationship_type_raises_error(self, service):
        """Raises ValueError for invalid relationship type."""
        with pytest.raises(ValueError, match="Invalid rel_type"):
            await service.upsert_relationship(
                source_id="M001",
                target_id="E001",
                rel_type="INVALID_REL",
                properties={},
            )

    @pytest.mark.asyncio
    async def test_all_valid_relationship_types_accepted(self, service, mock_session):
        """All defined relationship types are accepted."""
        mock_session.run.return_value = MockResult()

        for rel_type in VALID_RELATIONSHIP_TYPES:
            await service.upsert_relationship(
                source_id="source-1",
                target_id="target-1",
                rel_type=rel_type,
                properties={},
            )


# =============================================================================
# Tests: query_related_evidence
# =============================================================================


class TestQueryRelatedEvidence:
    """Tests for the query_related_evidence method."""

    @pytest.mark.asyncio
    async def test_returns_evidence_for_condition(self, service, mock_session):
        """Returns evidence nodes linked to a specific condition."""
        mock_session.run.return_value = MockResult(
            records=[
                {"ev": {"evidence_id": "EV001", "document_id": "DOC-1", "type": "clinical_note"}},
                {"ev": {"evidence_id": "EV002", "document_id": "DOC-2", "type": "lab_result"}},
            ]
        )

        result = await service.query_related_evidence("M001", "J45.0")

        assert len(result) == 2
        assert result[0]["evidence_id"] == "EV001"
        assert result[1]["evidence_id"] == "EV002"

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_evidence(self, service, mock_session):
        """Returns empty list when no evidence is linked."""
        mock_session.run.return_value = MockResult(records=[])

        result = await service.query_related_evidence("M001", "Z99.9")

        assert result == []

    @pytest.mark.asyncio
    async def test_filters_by_condition_code(self, service, mock_session):
        """Query uses the condition_code parameter for filtering."""
        mock_session.run.return_value = MockResult(records=[])

        await service.query_related_evidence("M001", "E11.9")

        call_args = mock_session.run.call_args
        assert call_args[1]["condition_code"] == "E11.9"
        assert call_args[1]["member_id"] == "M001"

    @pytest.mark.asyncio
    async def test_excludes_inactive_conditions(self, service, mock_session):
        """Requirement 3.5: Only queries evidence for active conditions."""
        mock_session.run.return_value = MockResult(records=[])

        await service.query_related_evidence("M001", "J45.0")

        call_args = mock_session.run.call_args
        assert call_args[1]["inactive_statuses"] == list(INACTIVE_STATUSES)


# =============================================================================
# Tests: ensure_constraints
# =============================================================================


class TestEnsureConstraints:
    """Tests for the ensure_constraints method."""

    @pytest.mark.asyncio
    async def test_creates_all_constraints(self, service, mock_session):
        """Creates all 5 uniqueness constraints."""
        mock_session.run.return_value = MockResult()

        await service.ensure_constraints()

        assert mock_session.run.call_count == 5

    @pytest.mark.asyncio
    async def test_handles_existing_constraints_gracefully(self, service, mock_session):
        """Continues if a constraint already exists."""
        mock_session.run.side_effect = [
            MockResult(),
            Exception("Constraint already exists"),
            MockResult(),
            MockResult(),
            MockResult(),
        ]

        # Should not raise
        await service.ensure_constraints()


# =============================================================================
# Tests: _get_id_field
# =============================================================================


class TestGetIdField:
    """Tests for the static _get_id_field helper."""

    def test_member_id_field(self):
        assert CausalOntologyGraphService._get_id_field("Member") == "member_id"

    def test_event_id_field(self):
        assert CausalOntologyGraphService._get_id_field("Event") == "event_id"

    def test_policy_id_field(self):
        assert CausalOntologyGraphService._get_id_field("PolicyRule") == "policy_id"

    def test_sdoh_id_field(self):
        assert CausalOntologyGraphService._get_id_field("SDOH_Factor") == "sdoh_id"

    def test_evidence_id_field(self):
        assert CausalOntologyGraphService._get_id_field("EvidenceSource") == "evidence_id"
