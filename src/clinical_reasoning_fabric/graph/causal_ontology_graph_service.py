"""
Causal Ontology Graph Service — Neo4j operations for active clinical state management.

Provides typed node and relationship management, active-state querying with
status filtering, and evidence linkage for the Clinical Reasoning Fabric.

Requirements referenced: 3.1, 3.2, 3.4, 3.5
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from neo4j import AsyncDriver, AsyncSession

from clinical_reasoning_fabric.models.core import MemberActiveState
from clinical_reasoning_fabric.models.exceptions import MemberNotFoundError

logger = logging.getLogger(__name__)

# Status values that indicate a record is no longer active (Requirement 3.5)
INACTIVE_STATUSES = frozenset({"resolved", "discontinued", "closed", "superseded"})

# Valid node types in the Causal Ontology Graph (Requirement 3.1)
VALID_NODE_TYPES = frozenset({"Member", "Event", "PolicyRule", "SDOH_Factor", "EvidenceSource"})

# Valid relationship types (Requirement 3.2)
VALID_RELATIONSHIP_TYPES = frozenset({
    "HAS_CONDITION",
    "IS_PRESCRIBED",
    "TRIGGERED_BY",
    "GOVERNED_BY",
    "EVIDENCED_BY",
    "INFERRED_FROM",
})

# Neo4j constraint definitions for uniqueness (Requirement 3.1)
UNIQUENESS_CONSTRAINTS = [
    "CREATE CONSTRAINT member_id IF NOT EXISTS FOR (m:Member) REQUIRE m.member_id IS UNIQUE",
    "CREATE CONSTRAINT event_id IF NOT EXISTS FOR (e:Event) REQUIRE e.event_id IS UNIQUE",
    "CREATE CONSTRAINT policy_id IF NOT EXISTS FOR (p:PolicyRule) REQUIRE p.policy_id IS UNIQUE",
    "CREATE CONSTRAINT sdoh_id IF NOT EXISTS FOR (s:SDOH_Factor) REQUIRE s.sdoh_id IS UNIQUE",
    "CREATE CONSTRAINT evidence_id IF NOT EXISTS FOR (ev:EvidenceSource) REQUIRE ev.evidence_id IS UNIQUE",
]

# Timeout for active-state queries (Requirement 3.4: within 2 seconds)
ACTIVE_STATE_QUERY_TIMEOUT_SECONDS = 2.0


class CausalOntologyGraphService:
    """Interface for querying and updating the Neo4j causal ontology graph.

    Requirement 3.1: Stores Member, Event, PolicyRule, SDOH_Factor, EvidenceSource as typed nodes.
    Requirement 3.2: Stores HAS_CONDITION, IS_PRESCRIBED, TRIGGERED_BY, GOVERNED_BY, EVIDENCED_BY
                     as typed directed relationships.
    Requirement 3.4: Returns active clinical state within 2 seconds.
    Requirement 3.5: Excludes resolved, discontinued, closed, superseded records.
    """

    def __init__(self, neo4j_driver: AsyncDriver) -> None:
        """Initialize with an async Neo4j driver.

        Args:
            neo4j_driver: An async Neo4j driver instance for database operations.
        """
        self.driver = neo4j_driver

    async def ensure_constraints(self) -> None:
        """Create uniqueness constraints for all node types if they don't exist.

        Defines Neo4j constraints for Member, Event, PolicyRule, SDOH_Factor,
        and EvidenceSource uniqueness as specified in the graph schema.
        """
        async with self.driver.session() as session:
            for constraint_query in UNIQUENESS_CONSTRAINTS:
                try:
                    await session.run(constraint_query)
                except Exception as e:
                    logger.warning(f"Constraint creation skipped (may already exist): {e}")

    async def get_member_active_state(self, member_id: str) -> MemberActiveState:
        """Return active clinical state for a member.

        Queries the Neo4j graph for current diagnoses, active prescriptions,
        linked SDOH factors, and governing policy rules. Only returns records
        that are NOT marked as resolved, discontinued, closed, or superseded.

        Must complete within 2 seconds (Requirement 3.4).

        Args:
            member_id: The unique member identifier.

        Returns:
            MemberActiveState with active diagnoses, prescriptions, SDOH factors,
            and governing policies.

        Raises:
            MemberNotFoundError: If the member does not exist in the graph.
            asyncio.TimeoutError: If the query exceeds the 2-second deadline.
        """
        try:
            result = await asyncio.wait_for(
                self._execute_active_state_query(member_id),
                timeout=ACTIVE_STATE_QUERY_TIMEOUT_SECONDS,
            )
            return result
        except asyncio.TimeoutError:
            logger.error(
                f"Active state query for member {member_id} exceeded "
                f"{ACTIVE_STATE_QUERY_TIMEOUT_SECONDS}s timeout"
            )
            raise

    async def _execute_active_state_query(self, member_id: str) -> MemberActiveState:
        """Execute the active state Cypher queries against Neo4j.

        Args:
            member_id: The unique member identifier.

        Returns:
            MemberActiveState populated with active clinical data.

        Raises:
            MemberNotFoundError: If the member node does not exist.
        """
        async with self.driver.session() as session:
            # First verify the member exists
            member_check = await session.run(
                "MATCH (m:Member {member_id: $member_id}) RETURN m",
                member_id=member_id,
            )
            member_record = await member_check.single()
            if member_record is None:
                raise MemberNotFoundError(
                    reason=f"Member '{member_id}' not found in the Causal Ontology Graph",
                    member_id=member_id,
                )

            # Query active diagnoses (HAS_CONDITION relationships to Event nodes)
            active_diagnoses = await self._query_active_diagnoses(session, member_id)

            # Query active prescriptions (IS_PRESCRIBED relationships to Event nodes)
            active_prescriptions = await self._query_active_prescriptions(session, member_id)

            # Query linked SDOH factors
            sdoh_factors = await self._query_sdoh_factors(session, member_id)

            # Query governing policies
            governing_policies = await self._query_governing_policies(session, member_id)

            return MemberActiveState(
                member_id=member_id,
                active_diagnoses=active_diagnoses,
                active_prescriptions=active_prescriptions,
                sdoh_factors=sdoh_factors,
                governing_policies=governing_policies,
                last_updated=datetime.now(timezone.utc),
            )

    async def _query_active_diagnoses(
        self, session: AsyncSession, member_id: str
    ) -> list[dict[str, Any]]:
        """Query active diagnosis events for a member.

        Returns only events linked via HAS_CONDITION that are not marked
        as resolved, discontinued, closed, or superseded.
        """
        query = """
        MATCH (m:Member {member_id: $member_id})-[:HAS_CONDITION]->(e:Event)
        WHERE NOT coalesce(e.status, '') IN $inactive_statuses
        RETURN e
        """
        result = await session.run(
            query,
            member_id=member_id,
            inactive_statuses=list(INACTIVE_STATUSES),
        )
        records = await result.data()
        return [dict(record["e"]) for record in records]

    async def _query_active_prescriptions(
        self, session: AsyncSession, member_id: str
    ) -> list[dict[str, Any]]:
        """Query active prescription events for a member.

        Returns only events linked via IS_PRESCRIBED that are not marked
        as resolved, discontinued, closed, or superseded.
        """
        query = """
        MATCH (m:Member {member_id: $member_id})-[:IS_PRESCRIBED]->(e:Event)
        WHERE NOT coalesce(e.status, '') IN $inactive_statuses
        RETURN e
        """
        result = await session.run(
            query,
            member_id=member_id,
            inactive_statuses=list(INACTIVE_STATUSES),
        )
        records = await result.data()
        return [dict(record["e"]) for record in records]

    async def _query_sdoh_factors(
        self, session: AsyncSession, member_id: str
    ) -> list[dict[str, Any]]:
        """Query SDOH factors linked to a member's conditions.

        Traverses Member -> HAS_CONDITION -> Event -> EVIDENCED_BY -> EvidenceSource
        and also finds SDOH_Factor nodes linked via INFERRED_FROM relationships.
        Only returns factors not marked as inactive.
        """
        query = """
        MATCH (m:Member {member_id: $member_id})-[:HAS_CONDITION]->(e:Event)
        WHERE NOT coalesce(e.status, '') IN $inactive_statuses
        WITH e
        OPTIONAL MATCH (s:SDOH_Factor)-[:INFERRED_FROM]->(ev:EvidenceSource)<-[:EVIDENCED_BY]-(e)
        WHERE NOT coalesce(s.status, '') IN $inactive_statuses
        RETURN DISTINCT s
        UNION
        MATCH (m:Member {member_id: $member_id})-[:HAS_CONDITION]->(e:Event)
        WHERE NOT coalesce(e.status, '') IN $inactive_statuses
        WITH m
        OPTIONAL MATCH (m)-[:HAS_SDOH]->(s:SDOH_Factor)
        WHERE NOT coalesce(s.status, '') IN $inactive_statuses
        RETURN DISTINCT s
        """
        result = await session.run(
            query,
            member_id=member_id,
            inactive_statuses=list(INACTIVE_STATUSES),
        )
        records = await result.data()
        return [dict(record["s"]) for record in records if record["s"] is not None]

    async def _query_governing_policies(
        self, session: AsyncSession, member_id: str
    ) -> list[dict[str, Any]]:
        """Query policy rules governing a member's active conditions.

        Traverses Member -> HAS_CONDITION/IS_PRESCRIBED -> Event -> GOVERNED_BY -> PolicyRule.
        Only returns policies for active events.
        """
        query = """
        MATCH (m:Member {member_id: $member_id})-[:HAS_CONDITION|IS_PRESCRIBED]->(e:Event)
              -[:GOVERNED_BY]->(p:PolicyRule)
        WHERE NOT coalesce(e.status, '') IN $inactive_statuses
        RETURN DISTINCT p
        """
        result = await session.run(
            query,
            member_id=member_id,
            inactive_statuses=list(INACTIVE_STATUSES),
        )
        records = await result.data()
        return [dict(record["p"]) for record in records]

    async def upsert_node(
        self,
        node_type: str,
        node_id: str,
        properties: dict[str, Any],
        execution_id: Optional[str] = None,
    ) -> None:
        """Upsert a graph node with optional agent execution provenance.

        Uses MERGE to create or update nodes, preserving existing relationships.
        If execution_id is provided, it is recorded as provenance metadata.

        Args:
            node_type: The node label (Member, Event, PolicyRule, SDOH_Factor, EvidenceSource).
            node_id: The unique identifier for the node.
            properties: Properties to set on the node.
            execution_id: Optional agent execution ID for provenance tracking.

        Raises:
            ValueError: If node_type is not a valid graph node type.
        """
        if node_type not in VALID_NODE_TYPES:
            raise ValueError(
                f"Invalid node_type '{node_type}'. Must be one of: {sorted(VALID_NODE_TYPES)}"
            )

        # Determine the ID field based on node type
        id_field = self._get_id_field(node_type)

        # Add provenance if execution_id is provided
        props = dict(properties)
        if execution_id:
            props["_last_execution_id"] = execution_id
            props["_last_updated_at"] = datetime.now(timezone.utc).isoformat()

        # Build MERGE + SET query
        query = f"""
        MERGE (n:{node_type} {{{id_field}: $node_id}})
        SET n += $properties
        """

        async with self.driver.session() as session:
            await session.run(query, node_id=node_id, properties=props)

        logger.debug(
            f"Upserted {node_type} node with {id_field}={node_id}"
            + (f" (execution_id={execution_id})" if execution_id else "")
        )

    async def upsert_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: dict[str, Any],
    ) -> None:
        """Upsert a directed relationship between two nodes.

        Uses MERGE to create or update the relationship without duplicating.
        Source and target nodes must already exist in the graph.

        Args:
            source_id: The unique identifier of the source node.
            target_id: The unique identifier of the target node.
            rel_type: The relationship type (HAS_CONDITION, IS_PRESCRIBED, etc.).
            properties: Properties to set on the relationship.

        Raises:
            ValueError: If rel_type is not a valid relationship type.
        """
        if rel_type not in VALID_RELATIONSHIP_TYPES:
            raise ValueError(
                f"Invalid rel_type '{rel_type}'. Must be one of: {sorted(VALID_RELATIONSHIP_TYPES)}"
            )

        # Add timestamp to relationship properties
        props = dict(properties)
        props["_updated_at"] = datetime.now(timezone.utc).isoformat()

        # Use a generic match that finds nodes by any ID field
        query = f"""
        MATCH (source) WHERE source.member_id = $source_id 
            OR source.event_id = $source_id 
            OR source.policy_id = $source_id 
            OR source.sdoh_id = $source_id 
            OR source.evidence_id = $source_id
        MATCH (target) WHERE target.member_id = $target_id 
            OR target.event_id = $target_id 
            OR target.policy_id = $target_id 
            OR target.sdoh_id = $target_id 
            OR target.evidence_id = $target_id
        MERGE (source)-[r:{rel_type}]->(target)
        SET r += $properties
        """

        async with self.driver.session() as session:
            await session.run(query, source_id=source_id, target_id=target_id, properties=props)

        logger.debug(f"Upserted relationship {source_id}-[:{rel_type}]->{target_id}")

    async def query_related_evidence(
        self, member_id: str, condition_code: str
    ) -> list[dict[str, Any]]:
        """Retrieve evidence nodes linked to a member's specific condition.

        Traverses:
        Member -[:HAS_CONDITION]-> Event (matching condition_code)
               -[:EVIDENCED_BY]-> EvidenceSource

        Args:
            member_id: The unique member identifier.
            condition_code: The condition code (e.g., ICD-10 code) to filter by.

        Returns:
            List of EvidenceSource node properties linked to the condition.
        """
        query = """
        MATCH (m:Member {member_id: $member_id})-[:HAS_CONDITION]->(e:Event)
        WHERE e.condition_code = $condition_code
          AND NOT coalesce(e.status, '') IN $inactive_statuses
        MATCH (e)-[:EVIDENCED_BY]->(ev:EvidenceSource)
        RETURN ev
        """

        async with self.driver.session() as session:
            result = await session.run(
                query,
                member_id=member_id,
                condition_code=condition_code,
                inactive_statuses=list(INACTIVE_STATUSES),
            )
            records = await result.data()
            return [dict(record["ev"]) for record in records]

    @staticmethod
    def _get_id_field(node_type: str) -> str:
        """Return the unique ID field name for a given node type.

        Args:
            node_type: The node label.

        Returns:
            The corresponding ID field name.
        """
        id_fields = {
            "Member": "member_id",
            "Event": "event_id",
            "PolicyRule": "policy_id",
            "SDOH_Factor": "sdoh_id",
            "EvidenceSource": "evidence_id",
        }
        return id_fields[node_type]
