"""Property-based tests for Graph Visualization Node Capacity.

**Validates: Requirements 15.5**

Property 33: Graph Visualization Node Capacity
- For any member state with N nodes (N ≤ 200), the graph API endpoint returns
  exactly N typed nodes with correct type labels and all corresponding labeled
  directed edges.
- The system supports up to 200 nodes in the rendered graph view.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from fastapi import FastAPI
from hypothesis import given, settings
from hypothesis import strategies as st

from src.clinical_reasoning_fabric.frontend.api_endpoints import (
    create_evidence_graph_router,
    store_member_graph,
    get_member_graph_store,
)


# =============================================================================
# Hypothesis Strategies
# =============================================================================

# Valid node types
node_type_strategy = st.sampled_from([
    "diagnosis", "medication", "sdoh_factor", "policy_rule", "member",
])

# Valid edge types
edge_type_strategy = st.sampled_from([
    "HAS_CONDITION", "IS_PRESCRIBED", "TRIGGERED_BY",
    "GOVERNED_BY", "EVIDENCED_BY", "INFERRED_FROM",
])

# Non-empty identifier strings
identifier_strategy = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789-_",
    min_size=1,
    max_size=30,
)

# Human-readable labels
label_strategy = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -_.",
    min_size=1,
    max_size=60,
)


@st.composite
def graph_node_strategy(draw, node_id: str | None = None):
    """Generate a single valid graph node dictionary."""
    return {
        "id": node_id or draw(identifier_strategy),
        "type": draw(node_type_strategy),
        "label": draw(label_strategy),
        "properties": {},
    }


@st.composite
def member_graph_strategy(draw):
    """Generate a member graph with 1-200 nodes and valid edges.

    Edges only reference node IDs that exist in the nodes list.
    """
    num_nodes = draw(st.integers(min_value=1, max_value=200))

    # Generate unique node IDs
    node_ids = []
    used_ids: set[str] = set()
    for i in range(num_nodes):
        node_id = f"node-{i}"
        used_ids.add(node_id)
        node_ids.append(node_id)

    # Generate nodes with assigned IDs
    nodes = []
    for node_id in node_ids:
        node = {
            "id": node_id,
            "type": draw(node_type_strategy),
            "label": draw(label_strategy),
            "properties": {},
        }
        nodes.append(node)

    # Generate edges between existing nodes (0 to min(num_nodes, 50) edges)
    max_edges = min(num_nodes, 50)
    num_edges = draw(st.integers(min_value=0, max_value=max_edges))
    edges = []
    for _ in range(num_edges):
        source_idx = draw(st.integers(min_value=0, max_value=num_nodes - 1))
        target_idx = draw(st.integers(min_value=0, max_value=num_nodes - 1))
        if source_idx != target_idx:
            edges.append({
                "source": node_ids[source_idx],
                "target": node_ids[target_idx],
                "type": draw(edge_type_strategy),
                "label": draw(label_strategy),
            })

    return {"nodes": nodes, "edges": edges}


# =============================================================================
# Helpers
# =============================================================================


def _create_app():
    """Create a fresh FastAPI app with the evidence/graph router."""
    app = FastAPI()
    router = create_evidence_graph_router()
    app.include_router(router)
    return app


async def _store_graph_and_query(member_id: str, graph_data: dict) -> dict:
    """Store a member graph and query the API endpoint.

    Returns the API response as a parsed dictionary.
    """
    # Clear the store for isolation
    get_member_graph_store().clear()

    app = _create_app()

    # Store the graph
    store_member_graph(member_id, graph_data)

    # Query the API endpoint
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get(f"/api/graph/member/{member_id}")
        assert response.status_code == 200
        return response.json()


# =============================================================================
# Property 33: Graph Visualization Node Capacity
# =============================================================================


@pytest.mark.property
class TestGraphVisualizationNodeCapacity:
    """Property 33: Graph Visualization Node Capacity.

    **Validates: Requirements 15.5**

    For any member state with N nodes (N ≤ 200), the graph API endpoint
    returns exactly N typed nodes with correct type labels and all
    corresponding labeled directed edges.
    """

    @given(graph_data=member_graph_strategy())
    @settings(max_examples=100)
    def test_all_nodes_returned_in_response(self, graph_data: dict):
        """API returns exactly N nodes for a graph with N stored nodes.

        **Validates: Requirements 15.5**

        For any graph with 1-200 nodes, the API response must contain
        exactly the same number of nodes as stored.
        """
        member_id = "member-prop33-count"
        result = asyncio.get_event_loop().run_until_complete(
            _store_graph_and_query(member_id, graph_data)
        )

        expected_count = len(graph_data["nodes"])
        actual_count = len(result["nodes"])
        assert actual_count == expected_count, (
            f"Expected {expected_count} nodes, got {actual_count}"
        )

    @given(graph_data=member_graph_strategy())
    @settings(max_examples=100)
    def test_all_edges_returned_in_response(self, graph_data: dict):
        """API returns exactly M edges for a graph with M stored edges.

        **Validates: Requirements 15.5**

        For any graph with edges, the API response must contain exactly
        the same number of edges as stored.
        """
        member_id = "member-prop33-edges"
        result = asyncio.get_event_loop().run_until_complete(
            _store_graph_and_query(member_id, graph_data)
        )

        expected_count = len(graph_data["edges"])
        actual_count = len(result["edges"])
        assert actual_count == expected_count, (
            f"Expected {expected_count} edges, got {actual_count}"
        )

    @given(graph_data=member_graph_strategy())
    @settings(max_examples=100)
    def test_node_ids_match_stored_data(self, graph_data: dict):
        """Node IDs in the response match the originally stored node IDs.

        **Validates: Requirements 15.5**

        The set of node IDs returned by the API must exactly match the
        set of node IDs that were stored.
        """
        member_id = "member-prop33-ids"
        result = asyncio.get_event_loop().run_until_complete(
            _store_graph_and_query(member_id, graph_data)
        )

        stored_ids = {n["id"] for n in graph_data["nodes"]}
        response_ids = {n["id"] for n in result["nodes"]}
        assert stored_ids == response_ids, (
            f"Node ID mismatch. Missing: {stored_ids - response_ids}, "
            f"Extra: {response_ids - stored_ids}"
        )

    @given(graph_data=member_graph_strategy())
    @settings(max_examples=100)
    def test_all_nodes_have_valid_type(self, graph_data: dict):
        """Every node in the response has a non-empty type field.

        **Validates: Requirements 15.5**

        Node types must be preserved from the stored data and present
        in every response node.
        """
        member_id = "member-prop33-types"
        result = asyncio.get_event_loop().run_until_complete(
            _store_graph_and_query(member_id, graph_data)
        )

        valid_types = {"diagnosis", "medication", "sdoh_factor", "policy_rule", "member"}
        for i, node in enumerate(result["nodes"]):
            assert "type" in node, f"node[{i}] is missing 'type' field"
            assert node["type"] is not None and len(node["type"]) > 0, (
                f"node[{i}] has empty type"
            )
            assert node["type"] in valid_types, (
                f"node[{i}] has invalid type '{node['type']}'"
            )

    @given(graph_data=member_graph_strategy())
    @settings(max_examples=100)
    def test_all_edges_have_valid_type(self, graph_data: dict):
        """Every edge in the response has a non-empty type field.

        **Validates: Requirements 15.5**

        Edge types must be preserved from the stored data and present
        in every response edge.
        """
        member_id = "member-prop33-edge-types"
        result = asyncio.get_event_loop().run_until_complete(
            _store_graph_and_query(member_id, graph_data)
        )

        valid_edge_types = {
            "HAS_CONDITION", "IS_PRESCRIBED", "TRIGGERED_BY",
            "GOVERNED_BY", "EVIDENCED_BY", "INFERRED_FROM",
        }
        for i, edge in enumerate(result["edges"]):
            assert "type" in edge, f"edge[{i}] is missing 'type' field"
            assert edge["type"] is not None and len(edge["type"]) > 0, (
                f"edge[{i}] has empty type"
            )
            assert edge["type"] in valid_edge_types, (
                f"edge[{i}] has invalid type '{edge['type']}'"
            )

    @given(graph_data=member_graph_strategy())
    @settings(max_examples=100)
    def test_all_edges_reference_existing_nodes(self, graph_data: dict):
        """Every edge in the response references nodes that exist in the graph.

        **Validates: Requirements 15.5**

        Edge source and target IDs must correspond to existing node IDs
        in the response, ensuring graph consistency.
        """
        member_id = "member-prop33-edge-refs"
        result = asyncio.get_event_loop().run_until_complete(
            _store_graph_and_query(member_id, graph_data)
        )

        node_ids = {n["id"] for n in result["nodes"]}
        for i, edge in enumerate(result["edges"]):
            assert edge["source"] in node_ids, (
                f"edge[{i}] source '{edge['source']}' not in node set"
            )
            assert edge["target"] in node_ids, (
                f"edge[{i}] target '{edge['target']}' not in node set"
            )

    @given(graph_data=member_graph_strategy())
    @settings(max_examples=100)
    def test_all_nodes_have_label(self, graph_data: dict):
        """Every node in the response has a non-empty label field.

        **Validates: Requirements 15.5**

        Labels are required for display purposes and must be present
        and non-empty for every node.
        """
        member_id = "member-prop33-labels"
        result = asyncio.get_event_loop().run_until_complete(
            _store_graph_and_query(member_id, graph_data)
        )

        for i, node in enumerate(result["nodes"]):
            assert "label" in node, f"node[{i}] is missing 'label' field"
            assert node["label"] is not None and len(node["label"]) > 0, (
                f"node[{i}] has empty label"
            )
