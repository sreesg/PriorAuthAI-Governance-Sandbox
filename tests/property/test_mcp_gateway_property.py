"""Property-based tests for MCP Gateway Service.

**Validates: Requirements 6.2, 6.5**

Property 10: MCP Tool Catalog Validation
- Tool invocation accepted iff tool exists in catalog AND parameters match schema;
  all others rejected with ToolValidationError.

Property 11: Tool Invocation Record Completeness
- Every invocation trace entry contains all 5 required fields non-null:
  tool_name, input_parameters (may be empty dict), duration_ms (>= 0),
  success (bool), invoked_at (datetime).
"""

import asyncio
from datetime import datetime, timezone
from typing import Any

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from clinical_reasoning_fabric.beacon.mcp_gateway_service import (
    MCPGatewayService,
    ToolCatalog,
    DefaultSandboxExecutor,
)
from clinical_reasoning_fabric.models.core import ToolDefinition, ToolResult
from clinical_reasoning_fabric.models.exceptions import ToolValidationError


# =============================================================================
# Fixtures: Known tool catalog for testing
# =============================================================================

# A set of known tools with defined schemas
KNOWN_TOOLS = [
    ToolDefinition(
        tool_name="qdrant_retrieval",
        description="Retrieve clinical evidence from Qdrant vector store",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 50},
                "namespace": {"type": "string"},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        timeout_seconds=30,
    ),
    ToolDefinition(
        tool_name="graph_query",
        description="Query the Neo4j causal ontology graph",
        input_schema={
            "type": "object",
            "properties": {
                "member_id": {"type": "string", "minLength": 1},
                "condition_code": {"type": "string"},
            },
            "required": ["member_id"],
            "additionalProperties": False,
        },
        timeout_seconds=10,
    ),
    ToolDefinition(
        tool_name="kms_verify",
        description="Verify KMS signature on a document chunk",
        input_schema={
            "type": "object",
            "properties": {
                "document_id": {"type": "string", "minLength": 1},
                "content_hash": {"type": "string", "minLength": 64, "maxLength": 64},
            },
            "required": ["document_id", "content_hash"],
            "additionalProperties": False,
        },
        timeout_seconds=15,
    ),
]

KNOWN_TOOL_NAMES = {t.tool_name for t in KNOWN_TOOLS}


def _make_catalog() -> ToolCatalog:
    """Create a ToolCatalog populated with known test tools."""
    return ToolCatalog(tools=KNOWN_TOOLS)


def _make_sandbox() -> DefaultSandboxExecutor:
    """Create a sandbox executor with handlers for known tools."""
    sandbox = DefaultSandboxExecutor()

    async def _qdrant_handler(params: dict) -> dict:
        return {"chunks": [], "count": 0}

    async def _graph_handler(params: dict) -> dict:
        return {"member_id": params.get("member_id"), "state": "active"}

    async def _kms_handler(params: dict) -> dict:
        return {"valid": True, "document_id": params.get("document_id")}

    sandbox.register_handler("qdrant_retrieval", _qdrant_handler)
    sandbox.register_handler("graph_query", _graph_handler)
    sandbox.register_handler("kms_verify", _kms_handler)

    return sandbox


def _make_gateway() -> MCPGatewayService:
    """Create a fully configured MCPGatewayService for testing."""
    return MCPGatewayService(tool_catalog=_make_catalog(), sandbox=_make_sandbox())


def _make_failing_sandbox() -> DefaultSandboxExecutor:
    """Create a sandbox executor where tools raise exceptions."""
    sandbox = DefaultSandboxExecutor()

    async def _failing_handler(params: dict) -> dict:
        raise RuntimeError("Tool execution failed")

    for tool in KNOWN_TOOLS:
        sandbox.register_handler(tool.tool_name, _failing_handler)

    return sandbox


# =============================================================================
# Hypothesis Strategies
# =============================================================================

# Strategy for valid tool names (from the known catalog)
valid_tool_name_strategy = st.sampled_from(list(KNOWN_TOOL_NAMES))

# Strategy for tool names NOT in catalog
invalid_tool_name_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_-"),
    min_size=1,
    max_size=50,
).filter(lambda name: name not in KNOWN_TOOL_NAMES)

# Strategy for agent identity strings
agent_identity_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_-"),
    min_size=1,
    max_size=50,
)


@st.composite
def valid_params_for_tool_strategy(draw):
    """Generate valid parameters for a randomly chosen tool from the catalog."""
    tool_name = draw(valid_tool_name_strategy)

    if tool_name == "qdrant_retrieval":
        query = draw(st.text(min_size=1, max_size=100).filter(lambda s: s.strip() != ""))
        params = {"query": query}
        # Optionally add top_k
        if draw(st.booleans()):
            params["top_k"] = draw(st.integers(min_value=1, max_value=50))
        # Optionally add namespace
        if draw(st.booleans()):
            params["namespace"] = draw(st.text(min_size=0, max_size=50))

    elif tool_name == "graph_query":
        member_id = draw(st.text(min_size=1, max_size=50).filter(lambda s: s.strip() != ""))
        params = {"member_id": member_id}
        # Optionally add condition_code
        if draw(st.booleans()):
            params["condition_code"] = draw(st.text(min_size=0, max_size=20))

    elif tool_name == "kms_verify":
        document_id = draw(st.text(min_size=1, max_size=50).filter(lambda s: s.strip() != ""))
        # content_hash must be exactly 64 hex chars (SHA-256)
        content_hash = draw(
            st.text(
                alphabet="0123456789abcdef",
                min_size=64,
                max_size=64,
            )
        )
        params = {"document_id": document_id, "content_hash": content_hash}

    else:
        params = {}

    return tool_name, params


@st.composite
def invalid_params_for_tool_strategy(draw):
    """Generate parameters that violate the schema of a known tool.

    Strategies:
    - Missing required fields
    - Wrong types for fields
    - Extra fields when additionalProperties is False
    """
    tool_name = draw(valid_tool_name_strategy)
    violation_type = draw(st.sampled_from(["missing_required", "wrong_type", "extra_field"]))

    if violation_type == "missing_required":
        # Provide empty dict (missing all required fields)
        params = {}

    elif violation_type == "wrong_type":
        if tool_name == "qdrant_retrieval":
            # query should be string, provide integer
            params = {"query": draw(st.integers())}
        elif tool_name == "graph_query":
            # member_id should be string, provide integer
            params = {"member_id": draw(st.integers())}
        elif tool_name == "kms_verify":
            # document_id should be string, provide list
            params = {"document_id": [], "content_hash": "x" * 64}
        else:
            params = {"invalid": True}

    else:  # extra_field
        if tool_name == "qdrant_retrieval":
            params = {
                "query": "test query",
                "unknown_extra_field": draw(st.text(min_size=1, max_size=20)),
            }
        elif tool_name == "graph_query":
            params = {
                "member_id": "member-123",
                "unknown_extra_field": draw(st.text(min_size=1, max_size=20)),
            }
        elif tool_name == "kms_verify":
            params = {
                "document_id": "doc-123",
                "content_hash": "a" * 64,
                "unknown_extra_field": draw(st.text(min_size=1, max_size=20)),
            }
        else:
            params = {"extra": "value"}

    return tool_name, params


# =============================================================================
# Property 10: MCP Tool Catalog Validation
# =============================================================================


@pytest.mark.property
class TestMCPToolCatalogValidation:
    """Property 10: MCP Tool Catalog Validation.

    **Validates: Requirements 6.2**

    Tool invocation accepted iff tool exists in catalog AND parameters match
    schema; all others rejected with ToolValidationError.
    """

    @given(data=valid_params_for_tool_strategy())
    @settings(max_examples=200)
    def test_valid_tool_and_params_accepted(self, data):
        """Tools in catalog with valid parameters are accepted.

        **Validates: Requirements 6.2**

        For any tool that exists in the catalog with parameters conforming
        to its schema, validate_tool_request must return True.
        """
        tool_name, params = data
        gateway = _make_gateway()

        result = gateway.validate_tool_request(tool_name, params)
        assert result is True, (
            f"Expected tool '{tool_name}' with valid params to be accepted, "
            f"but validate_tool_request returned {result}"
        )

    @given(tool_name=invalid_tool_name_strategy, params=st.fixed_dictionaries({}))
    @settings(max_examples=200)
    def test_tool_not_in_catalog_rejected(self, tool_name, params):
        """Tools NOT in catalog are rejected with ToolValidationError.

        **Validates: Requirements 6.2**

        For any tool name that does not exist in the approved catalog,
        validate_tool_request must raise ToolValidationError.
        """
        gateway = _make_gateway()

        with pytest.raises(ToolValidationError) as exc_info:
            gateway.validate_tool_request(tool_name, params)

        assert exc_info.value.tool_name == tool_name, (
            f"ToolValidationError should reference tool '{tool_name}', "
            f"got '{exc_info.value.tool_name}'"
        )

    @given(data=invalid_params_for_tool_strategy())
    @settings(max_examples=200)
    def test_invalid_params_rejected(self, data):
        """Tools in catalog with invalid parameters are rejected.

        **Validates: Requirements 6.2**

        For any tool in the catalog where supplied parameters do NOT conform
        to the tool's permitted parameter schema, validate_tool_request must
        raise ToolValidationError.
        """
        tool_name, params = data
        gateway = _make_gateway()

        with pytest.raises(ToolValidationError) as exc_info:
            gateway.validate_tool_request(tool_name, params)

        assert exc_info.value.tool_name == tool_name
        assert exc_info.value.validation_errors is not None
        assert len(exc_info.value.validation_errors) > 0, (
            "ToolValidationError should include at least one validation error message"
        )

    @given(
        data=valid_params_for_tool_strategy(),
        agent_identity=agent_identity_strategy,
    )
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_valid_invocation_succeeds(self, data, agent_identity):
        """Valid tool + valid params invocation succeeds through invoke_tool.

        **Validates: Requirements 6.2**

        End-to-end: a valid tool with valid params and a registered handler
        should succeed when invoked through the gateway.
        """
        tool_name, params = data
        gateway = _make_gateway()

        result = await gateway.invoke_tool(tool_name, params, agent_identity)

        assert result.success is True, (
            f"Expected successful invocation for '{tool_name}', got success=False"
        )
        assert result.tool_name == tool_name

    @given(
        tool_name=invalid_tool_name_strategy,
        params=st.fixed_dictionaries({}),
        agent_identity=agent_identity_strategy,
    )
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_invalid_tool_invocation_raises(self, tool_name, params, agent_identity):
        """Invoking a tool NOT in catalog raises ToolValidationError.

        **Validates: Requirements 6.2**

        invoke_tool must raise ToolValidationError (not just return failure)
        when the tool is not in the approved catalog.
        """
        gateway = _make_gateway()

        with pytest.raises(ToolValidationError):
            await gateway.invoke_tool(tool_name, params, agent_identity)


# =============================================================================
# Property 11: Tool Invocation Record Completeness
# =============================================================================


@pytest.mark.property
class TestToolInvocationRecordCompleteness:
    """Property 11: Tool Invocation Record Completeness.

    **Validates: Requirements 6.5**

    Every invocation trace entry contains all 5 required fields non-null:
    tool_name, input_parameters (may be empty dict), duration_ms (>= 0),
    success (bool), invoked_at (datetime).
    """

    @given(
        data=valid_params_for_tool_strategy(),
        agent_identity=agent_identity_strategy,
    )
    @settings(max_examples=200)
    @pytest.mark.asyncio
    async def test_successful_invocation_has_complete_record(self, data, agent_identity):
        """Successful invocations produce records with all 5 required fields.

        **Validates: Requirements 6.5**

        After a successful tool invocation, the recorded ToolResult must have:
        - tool_name: non-null string
        - input_parameters: non-null dict (may be empty)
        - duration_ms: integer >= 0
        - success: True (bool)
        - invoked_at: non-null datetime
        """
        tool_name, params = data
        gateway = _make_gateway()

        result = await gateway.invoke_tool(tool_name, params, agent_identity)

        # Verify all 5 required fields on the result
        assert result.tool_name is not None and len(result.tool_name) > 0, (
            "tool_name must be non-null and non-empty"
        )
        assert result.input_parameters is not None, (
            "input_parameters must be non-null (may be empty dict)"
        )
        assert isinstance(result.input_parameters, dict), (
            f"input_parameters must be a dict, got {type(result.input_parameters)}"
        )
        assert result.duration_ms is not None and result.duration_ms >= 0, (
            f"duration_ms must be >= 0, got {result.duration_ms}"
        )
        assert isinstance(result.success, bool), (
            f"success must be a bool, got {type(result.success)}"
        )
        assert result.invoked_at is not None, "invoked_at must be non-null"
        assert isinstance(result.invoked_at, datetime), (
            f"invoked_at must be a datetime, got {type(result.invoked_at)}"
        )

    @given(
        data=valid_params_for_tool_strategy(),
        agent_identity=agent_identity_strategy,
    )
    @settings(max_examples=200)
    @pytest.mark.asyncio
    async def test_failed_invocation_has_complete_record(self, data, agent_identity):
        """Failed invocations also produce records with all 5 required fields.

        **Validates: Requirements 6.5**

        Even when a tool invocation fails (e.g., handler raises exception),
        the recorded ToolResult must have all 5 required fields non-null.
        """
        tool_name, params = data
        # Use a failing sandbox so invocations always error
        failing_gateway = MCPGatewayService(
            tool_catalog=_make_catalog(), sandbox=_make_failing_sandbox()
        )

        result = await failing_gateway.invoke_tool(tool_name, params, agent_identity)

        # Even failed invocations must have all required fields
        assert result.tool_name is not None and len(result.tool_name) > 0, (
            "tool_name must be non-null and non-empty on failure"
        )
        assert result.input_parameters is not None, (
            "input_parameters must be non-null on failure (may be empty dict)"
        )
        assert isinstance(result.input_parameters, dict), (
            f"input_parameters must be a dict on failure, got {type(result.input_parameters)}"
        )
        assert result.duration_ms is not None and result.duration_ms >= 0, (
            f"duration_ms must be >= 0 on failure, got {result.duration_ms}"
        )
        assert isinstance(result.success, bool), (
            f"success must be a bool on failure, got {type(result.success)}"
        )
        assert result.success is False, (
            "success should be False for a failed invocation"
        )
        assert result.invoked_at is not None, "invoked_at must be non-null on failure"
        assert isinstance(result.invoked_at, datetime), (
            f"invoked_at must be a datetime on failure, got {type(result.invoked_at)}"
        )

    @given(
        data=valid_params_for_tool_strategy(),
        agent_identity=agent_identity_strategy,
    )
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_invocation_log_records_match_results(self, data, agent_identity):
        """The invocation log entry matches the returned ToolResult.

        **Validates: Requirements 6.5**

        Every invocation must be recorded in the execution trace with the
        same data that is returned to the caller.
        """
        tool_name, params = data
        gateway = _make_gateway()

        result = await gateway.invoke_tool(tool_name, params, agent_identity)

        # Check that the invocation was logged
        log = gateway.invocation_log
        assert len(log) == 1, f"Expected 1 log entry, got {len(log)}"

        logged = log[0]
        assert logged.tool_name == result.tool_name
        assert logged.input_parameters == result.input_parameters
        assert logged.duration_ms == result.duration_ms
        assert logged.success == result.success
        assert logged.invoked_at == result.invoked_at

    @given(
        data=valid_params_for_tool_strategy(),
        agent_identity=agent_identity_strategy,
    )
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_invoked_at_is_utc_aware(self, data, agent_identity):
        """The invoked_at timestamp is timezone-aware (UTC).

        **Validates: Requirements 6.5**

        All timestamps in the invocation trace must be UTC-aware for
        consistent audit trail across time zones.
        """
        tool_name, params = data
        gateway = _make_gateway()

        result = await gateway.invoke_tool(tool_name, params, agent_identity)

        assert result.invoked_at.tzinfo is not None, (
            "invoked_at must be timezone-aware"
        )
        assert result.invoked_at.tzinfo == timezone.utc, (
            f"invoked_at timezone should be UTC, got {result.invoked_at.tzinfo}"
        )

    @given(
        invocations=st.lists(
            valid_params_for_tool_strategy(),
            min_size=2,
            max_size=5,
        ),
        agent_identity=agent_identity_strategy,
    )
    @settings(max_examples=50)
    @pytest.mark.asyncio
    async def test_multiple_invocations_all_recorded(self, invocations, agent_identity):
        """All invocations across multiple calls are recorded completely.

        **Validates: Requirements 6.5**

        The invocation trace must capture every invocation, not just the
        most recent one.
        """
        gateway = _make_gateway()

        for tool_name, params in invocations:
            await gateway.invoke_tool(tool_name, params, agent_identity)

        log = gateway.invocation_log
        assert len(log) == len(invocations), (
            f"Expected {len(invocations)} log entries, got {len(log)}"
        )

        # Verify each entry has all required fields
        for i, entry in enumerate(log):
            assert entry.tool_name is not None and len(entry.tool_name) > 0, (
                f"Log entry {i}: tool_name must be non-null and non-empty"
            )
            assert entry.input_parameters is not None, (
                f"Log entry {i}: input_parameters must be non-null"
            )
            assert entry.duration_ms >= 0, (
                f"Log entry {i}: duration_ms must be >= 0"
            )
            assert isinstance(entry.success, bool), (
                f"Log entry {i}: success must be a bool"
            )
            assert entry.invoked_at is not None, (
                f"Log entry {i}: invoked_at must be non-null"
            )
