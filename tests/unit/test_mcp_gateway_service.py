"""Unit tests for MCPGatewayService (BEACON Layer 3).

Tests tool catalog validation, sandboxed execution, timeout handling,
invocation logging, and unapproved tool rejection.

Validates:
    - validate_tool_request() accepts approved tools with valid parameters
    - validate_tool_request() rejects tools not in catalog
    - validate_tool_request() rejects parameters that don't conform to schema
    - invoke_tool() executes approved tools successfully in sandbox
    - invoke_tool() records successful invocations with all required fields
    - invoke_tool() handles timeout with structured error and no retry
    - invoke_tool() handles execution errors with error category
    - invoke_tool() rejects unapproved tools with logging
    - All invocations (success or fail) are recorded in trace

Requirements referenced: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6
"""

import asyncio
from datetime import datetime, timezone

import pytest

from clinical_reasoning_fabric.beacon.mcp_gateway_service import (
    DefaultSandboxExecutor,
    MCPGatewayService,
    SandboxExecutor,
    ToolCatalog,
)
from clinical_reasoning_fabric.models.core import ToolDefinition, ToolResult
from clinical_reasoning_fabric.models.exceptions import ToolValidationError


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def tool_definitions():
    """Standard set of approved tool definitions for testing."""
    return [
        ToolDefinition(
            tool_name="qdrant_retrieval",
            description="Retrieve clinical evidence from Qdrant vector store",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 50},
                    "namespace": {"type": "string"},
                },
                "required": ["query"],
            },
            timeout_seconds=10,
        ),
        ToolDefinition(
            tool_name="graph_query",
            description="Query the causal ontology graph",
            input_schema={
                "type": "object",
                "properties": {
                    "member_id": {"type": "string"},
                    "query_type": {
                        "type": "string",
                        "enum": ["active_state", "related_evidence"],
                    },
                },
                "required": ["member_id", "query_type"],
            },
            timeout_seconds=5,
        ),
        ToolDefinition(
            tool_name="kms_verify",
            description="Verify KMS signature on a document",
            input_schema={
                "type": "object",
                "properties": {
                    "document_id": {"type": "string"},
                    "content_hash": {"type": "string"},
                },
                "required": ["document_id", "content_hash"],
            },
            timeout_seconds=30,
        ),
    ]


@pytest.fixture
def tool_catalog(tool_definitions):
    """ToolCatalog populated with test tools."""
    return ToolCatalog(tools=tool_definitions)


@pytest.fixture
def sandbox():
    """DefaultSandboxExecutor with registered handlers."""
    executor = DefaultSandboxExecutor()

    async def qdrant_handler(params):
        return {"chunks": [{"id": "chunk-1", "text": "sample"}], "count": 1}

    async def graph_handler(params):
        return {"member_id": params["member_id"], "diagnoses": ["diabetes"]}

    async def kms_handler(params):
        return {"valid": True, "document_id": params["document_id"]}

    executor.register_handler("qdrant_retrieval", qdrant_handler)
    executor.register_handler("graph_query", graph_handler)
    executor.register_handler("kms_verify", kms_handler)

    return executor


@pytest.fixture
def gateway(tool_catalog, sandbox):
    """MCPGatewayService configured with test catalog and sandbox."""
    return MCPGatewayService(tool_catalog=tool_catalog, sandbox=sandbox)


# =============================================================================
# ToolCatalog Tests
# =============================================================================


class TestToolCatalog:
    """Tests for ToolCatalog management."""

    def test_catalog_has_tool(self, tool_catalog):
        """Catalog reports tools that exist."""
        assert tool_catalog.has_tool("qdrant_retrieval") is True
        assert tool_catalog.has_tool("graph_query") is True
        assert tool_catalog.has_tool("kms_verify") is True

    def test_catalog_missing_tool(self, tool_catalog):
        """Catalog reports tools that don't exist."""
        assert tool_catalog.has_tool("unauthorized_tool") is False
        assert tool_catalog.has_tool("") is False

    def test_catalog_get_tool(self, tool_catalog):
        """get_tool returns the definition for existing tools."""
        tool = tool_catalog.get_tool("qdrant_retrieval")
        assert tool is not None
        assert tool.tool_name == "qdrant_retrieval"
        assert tool.timeout_seconds == 10

    def test_catalog_get_tool_returns_none_for_missing(self, tool_catalog):
        """get_tool returns None for tools not in catalog."""
        assert tool_catalog.get_tool("nonexistent") is None

    def test_catalog_list_tools(self, tool_catalog):
        """list_tools returns all approved tools."""
        tools = tool_catalog.list_tools()
        assert len(tools) == 3
        names = {t.tool_name for t in tools}
        assert names == {"qdrant_retrieval", "graph_query", "kms_verify"}

    def test_catalog_add_tool(self, tool_catalog):
        """add_tool adds a new tool to the catalog."""
        new_tool = ToolDefinition(
            tool_name="new_tool",
            description="A new tool",
            input_schema={"type": "object", "properties": {}},
            timeout_seconds=15,
        )
        tool_catalog.add_tool(new_tool)
        assert tool_catalog.has_tool("new_tool") is True

    def test_catalog_remove_tool(self, tool_catalog):
        """remove_tool removes an existing tool."""
        assert tool_catalog.remove_tool("kms_verify") is True
        assert tool_catalog.has_tool("kms_verify") is False

    def test_catalog_remove_nonexistent_tool(self, tool_catalog):
        """remove_tool returns False for non-existent tools."""
        assert tool_catalog.remove_tool("nonexistent") is False


# =============================================================================
# Validate Tool Request Tests
# =============================================================================


class TestValidateToolRequest:
    """Tests for validate_tool_request (Requirement 6.2)."""

    def test_valid_tool_with_valid_parameters(self, gateway):
        """Approved tool with conforming parameters returns True."""
        result = gateway.validate_tool_request(
            "qdrant_retrieval",
            {"query": "diabetes treatment", "top_k": 10},
        )
        assert result is True

    def test_valid_tool_with_required_only(self, gateway):
        """Approved tool with only required parameters is valid."""
        result = gateway.validate_tool_request(
            "qdrant_retrieval",
            {"query": "clinical notes"},
        )
        assert result is True

    def test_unapproved_tool_raises_validation_error(self, gateway):
        """Tool not in catalog raises ToolValidationError."""
        with pytest.raises(ToolValidationError) as exc_info:
            gateway.validate_tool_request("unauthorized_tool", {"key": "value"})

        error = exc_info.value
        assert error.tool_name == "unauthorized_tool"
        assert "not in the approved catalog" in error.reason

    def test_missing_required_parameters_raises_validation_error(self, gateway):
        """Missing required parameters raises ToolValidationError."""
        with pytest.raises(ToolValidationError) as exc_info:
            gateway.validate_tool_request("graph_query", {"member_id": "m-001"})

        error = exc_info.value
        assert error.tool_name == "graph_query"
        assert error.validation_errors is not None
        assert len(error.validation_errors) > 0

    def test_wrong_parameter_type_raises_validation_error(self, gateway):
        """Parameter with wrong type raises ToolValidationError."""
        with pytest.raises(ToolValidationError) as exc_info:
            gateway.validate_tool_request(
                "qdrant_retrieval",
                {"query": "test", "top_k": "not_an_integer"},
            )

        error = exc_info.value
        assert error.tool_name == "qdrant_retrieval"
        assert error.validation_errors is not None

    def test_parameter_out_of_range_raises_validation_error(self, gateway):
        """Parameter violating schema constraints raises ToolValidationError."""
        with pytest.raises(ToolValidationError) as exc_info:
            gateway.validate_tool_request(
                "qdrant_retrieval",
                {"query": "test", "top_k": 0},  # minimum is 1
            )

        error = exc_info.value
        assert error.tool_name == "qdrant_retrieval"

    def test_invalid_enum_value_raises_validation_error(self, gateway):
        """Parameter not matching enum constraint raises ToolValidationError."""
        with pytest.raises(ToolValidationError) as exc_info:
            gateway.validate_tool_request(
                "graph_query",
                {"member_id": "m-001", "query_type": "invalid_type"},
            )

        error = exc_info.value
        assert error.tool_name == "graph_query"


# =============================================================================
# Invoke Tool - Success Tests
# =============================================================================


class TestInvokeToolSuccess:
    """Tests for successful tool invocation (Requirements 6.4, 6.5)."""

    async def test_invoke_approved_tool_returns_success(self, gateway):
        """Invoking an approved tool with valid params returns success ToolResult."""
        result = await gateway.invoke_tool(
            "qdrant_retrieval",
            {"query": "diabetes mellitus"},
            "agent-001",
        )

        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.tool_name == "qdrant_retrieval"
        assert result.agent_identity == "agent-001"
        assert result.output_result is not None
        assert result.error_category is None

    async def test_invoke_records_all_required_fields(self, gateway):
        """Successful invocation records all 5 required fields (Requirement 6.5)."""
        result = await gateway.invoke_tool(
            "graph_query",
            {"member_id": "m-123", "query_type": "active_state"},
            "agent-002",
        )

        # Required fields: tool_name, input_parameters, output_result, duration_ms, success
        assert result.tool_name == "graph_query"
        assert result.input_parameters == {"member_id": "m-123", "query_type": "active_state"}
        assert result.output_result is not None
        assert result.duration_ms >= 0
        assert result.success is True

    async def test_invoke_records_timing(self, gateway):
        """Invocation duration_ms is recorded as non-negative integer."""
        result = await gateway.invoke_tool(
            "kms_verify",
            {"document_id": "doc-1", "content_hash": "abc123"},
            "agent-001",
        )

        assert isinstance(result.duration_ms, int)
        assert result.duration_ms >= 0

    async def test_invoke_records_timestamp(self, gateway):
        """Invocation has an invoked_at timestamp."""
        result = await gateway.invoke_tool(
            "qdrant_retrieval",
            {"query": "test"},
            "agent-001",
        )

        assert isinstance(result.invoked_at, datetime)

    async def test_successful_invocation_stored_in_log(self, gateway):
        """Successful invocations are recorded in the invocation log."""
        await gateway.invoke_tool("qdrant_retrieval", {"query": "test"}, "agent-001")

        log = gateway.invocation_log
        assert len(log) == 1
        assert log[0].tool_name == "qdrant_retrieval"
        assert log[0].success is True


# =============================================================================
# Invoke Tool - Rejection Tests
# =============================================================================


class TestInvokeToolRejection:
    """Tests for unapproved tool rejection (Requirement 6.3)."""

    async def test_unapproved_tool_raises_and_logs(self, gateway, caplog):
        """Invoking unapproved tool raises ToolValidationError and logs."""
        with pytest.raises(ToolValidationError) as exc_info:
            await gateway.invoke_tool(
                "unauthorized_tool",
                {"data": "malicious"},
                "agent-rogue",
            )

        error = exc_info.value
        assert error.tool_name == "unauthorized_tool"

        # Requirement 6.3: Log includes agent identity, tool name, timestamp
        log_output = caplog.text
        assert "agent-rogue" in log_output
        assert "unauthorized_tool" in log_output

    async def test_unapproved_tool_recorded_in_log(self, gateway):
        """Rejected invocations are still recorded in the invocation log."""
        with pytest.raises(ToolValidationError):
            await gateway.invoke_tool(
                "unauthorized_tool",
                {"data": "test"},
                "agent-001",
            )

        log = gateway.invocation_log
        assert len(log) == 1
        assert log[0].success is False
        assert log[0].error_category == "validation_error"

    async def test_invalid_parameters_raises_and_records(self, gateway):
        """Invalid parameters raise ToolValidationError and record failure."""
        with pytest.raises(ToolValidationError):
            await gateway.invoke_tool(
                "graph_query",
                {"member_id": "m-001"},  # missing required query_type
                "agent-001",
            )

        log = gateway.invocation_log
        assert len(log) == 1
        assert log[0].success is False
        assert log[0].error_category == "validation_error"


# =============================================================================
# Invoke Tool - Timeout Tests
# =============================================================================


class TestInvokeToolTimeout:
    """Tests for tool timeout handling (Requirement 6.6)."""

    async def test_timeout_returns_structured_error(self, tool_catalog):
        """Timed-out tool returns ToolResult with success=False and error_category='timeout'."""
        # Create a slow handler that exceeds the tool's timeout
        executor = DefaultSandboxExecutor()

        async def slow_handler(params):
            await asyncio.sleep(20)  # Exceeds 10s timeout
            return {"result": "too late"}

        executor.register_handler("qdrant_retrieval", slow_handler)
        gateway = MCPGatewayService(tool_catalog=tool_catalog, sandbox=executor)

        result = await gateway.invoke_tool(
            "qdrant_retrieval",
            {"query": "slow query"},
            "agent-001",
        )

        assert result.success is False
        assert result.error_category == "timeout"
        assert result.output_result is None
        assert result.tool_name == "qdrant_retrieval"
        assert result.agent_identity == "agent-001"

    async def test_timeout_does_not_retry(self, tool_catalog):
        """Timed-out tool is not retried (Requirement 6.6)."""
        call_count = 0
        executor = DefaultSandboxExecutor()

        async def slow_handler(params):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(20)
            return {"result": "done"}

        executor.register_handler("qdrant_retrieval", slow_handler)
        gateway = MCPGatewayService(tool_catalog=tool_catalog, sandbox=executor)

        await gateway.invoke_tool("qdrant_retrieval", {"query": "test"}, "agent-001")

        # Handler should only be called once — no retries
        assert call_count == 1

    async def test_timeout_recorded_in_log(self, tool_catalog):
        """Timeout failure is recorded in the invocation log."""
        executor = DefaultSandboxExecutor()

        async def slow_handler(params):
            await asyncio.sleep(20)
            return {}

        executor.register_handler("qdrant_retrieval", slow_handler)
        gateway = MCPGatewayService(tool_catalog=tool_catalog, sandbox=executor)

        await gateway.invoke_tool("qdrant_retrieval", {"query": "test"}, "agent-001")

        log = gateway.invocation_log
        assert len(log) == 1
        assert log[0].success is False
        assert log[0].error_category == "timeout"


# =============================================================================
# Invoke Tool - Error Handling Tests
# =============================================================================


class TestInvokeToolErrors:
    """Tests for tool crash/error handling (Requirement 6.6)."""

    async def test_crash_returns_structured_error(self, tool_catalog):
        """Tool that crashes returns ToolResult with success=False and error category."""
        executor = DefaultSandboxExecutor()

        async def crashing_handler(params):
            raise RuntimeError("Unexpected internal error")

        executor.register_handler("qdrant_retrieval", crashing_handler)
        gateway = MCPGatewayService(tool_catalog=tool_catalog, sandbox=executor)

        result = await gateway.invoke_tool(
            "qdrant_retrieval",
            {"query": "will crash"},
            "agent-001",
        )

        assert result.success is False
        assert result.error_category == "runtime_error"
        assert result.output_result is None

    async def test_connection_error_classified(self, tool_catalog):
        """ConnectionError is classified with appropriate error category."""
        executor = DefaultSandboxExecutor()

        async def conn_error_handler(params):
            raise ConnectionError("Service unavailable")

        executor.register_handler("qdrant_retrieval", conn_error_handler)
        gateway = MCPGatewayService(tool_catalog=tool_catalog, sandbox=executor)

        result = await gateway.invoke_tool(
            "qdrant_retrieval",
            {"query": "test"},
            "agent-001",
        )

        assert result.success is False
        assert result.error_category == "connection_error"

    async def test_value_error_classified(self, tool_catalog):
        """ValueError is classified correctly."""
        executor = DefaultSandboxExecutor()

        async def bad_value_handler(params):
            raise ValueError("Invalid input data")

        executor.register_handler("graph_query", bad_value_handler)
        gateway = MCPGatewayService(tool_catalog=tool_catalog, sandbox=executor)

        result = await gateway.invoke_tool(
            "graph_query",
            {"member_id": "m-001", "query_type": "active_state"},
            "agent-001",
        )

        assert result.success is False
        assert result.error_category == "value_error"

    async def test_error_does_not_retry(self, tool_catalog):
        """Failed tool is not retried (Requirement 6.6)."""
        call_count = 0
        executor = DefaultSandboxExecutor()

        async def failing_handler(params):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Always fails")

        executor.register_handler("qdrant_retrieval", failing_handler)
        gateway = MCPGatewayService(tool_catalog=tool_catalog, sandbox=executor)

        await gateway.invoke_tool("qdrant_retrieval", {"query": "test"}, "agent-001")

        assert call_count == 1

    async def test_error_recorded_in_log(self, tool_catalog):
        """Error failure is recorded in the invocation log with all fields."""
        executor = DefaultSandboxExecutor()

        async def failing_handler(params):
            raise RuntimeError("Crash")

        executor.register_handler("kms_verify", failing_handler)
        gateway = MCPGatewayService(tool_catalog=tool_catalog, sandbox=executor)

        result = await gateway.invoke_tool(
            "kms_verify",
            {"document_id": "doc-1", "content_hash": "hash1"},
            "agent-003",
        )

        log = gateway.invocation_log
        assert len(log) == 1
        entry = log[0]
        assert entry.tool_name == "kms_verify"
        assert entry.input_parameters == {"document_id": "doc-1", "content_hash": "hash1"}
        assert entry.output_result is None
        assert entry.duration_ms >= 0
        assert entry.success is False
        assert entry.error_category is not None
        assert entry.agent_identity == "agent-003"


# =============================================================================
# Invocation Log Completeness Tests
# =============================================================================


class TestInvocationLogCompleteness:
    """Tests for invocation trace recording (Requirement 6.5)."""

    async def test_all_invocations_recorded(self, gateway):
        """Both successful and failed invocations are recorded."""
        # Successful invocation
        await gateway.invoke_tool("qdrant_retrieval", {"query": "test"}, "agent-001")

        # Failed invocation (unapproved tool)
        with pytest.raises(ToolValidationError):
            await gateway.invoke_tool("bad_tool", {}, "agent-002")

        log = gateway.invocation_log
        assert len(log) == 2
        assert log[0].success is True
        assert log[1].success is False

    async def test_each_log_entry_has_required_fields(self, gateway):
        """Every log entry contains: tool_name, input_parameters, output_result, duration_ms, success."""
        await gateway.invoke_tool(
            "graph_query",
            {"member_id": "m-001", "query_type": "active_state"},
            "agent-001",
        )

        log = gateway.invocation_log
        assert len(log) == 1
        entry = log[0]

        # All 5 required fields present (Requirement 6.5)
        assert entry.tool_name is not None and entry.tool_name != ""
        assert entry.input_parameters is not None
        assert entry.duration_ms is not None and entry.duration_ms >= 0
        assert entry.success is not None
        # output_result can be None for failures, but field must exist
        assert hasattr(entry, "output_result")

    async def test_multiple_invocations_maintain_order(self, gateway):
        """Invocation log maintains chronological order."""
        await gateway.invoke_tool("qdrant_retrieval", {"query": "first"}, "agent-001")
        await gateway.invoke_tool("graph_query", {"member_id": "m-1", "query_type": "active_state"}, "agent-001")
        await gateway.invoke_tool("kms_verify", {"document_id": "d-1", "content_hash": "h-1"}, "agent-001")

        log = gateway.invocation_log
        assert len(log) == 3
        assert log[0].tool_name == "qdrant_retrieval"
        assert log[1].tool_name == "graph_query"
        assert log[2].tool_name == "kms_verify"
