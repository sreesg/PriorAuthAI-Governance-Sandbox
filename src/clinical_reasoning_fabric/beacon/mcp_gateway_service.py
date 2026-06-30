"""BEACON Layer 3 — MCP Gateway Service.

Controlled tool invocation gateway with catalog validation and sandboxed execution.
The MCP Gateway maintains a catalog of approved tools, validates invocations
against the catalog's schema, executes tools in a sandboxed environment with
configurable timeouts, and records all invocation traces.

Requirements referenced: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Optional

import jsonschema

from clinical_reasoning_fabric.models.core import ToolDefinition, ToolResult
from clinical_reasoning_fabric.models.exceptions import ToolValidationError

logger = logging.getLogger(__name__)


# =============================================================================
# Tool Catalog
# =============================================================================


class ToolCatalog:
    """Catalog of approved tools the agent may invoke.

    Requirement 6.1: Maintain a catalog of approved tools where each entry
    specifies the tool name, permitted input parameter schema, and description.
    """

    def __init__(self, tools: Optional[list[ToolDefinition]] = None):
        self._tools: dict[str, ToolDefinition] = {}
        if tools:
            for tool in tools:
                self._tools[tool.tool_name] = tool

    def get_tool(self, tool_name: str) -> Optional[ToolDefinition]:
        """Look up a tool definition by name. Returns None if not found."""
        return self._tools.get(tool_name)

    def has_tool(self, tool_name: str) -> bool:
        """Check if a tool exists in the catalog."""
        return tool_name in self._tools

    def list_tools(self) -> list[ToolDefinition]:
        """Return all approved tools in the catalog."""
        return list(self._tools.values())

    def add_tool(self, tool: ToolDefinition) -> None:
        """Add a tool to the catalog."""
        self._tools[tool.tool_name] = tool

    def remove_tool(self, tool_name: str) -> bool:
        """Remove a tool from the catalog. Returns True if removed, False if not found."""
        if tool_name in self._tools:
            del self._tools[tool_name]
            return True
        return False


# =============================================================================
# Sandbox Executor
# =============================================================================


class SandboxExecutor(ABC):
    """Abstract interface for sandboxed tool execution.

    Requirement 6.4: Execute calls within the sandboxed execution environment
    defined by Layer 4 subject to a configurable per-tool timeout.
    """

    @abstractmethod
    async def execute(
        self, tool_name: str, parameters: dict[str, Any], timeout_seconds: int
    ) -> Any:
        """Execute a tool within the sandbox with the given timeout.

        Args:
            tool_name: Name of the tool to execute.
            parameters: Input parameters for the tool.
            timeout_seconds: Maximum execution time before timeout.

        Returns:
            The tool's output result.

        Raises:
            asyncio.TimeoutError: If execution exceeds timeout.
            Exception: Any tool execution error.
        """
        ...


class DefaultSandboxExecutor(SandboxExecutor):
    """Default sandbox executor that runs tool handlers with timeout enforcement.

    Tool handlers are registered as async callables keyed by tool name.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, Any] = {}

    def register_handler(self, tool_name: str, handler: Any) -> None:
        """Register an async handler for a tool."""
        self._handlers[tool_name] = handler

    async def execute(
        self, tool_name: str, parameters: dict[str, Any], timeout_seconds: int
    ) -> Any:
        """Execute a registered tool handler with timeout enforcement."""
        handler = self._handlers.get(tool_name)
        if handler is None:
            raise RuntimeError(f"No handler registered for tool: {tool_name}")

        return await asyncio.wait_for(
            handler(parameters), timeout=timeout_seconds
        )


# =============================================================================
# MCP Gateway Service
# =============================================================================


class MCPGatewayService:
    """Controlled tool invocation gateway with catalog validation.

    Requirement 6.1: Maintain catalog of approved tools.
    Requirement 6.2: Validate tool name exists and parameters conform to schema.
    Requirement 6.3: Reject unapproved tools with logging.
    Requirement 6.4: Execute in sandbox with configurable timeout (default 30s).
    Requirement 6.5: Record all invocations with required fields.
    Requirement 6.6: On failure, record with error category, return structured error.
    """

    def __init__(self, tool_catalog: ToolCatalog, sandbox: SandboxExecutor):
        self.catalog = tool_catalog
        self.sandbox = sandbox
        self._invocation_log: list[ToolResult] = []

    @property
    def invocation_log(self) -> list[ToolResult]:
        """Access the recorded invocation trace entries."""
        return list(self._invocation_log)

    def validate_tool_request(self, tool_name: str, parameters: dict[str, Any]) -> bool:
        """Check tool exists in catalog and parameters conform to schema.

        Requirement 6.2: Validate that the requested tool name exists in the
        approved catalog and that the supplied input parameters conform to the
        tool's permitted parameter schema before execution.

        Args:
            tool_name: Name of the tool to validate.
            parameters: Input parameters to validate against the tool's schema.

        Returns:
            True if the tool exists and parameters are valid.

        Raises:
            ToolValidationError: If tool is not in catalog or parameters fail validation.
        """
        # Check tool exists in catalog
        tool_def = self.catalog.get_tool(tool_name)
        if tool_def is None:
            raise ToolValidationError(
                reason=f"Tool '{tool_name}' is not in the approved catalog",
                tool_name=tool_name,
                validation_errors=[f"Tool '{tool_name}' not found in approved catalog"],
            )

        # Validate parameters against JSON Schema
        validation_errors = self._validate_parameters(parameters, tool_def.input_schema)
        if validation_errors:
            raise ToolValidationError(
                reason=f"Parameters for tool '{tool_name}' do not conform to schema",
                tool_name=tool_name,
                validation_errors=validation_errors,
            )

        return True

    async def invoke_tool(
        self, tool_name: str, parameters: dict[str, Any], agent_identity: str
    ) -> ToolResult:
        """Validate tool is in catalog, parameters match schema, execute in sandbox.

        Requirement 6.2: Validate tool and parameters before execution.
        Requirement 6.4: Execute in sandbox with per-tool timeout (default 30s).
        Requirement 6.5: Record all invocations with required fields.
        Requirement 6.6: On failure, record with error category, no retry.

        Args:
            tool_name: Name of the tool to invoke.
            parameters: Input parameters for the tool.
            agent_identity: Identity of the requesting agent for audit trail.

        Returns:
            ToolResult with invocation details regardless of success/failure.
        """
        invoked_at = datetime.now(timezone.utc)
        start_time = time.monotonic()

        # Step 1: Validate tool request
        try:
            self.validate_tool_request(tool_name, parameters)
        except ToolValidationError as e:
            # Requirement 6.3: Log unauthorized tool request
            logger.warning(
                "Unapproved tool invocation rejected | "
                f"agent_identity={agent_identity} | "
                f"tool_name={tool_name} | "
                f"timestamp={invoked_at.isoformat()}"
            )
            duration_ms = int((time.monotonic() - start_time) * 1000)
            result = ToolResult(
                tool_name=tool_name,
                input_parameters=parameters,
                output_result=None,
                duration_ms=duration_ms,
                success=False,
                error_category="validation_error",
                invoked_at=invoked_at,
                agent_identity=agent_identity,
            )
            self._record_invocation(result)
            raise

        # Step 2: Get tool definition for timeout
        tool_def = self.catalog.get_tool(tool_name)
        timeout = tool_def.timeout_seconds

        # Step 3: Execute in sandbox with timeout
        try:
            output = await self.sandbox.execute(tool_name, parameters, timeout)
            duration_ms = int((time.monotonic() - start_time) * 1000)

            # Record successful invocation
            result = ToolResult(
                tool_name=tool_name,
                input_parameters=parameters,
                output_result=output,
                duration_ms=duration_ms,
                success=True,
                error_category=None,
                invoked_at=invoked_at,
                agent_identity=agent_identity,
            )
            self._record_invocation(result)

            logger.info(
                f"Tool invocation succeeded | "
                f"tool_name={tool_name} | "
                f"agent_identity={agent_identity} | "
                f"duration_ms={duration_ms}"
            )
            return result

        except asyncio.TimeoutError:
            # Requirement 6.6: Record timeout failure, no retry
            duration_ms = int((time.monotonic() - start_time) * 1000)
            result = ToolResult(
                tool_name=tool_name,
                input_parameters=parameters,
                output_result=None,
                duration_ms=duration_ms,
                success=False,
                error_category="timeout",
                invoked_at=invoked_at,
                agent_identity=agent_identity,
            )
            self._record_invocation(result)

            logger.error(
                f"Tool invocation timed out | "
                f"tool_name={tool_name} | "
                f"agent_identity={agent_identity} | "
                f"timeout_seconds={timeout} | "
                f"duration_ms={duration_ms}"
            )
            return result

        except Exception as e:
            # Requirement 6.6: Record crash/error failure, no retry
            duration_ms = int((time.monotonic() - start_time) * 1000)
            error_category = self._classify_error(e)
            result = ToolResult(
                tool_name=tool_name,
                input_parameters=parameters,
                output_result=None,
                duration_ms=duration_ms,
                success=False,
                error_category=error_category,
                invoked_at=invoked_at,
                agent_identity=agent_identity,
            )
            self._record_invocation(result)

            logger.error(
                f"Tool invocation failed | "
                f"tool_name={tool_name} | "
                f"agent_identity={agent_identity} | "
                f"error_category={error_category} | "
                f"error={str(e)} | "
                f"duration_ms={duration_ms}"
            )
            return result

    def _validate_parameters(
        self, parameters: dict[str, Any], schema: dict[str, Any]
    ) -> list[str]:
        """Validate parameters against JSON Schema.

        Returns a list of validation error messages. Empty list means valid.
        """
        errors: list[str] = []
        try:
            jsonschema.validate(instance=parameters, schema=schema)
        except jsonschema.ValidationError as e:
            errors.append(e.message)
        except jsonschema.SchemaError as e:
            errors.append(f"Invalid schema definition: {e.message}")
        return errors

    def _record_invocation(self, result: ToolResult) -> None:
        """Record an invocation result to the internal trace log.

        Requirement 6.5: Record tool_name, input_parameters, output_result,
        duration_ms, and success/failure status for every invocation.
        """
        self._invocation_log.append(result)

    def _classify_error(self, error: Exception) -> str:
        """Classify an execution error into a category string."""
        if isinstance(error, asyncio.TimeoutError):
            return "timeout"
        elif isinstance(error, ConnectionError):
            return "connection_error"
        elif isinstance(error, PermissionError):
            return "permission_error"
        elif isinstance(error, RuntimeError):
            return "runtime_error"
        elif isinstance(error, ValueError):
            return "value_error"
        elif isinstance(error, TypeError):
            return "type_error"
        else:
            return "execution_error"
