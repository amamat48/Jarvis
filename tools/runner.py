"""Authorized tool executor. Model callers must not dispatch by tool name."""
import copy
import inspect

from tools.registry import TOOLS
from tools.security import AuthorizedOperation, AuthorizationError, SecurityGate, TaskContext


DEFAULT_SECURITY_GATE = SecurityGate()


def execute(
    task_context: TaskContext,
    authorized_operation: AuthorizedOperation,
    *,
    gate: SecurityGate = DEFAULT_SECURITY_GATE,
) -> str:
    """Execute a one-use operation issued by SecurityGate, failing closed."""
    if not isinstance(authorized_operation, AuthorizedOperation):
        return "Execution denied: Security Gate authorization is required."
    tool = TOOLS.get(authorized_operation.tool_name)
    if tool is None:
        return f"Execution denied: unknown capability '{authorized_operation.tool_name}'."
    arguments = copy.deepcopy(authorized_operation.arguments)
    try:
        inspect.signature(tool).bind(**arguments)
        # This consumes and revalidates the authorization immediately before dispatch.
        gate.consume(task_context, authorized_operation)
    except (AuthorizationError, TypeError, ValueError) as error:
        return f"Execution denied for '{authorized_operation.tool_name}': {error}"
    try:
        return tool(**arguments)
    except Exception as error:
        return f"Tool '{authorized_operation.tool_name}' failed: {error}"
