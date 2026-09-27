"""Authorized tool executor. Model callers must not dispatch by tool name."""
import copy
from dataclasses import dataclass
import inspect
import secrets
import threading
from typing import Any, Callable

from tools.registry import TOOLS
from tools.security import AuthorizedOperation, AuthorizationError, SecurityGate, TaskContext


DEFAULT_SECURITY_GATE = SecurityGate()


@dataclass(frozen=True)
class _ExecutionClaim:
    """Opaque handle for a one-use dispatch payload held by this module."""

    claim_id: str


_CLAIM_LOCK = threading.Lock()
_PENDING_CLAIMS: dict[str, tuple[str, Callable[..., Any], dict[str, Any]]] = {}


def claim(
    task_context: TaskContext,
    authorized_operation: AuthorizedOperation,
    *,
    gate: SecurityGate = DEFAULT_SECURITY_GATE,
) -> _ExecutionClaim | str:
    """Validate and consume an authorization, returning a detached dispatch snapshot."""
    if not isinstance(authorized_operation, AuthorizedOperation):
        return "Execution denied: Security Gate authorization is required."

    # AuthorizedOperation is frozen only shallowly: detach its argument graph before
    # validation/consumption so later mutation of the original cannot alter dispatch.
    stable_operation = copy.deepcopy(authorized_operation)
    tool = TOOLS.get(stable_operation.tool_name)
    if tool is None:
        return f"Execution denied: unknown capability '{stable_operation.tool_name}'."
    arguments = copy.deepcopy(stable_operation.arguments)
    try:
        inspect.signature(tool).bind(**arguments)
        gate.consume(task_context, stable_operation)
    except (AuthorizationError, TypeError, ValueError) as error:
        return f"Execution denied for '{stable_operation.tool_name}': {error}"

    # Do not put dispatch arguments or the callable on the returned object. The
    # handle is not authority by itself: only a claim registered here after Gate
    # consumption can be executed, and execution atomically removes that record.
    claim_id = secrets.token_urlsafe(32)
    with _CLAIM_LOCK:
        _PENDING_CLAIMS[claim_id] = (
            stable_operation.tool_name,
            tool,
            copy.deepcopy(arguments),
        )
    return _ExecutionClaim(claim_id)


def execute_claimed(claimed: _ExecutionClaim) -> str:
    """Dispatch a Gate-issued claim once, using its detached private payload."""
    if not isinstance(claimed, _ExecutionClaim):
        return "Execution denied: a valid execution claim is required."
    with _CLAIM_LOCK:
        payload = _PENDING_CLAIMS.pop(claimed.claim_id, None)
    if payload is None:
        return "Execution denied: execution claim is invalid or already used."

    tool_name, tool, arguments = payload
    try:
        return tool(**arguments)
    except Exception as error:
        return f"Tool '{tool_name}' failed: {error}"


def execute(
    task_context: TaskContext,
    authorized_operation: AuthorizedOperation,
    *,
    gate: SecurityGate = DEFAULT_SECURITY_GATE,
) -> str:
    """Execute a one-use operation issued by SecurityGate, failing closed."""
    claimed = claim(task_context, authorized_operation, gate=gate)
    if isinstance(claimed, str):
        return claimed
    return execute_claimed(claimed)
