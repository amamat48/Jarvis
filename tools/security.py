"""Authorization boundary for model-proposed tool operations."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import secrets
import threading
from typing import Any
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class TaskContext:
    task_id: str
    revision: int
    cancelled: bool = False


@dataclass(frozen=True)
class ToolSecurity:
    capability: str
    risk: str
    filesystem_scope: str = "none"
    network: bool = False
    subprocess: bool = False
    secret_sensitive: bool = False
    human_approval: bool = False


TOOL_SECURITY = {
    "calculate": ToolSecurity("calculate", "low"),
    "read_file": ToolSecurity("read_file", "low", "project-read", secret_sensitive=True),
    "list_files": ToolSecurity("list_files", "low", "project-read", secret_sensitive=True),
    "search_files": ToolSecurity("search_files", "low", "project-read", secret_sensitive=True),
    "remember_memory": ToolSecurity("remember_memory", "moderate", "memory-write", secret_sensitive=True),
    "recall_memory": ToolSecurity("recall_memory", "low", "memory-read", secret_sensitive=True),
    "run_python_file": ToolSecurity("run_python_file", "high", "project-read-write", network=True, subprocess=True, secret_sensitive=True, human_approval=True),
    "debug_python_file": ToolSecurity("debug_python_file", "high", "project-read-write", network=True, subprocess=True, secret_sensitive=True, human_approval=True),
    "install_package": ToolSecurity("install_package", "high", "environment-write", network=True, subprocess=True, secret_sensitive=True, human_approval=True),
    "delete_file": ToolSecurity("delete_file", "destructive", "project-write", secret_sensitive=True, human_approval=True),
}

SECRET_PARTS = {
    ".env", ".ssh", ".aws", ".azure", ".gnupg", ".config", ".kube",
    ".docker", ".npm", ".npmrc", ".pypirc", ".netrc", "credentials",
    "secrets", "browser", "chrome", "firefox", "edge", "user data",
}
SECRET_NAMES = re.compile(
    r"(^|[._-])(id_(rsa|ed25519|ecdsa)|.*(api[_-]?key|private[_-]?key|credential|secret|token).*)([._-]|$)",
    re.I,
)


class AuthorizationError(PermissionError):
    pass


class ApprovalRequired(AuthorizationError):
    def __init__(self, decision: "PolicyDecision"):
        super().__init__("exact human approval required")
        self.decision = decision


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    requires_approval: bool
    tool_name: str
    risk: str
    operation_id: str
    target: str
    reason: str


@dataclass(frozen=True)
class Operation:
    operation_id: str
    task_id: str
    revision: int
    tool_name: str
    arguments: dict[str, Any]

    @property
    def fingerprint(self) -> str:
        return operation_fingerprint(self.task_id, self.revision, self.tool_name, self.arguments)


@dataclass(frozen=True)
class ApprovalGrant:
    operation_id: str
    task_id: str
    revision: int
    fingerprint: str
    token: str


@dataclass(frozen=True)
class AuthorizedOperation:
    operation_id: str
    task_id: str
    revision: int
    tool_name: str
    arguments: dict[str, Any]
    fingerprint: str
    token: str


def operation_fingerprint(task_id: str, revision: int, tool_name: str, arguments: dict[str, Any]) -> str:
    normalized = normalize_arguments(arguments)
    canonical = json.dumps(
        [task_id, revision, tool_name, normalized],
        sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def normalize_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        raise AuthorizationError("arguments must be an object")
    try:
        encoded = json.dumps(arguments, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        result = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise AuthorizationError(f"arguments are not normalized JSON values: {error}") from error
    if not isinstance(result, dict):
        raise AuthorizationError("arguments must be an object")
    return result


class SecurityGate:
    """Policy, one-use approvals, revocation, and authorization verification."""

    def __init__(self, project_root: Path = PROJECT_ROOT):
        self.project_root = Path(project_root).resolve()
        self._lock = threading.RLock()
        self._latest_revision: dict[str, int] = {}
        self._cancelled: set[str] = set()
        self._operations: dict[str, tuple[str, str, int, str]] = {}
        self._approvals: dict[str, ApprovalGrant] = {}
        self._authorized: dict[str, tuple[str, str, int]] = {}

    def inspect(self, tool_name: str, arguments: dict[str, Any]) -> tuple[dict[str, Any], PolicyDecision]:
        normalized = normalize_arguments(arguments)
        operation_id = str(uuid4())
        metadata = TOOL_SECURITY.get(tool_name)
        from tools.registry import TOOLS
        if metadata is None or tool_name not in TOOLS:
            decision = PolicyDecision(False, False, tool_name, "unknown", operation_id, "", "unknown capability")
            return normalized, decision
        allowed, reason = self._scope_decision(metadata, normalized)
        target = self._safe_target(normalized)
        decision = PolicyDecision(
            allowed=allowed,
            requires_approval=allowed and (metadata.human_approval or metadata.network),
            tool_name=tool_name,
            risk=metadata.risk,
            operation_id=operation_id,
            target=target,
            reason=reason,
        )
        return normalized, decision

    def propose(self, context: TaskContext, tool_name: str, arguments: dict[str, Any]) -> tuple[Operation, PolicyDecision]:
        self._validate_context(context)
        normalized, decision = self.inspect(tool_name, arguments)
        operation = Operation(decision.operation_id, context.task_id, context.revision, tool_name, normalized)
        if not decision.allowed:
            raise AuthorizationError(decision.reason)
        with self._lock:
            self._operations[operation.operation_id] = (
                operation.fingerprint, operation.task_id, operation.revision, operation.tool_name,
            )
        return operation, decision

    def approve(self, context: TaskContext, operation: Operation) -> ApprovalGrant:
        self._validate_operation_context(context, operation)
        decision = self._decision_for(operation)
        if not decision.allowed:
            raise AuthorizationError(decision.reason)
        if not decision.requires_approval:
            raise AuthorizationError("operation does not require human approval")
        with self._lock:
            stored_operation = self._operations.get(operation.operation_id)
            if stored_operation is None or stored_operation[0] != operation.fingerprint:
                self._operations.pop(operation.operation_id, None)
                self._approvals.pop(operation.operation_id, None)
                raise AuthorizationError("operation is stale or its arguments changed")
            grant = ApprovalGrant(
                operation.operation_id, operation.task_id, operation.revision,
                operation.fingerprint, secrets.token_urlsafe(32),
            )
            self._approvals[operation.operation_id] = grant
            return grant

    def authorize(
        self, context: TaskContext, operation: Operation, grant: ApprovalGrant | None = None
    ) -> AuthorizedOperation:
        self._validate_operation_context(context, operation)
        decision = self._decision_for(operation)
        if not decision.allowed:
            raise AuthorizationError(decision.reason)
        with self._lock:
            fingerprint = operation.fingerprint
            stored_operation = self._operations.get(operation.operation_id)
            if stored_operation is None or stored_operation[0] != fingerprint:
                self._operations.pop(operation.operation_id, None)
                self._approvals.pop(operation.operation_id, None)
                raise AuthorizationError("unknown, stale, or modified operation")
            if decision.requires_approval:
                stored = self._approvals.get(operation.operation_id)
                if grant is None or stored != grant or grant.fingerprint != fingerprint:
                    raise ApprovalRequired(decision)
                self._approvals.pop(operation.operation_id, None)
            self._operations.pop(operation.operation_id, None)
            authorized = AuthorizedOperation(
                operation.operation_id, operation.task_id, operation.revision,
                operation.tool_name, normalize_arguments(operation.arguments),
                fingerprint, secrets.token_urlsafe(32),
            )
            self._authorized[authorized.token] = (fingerprint, authorized.task_id, authorized.revision)
            return authorized

    def consume(self, context: TaskContext, authorized: AuthorizedOperation) -> None:
        """Verify and consume the authorization immediately before dispatch."""
        try:
            self._validate_context(context)
        except AuthorizationError:
            with self._lock:
                self._authorized.pop(authorized.token, None)
            raise
        with self._lock:
            if context.task_id != authorized.task_id or context.revision != authorized.revision:
                self._authorized.pop(authorized.token, None)
                raise AuthorizationError("task identity or revision changed")
            expected = operation_fingerprint(
                authorized.task_id, authorized.revision,
                authorized.tool_name, authorized.arguments,
            )
            issued = self._authorized.get(authorized.token)
            if issued != (authorized.fingerprint, authorized.task_id, authorized.revision) or expected != authorized.fingerprint:
                self._authorized.pop(authorized.token, None)
                raise AuthorizationError("authorization is invalid, reused, or arguments changed")
            self._authorized.pop(authorized.token, None)
        metadata = TOOL_SECURITY.get(authorized.tool_name)
        if metadata is None:
            raise AuthorizationError("unknown capability")
        if metadata.network or metadata.subprocess:
            raise AuthorizationError("execution blocked: no OS-enforced sandbox is configured")
        allowed, reason = self._scope_decision(metadata, authorized.arguments)
        if not allowed:
            raise AuthorizationError(reason)

    def cancel_task(self, task_id: str) -> None:
        with self._lock:
            self._cancelled.add(task_id)
            self._latest_revision.pop(task_id, None)
            for operation_id, (_, issued_task_id, _, _) in tuple(self._operations.items()):
                if issued_task_id == task_id:
                    self._operations.pop(operation_id, None)
                    self._approvals.pop(operation_id, None)
            for token, (_, issued_task_id, _) in tuple(self._authorized.items()):
                if issued_task_id == task_id:
                    self._authorized.pop(token, None)

    def reject(self, operation: Operation) -> None:
        with self._lock:
            self._operations.pop(operation.operation_id, None)
            self._approvals.pop(operation.operation_id, None)

    def _validate_operation_context(self, context: TaskContext, operation: Operation) -> None:
        self._validate_context(context)
        if (context.task_id, context.revision) != (operation.task_id, operation.revision):
            self.invalidate_task_revision(context.task_id, context.revision)
            raise AuthorizationError("task identity or revision changed")

    def _validate_context(self, context: TaskContext) -> None:
        if (
            not isinstance(context, TaskContext)
            or not isinstance(context.task_id, str)
            or not context.task_id
            or not isinstance(context.revision, int)
            or isinstance(context.revision, bool)
            or context.revision < 0
        ):
            raise AuthorizationError("valid task identity and revision are required")
        if context.cancelled:
            self.cancel_task(context.task_id)
            raise AuthorizationError("task is cancelled")
        with self._lock:
            if context.task_id in self._cancelled:
                raise AuthorizationError("task is cancelled")
            latest = self._latest_revision.get(context.task_id)
            if latest is not None and context.revision < latest:
                raise AuthorizationError("stale task revision")
            if latest is None or context.revision > latest:
                self._latest_revision[context.task_id] = context.revision
                self._drop_older_operations(context.task_id, context.revision)

    def invalidate_task_revision(self, task_id: str, revision: int) -> None:
        with self._lock:
            previous = self._latest_revision.get(task_id, -1)
            if revision > previous:
                self._latest_revision[task_id] = revision
            self._drop_older_operations(task_id, revision)

    def _drop_older_operations(self, task_id: str, revision: int) -> None:
        for operation_id, (_, issued_task_id, issued_revision, _) in tuple(self._operations.items()):
            if issued_task_id == task_id and issued_revision != revision:
                self._approvals.pop(operation_id, None)
                self._operations.pop(operation_id, None)
        for token, (_, issued_task_id, issued_revision) in tuple(self._authorized.items()):
            if issued_task_id == task_id and issued_revision != revision:
                self._authorized.pop(token, None)

    def _decision_for(self, operation: Operation) -> PolicyDecision:
        normalized, decision = self.inspect(operation.tool_name, operation.arguments)
        if normalized != operation.arguments:
            return PolicyDecision(False, False, operation.tool_name, "unknown", operation.operation_id, "", "operation arguments are not normalized")
        return PolicyDecision(
            decision.allowed, decision.requires_approval, operation.tool_name,
            decision.risk, operation.operation_id, decision.target, decision.reason,
        )

    def _scope_decision(self, metadata: ToolSecurity, arguments: dict[str, Any]) -> tuple[bool, str]:
        if metadata.filesystem_scope.startswith("project-"):
            value = arguments.get("path")
            if value is not None:
                try:
                    self.resolve_project_path(str(value))
                except AuthorizationError as error:
                    return False, str(error)
        return True, "exact human approval required" if metadata.human_approval or metadata.network else "allowed by policy"

    def resolve_project_path(self, value: str) -> Path:
        candidate = (self.project_root / value).resolve()
        if not candidate.is_relative_to(self.project_root):
            raise AuthorizationError("filesystem path escapes project scope")
        relative = candidate.relative_to(self.project_root)
        if any(self._is_secret_part(part) for part in relative.parts):
            raise AuthorizationError("secret or credential path denied")
        return candidate

    @staticmethod
    def _is_secret_part(part: str) -> bool:
        folded = part.casefold()
        return folded in SECRET_PARTS or folded == ".env" or folded.startswith(".env.") or bool(SECRET_NAMES.search(part))

    @staticmethod
    def _safe_target(arguments: dict[str, Any]) -> str:
        path = arguments.get("path")
        if isinstance(path, str):
            return re.sub(r"[\x00-\x1f\x7f]", "?", path)[:180]
        name = arguments.get("name")
        if isinstance(name, str):
            return re.sub(r"[\x00-\x1f\x7f]", "?", name)[:100]
        return ""
