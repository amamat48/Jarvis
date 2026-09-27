"""Fail-closed seam for future OS-enforced untrusted-code execution."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionRequirements:
    filesystem_roots: tuple[str, ...]
    allow_network: bool = False
    allow_subprocess: bool = False
    timeout_seconds: int = 10
    memory_limit_bytes: int | None = None
    cpu_time_seconds: int | None = None
    isolated_secrets: bool = True
    cancellable: bool = True


class ExecutionAdapterUnavailable(RuntimeError):
    pass


class HostPythonAdapter:
    def execute(self, source_path: str, requirements: ExecutionRequirements) -> str:
        raise ExecutionAdapterUnavailable(
            "Python execution is disabled: OS-enforced filesystem/network isolation and resource controls are not configured."
        )
