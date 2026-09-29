"""Windows-compatible sandboxed Python execution adapter."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ExecutionRequirements:
    filesystem_roots: tuple[str, ...]
    allow_network: bool = False
    allow_subprocess: bool = False
    timeout_seconds: int = 30
    memory_limit_mb: int | None = None
    cpu_time_seconds: int | None = None
    isolated_secrets: bool = True
    cancellable: bool = True


class ExecutionError(RuntimeError):
    pass


class ExecutionTimeoutError(ExecutionError):
    pass


class ResourceLimitExceededError(ExecutionError):
    pass


class SandboxViolationError(ExecutionError):
    pass


class SubprocessExecutionAdapter:
    """Windows-compatible subprocess-based execution with sandboxing."""

    def __init__(self):
        self._active_processes: dict[int, subprocess.Popen] = {}
        self._lock = threading.Lock()

    def execute(self, source_path: str, requirements: ExecutionRequirements) -> str:
        """Execute a Python file in a sandboxed subprocess."""
        source_path = Path(source_path).resolve()

        # Validate source file exists
        if not source_path.is_file():
            raise ExecutionError(f"Source file not found: {source_path}")

        # Validate file is within allowed filesystem roots
        self._validate_filesystem_access(source_path, requirements.filesystem_roots)

        # Prepare environment
        env = self._prepare_environment(requirements)

        # Prepare command
        cmd = [sys.executable, str(source_path)]

        # Execute with timeout and resource monitoring
        try:
            result = self._run_subprocess(cmd, env, requirements, source_path.parent)
            return result
        except subprocess.TimeoutExpired:
            raise ExecutionTimeoutError(
                f"Execution timed out after {requirements.timeout_seconds} seconds"
            )
        except ExecutionError:
            raise
        except Exception as e:
            raise ExecutionError(f"Execution failed: {e}")

    def _validate_filesystem_access(self, source_path: Path, filesystem_roots: tuple[str, ...]) -> None:
        """Validate that the source path is within allowed filesystem roots."""
        if not filesystem_roots:
            raise ExecutionError("No filesystem roots configured")

        source_resolved = source_path.resolve()
        allowed = False

        for root in filesystem_roots:
            root_path = Path(root).resolve()
            try:
                source_resolved.relative_to(root_path)
                allowed = True
                break
            except ValueError:
                continue

        if not allowed:
            raise SandboxViolationError(
                f"Source path {source_path} is not within allowed filesystem roots: {filesystem_roots}"
            )

    def _prepare_environment(self, requirements: ExecutionRequirements) -> dict[str, str]:
        """Prepare a restricted environment for the subprocess."""
        # Start with a minimal environment
        env = {
            "PATH": os.environ.get("PATH", ""),
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
            "TEMP": os.environ.get("TEMP", tempfile.gettempdir()),
            "TMP": os.environ.get("TMP", tempfile.gettempdir()),
            "PYTHONPATH": "",
            "PYTHONIOENCODING": "utf-8",
        }

        # Remove sensitive environment variables
        sensitive_vars = {
            "API_KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL",
            "AWS_", "AZURE_", "GCP_", "DOCKER_", "KUBE_",
            "SSH_", "GPG_", "GITHUB_", "GITLAB_",
        }

        for key in list(os.environ.keys()):
            if not any(key.startswith(prefix) for prefix in sensitive_vars):
                # Only pass through non-sensitive vars
                if key in {"HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA"}:
                    continue  # Skip user profile paths
                env[key] = os.environ[key]

        return env

    def _run_subprocess(
        self,
        cmd: list[str],
        env: dict[str, str],
        requirements: ExecutionRequirements,
        cwd: Path,
    ) -> str:
        """Run the subprocess with monitoring."""
        # Use a thread to track the process for cancellation
        process_holder: dict[str, Optional[subprocess.Popen]] = {"proc": None}
        exception_holder: dict[str, Optional[Exception]] = {"exc": None}

        def target():
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(cwd),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                process_holder["proc"] = proc

                # Wait with timeout
                try:
                    stdout, stderr = proc.communicate(timeout=requirements.timeout_seconds)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    stdout, stderr = proc.communicate()
                    raise subprocess.TimeoutExpired(cmd, requirements.timeout_seconds)

                # Check return code
                if proc.returncode != 0:
                    raise ExecutionError(
                        f"Process exited with code {proc.returncode}\n"
                        f"STDOUT:\n{stdout}\n"
                        f"STDERR:\n{stderr}"
                    )

                # Format output
                output_parts = []
                if stdout:
                    output_parts.append(f"STDOUT:\n{stdout}")
                if stderr:
                    output_parts.append(f"STDERR:\n{stderr}")
                output_parts.append(f"Exit code: {proc.returncode}")

                exception_holder["result"] = "\n".join(output_parts)

            except Exception as e:
                exception_holder["exc"] = e

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout=requirements.timeout_seconds + 5)

        if thread.is_alive():
            # Thread didn't finish, try to kill process
            if process_holder["proc"]:
                try:
                    process_holder["proc"].kill()
                except Exception:
                    pass
            thread.join(timeout=2)
            raise ExecutionTimeoutError(
                f"Execution timed out after {requirements.timeout_seconds} seconds"
            )

        if exception_holder["exc"]:
            raise exception_holder["exc"]

        return exception_holder.get("result", "Execution completed with no output")

    def cancel_all(self) -> None:
        """Cancel all active processes."""
        with self._lock:
            for proc in self._active_processes.values():
                try:
                    proc.kill()
                except Exception:
                    pass
            self._active_processes.clear()


# Global adapter instance
_ADAPTER = SubprocessExecutionAdapter()


def execute(source_path: str, requirements: ExecutionRequirements) -> str:
    """Module-level execute function for backward compatibility."""
    return _ADAPTER.execute(source_path, requirements)