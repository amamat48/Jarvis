from pathlib import Path

from tools.execution_adapter import ExecutionRequirements, ExecutionError, ExecutionTimeoutError
from tools.security import AuthorizationError, SecurityGate


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_python_file(path: str) -> str:
    """Execute a Python file in a sandboxed subprocess."""
    try:
        file_path = SecurityGate(PROJECT_ROOT).resolve_project_path(path)
        if file_path.suffix.lower() != ".py":
            return "Execution denied: only Python files can be run."
        if not file_path.is_file():
            return f"File not found: {path}"
        try:
            from tools.execution_adapter import execute as execute_adapter
            return execute_adapter(
                str(file_path), ExecutionRequirements(filesystem_roots=(str(file_path.parent),))
            )
        except ExecutionTimeoutError as error:
            return f"Execution timed out: {error}"
        except ExecutionError as error:
            return f"Execution failed: {error}"
    except AuthorizationError as error:
        return f"Execution denied: {error}"
    except Exception as error:
        return f"Execution unavailable: {error}"
