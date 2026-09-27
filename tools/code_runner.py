from pathlib import Path

from tools.execution_adapter import ExecutionRequirements, HostPythonAdapter
from tools.security import AuthorizationError, SecurityGate


PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ADAPTER = HostPythonAdapter()


def run_python_file(path: str) -> str:
    """Fail closed until an OS-enforced execution adapter is configured."""
    try:
        file_path = SecurityGate(PROJECT_ROOT).resolve_project_path(path)
        if file_path.suffix.lower() != ".py":
            return "Execution denied: only Python files can be run."
        if not file_path.is_file():
            return f"File not found: {path}"
        return _ADAPTER.execute(
            str(file_path), ExecutionRequirements(filesystem_roots=(str(file_path.parent),))
        )
    except AuthorizationError as error:
        return f"Execution denied: {error}"
    except Exception as error:
        return f"Execution unavailable: {error}"
