from pathlib import Path
from tools.security import AuthorizationError, SecurityGate


PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SECURITY = SecurityGate(PROJECT_ROOT)


def list_files() -> str:
    """
    List files in the JARVIS project.
    """

    files = []

    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file():
            continue

        relative_path = path.relative_to(PROJECT_ROOT)

        try:
            _SECURITY.resolve_project_path(str(relative_path))
        except AuthorizationError:
            continue

        if any(_SECURITY._is_secret_part(part) for part in relative_path.parts):
            continue
        if ".git" in relative_path.parts or ".venv" in relative_path.parts:
            continue

        files.append(str(relative_path))

    return "\n".join(sorted(files))
