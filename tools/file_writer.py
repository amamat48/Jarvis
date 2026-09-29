from pathlib import Path
from tools.security import AuthorizationError, SecurityGate


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def write_file(path: str, content: str) -> str:
    """
    Create a new file or overwrite an existing file in the JARVIS project.

    Args:
        path: Project-relative path of the file to write.
        content: Text content to write to the file.

    Returns:
        Success or error message.
    """

    try:
        file_path = SecurityGate(PROJECT_ROOT).resolve_project_path(path)

        # Prevent writing outside the JARVIS project
        if not file_path.is_relative_to(PROJECT_ROOT):
            return "Access denied: file is outside the JARVIS project."

        # Prevent writing to protected directories
        if ".git" in file_path.parts or ".venv" in file_path.parts:
            return "Access denied: protected file or directory."

        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the file
        file_path.write_text(content, encoding="utf-8")

        return f"File written: {path} ({len(content)} bytes)"

    except AuthorizationError as error:
        return f"Access denied: {error}"
    except Exception as error:
        return f"Error writing file: {error}"