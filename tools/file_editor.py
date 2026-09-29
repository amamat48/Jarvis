from pathlib import Path
from tools.security import AuthorizationError, SecurityGate


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def edit_file(path: str, old_text: str, new_text: str) -> str:
    """
    Edit an existing file by replacing old_text with new_text.

    Args:
        path: Project-relative path of the file to edit.
        old_text: The exact text to find and replace.
        new_text: The replacement text.

    Returns:
        Success or error message.
    """

    if not old_text:
        return "Error: old_text cannot be empty."

    try:
        file_path = SecurityGate(PROJECT_ROOT).resolve_project_path(path)

        # Prevent editing outside the JARVIS project
        if not file_path.is_relative_to(PROJECT_ROOT):
            return "Access denied: file is outside the JARVIS project."

        # Prevent editing protected directories
        if ".git" in file_path.parts or ".venv" in file_path.parts:
            return "Access denied: protected file or directory."

        if not file_path.is_file():
            return f"File not found: {path}"

        # Read current content
        current_content = file_path.read_text(encoding="utf-8")

        # Check if old_text exists
        if old_text not in current_content:
            return f"Error: old_text not found in file. The text to replace must match exactly (including whitespace)."

        # Count occurrences
        count = current_content.count(old_text)
        if count > 1:
            return f"Error: old_text matches {count} locations. Provide more context to make the match unique."

        # Replace and write
        new_content = current_content.replace(old_text, new_text)
        file_path.write_text(new_content, encoding="utf-8")

        return f"File edited: {path} (1 replacement)"

    except AuthorizationError as error:
        return f"Access denied: {error}"
    except Exception as error:
        return f"Error editing file: {error}"