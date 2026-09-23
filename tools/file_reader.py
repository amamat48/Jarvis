from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def read_file(path: str) -> str:
    """
    Read a text file inside the JARVIS project.

    Args:
        path: Path to the file, relative to the JARVIS project.

    Returns:
        The contents of the file.
    """

    try:
        file_path = (PROJECT_ROOT / path).resolve()

        # Prevent access outside the JARVIS project
        if not file_path.is_relative_to(PROJECT_ROOT):
            return "Access denied: file is outside the JARVIS project."

        # Prevent reading files that may contain secrets or repository data
        if ".env" in file_path.parts or ".git" in file_path.parts or ".venv" in file_path.parts:
            return "Access denied: protected file or directory."

        return file_path.read_text()

    except FileNotFoundError:
        return f"File not found: {path}"

    except Exception as error:
        return f"Error reading file: {error}"