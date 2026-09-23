from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def list_files() -> str:
    """
    List files in the JARVIS project.
    """

    files = []

    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file():
            continue

        relative_path = path.relative_to(PROJECT_ROOT)

        if ".git" in relative_path.parts:
            continue

        if ".venv" in relative_path.parts:
            continue

        if ".env" in relative_path.parts:
            continue

        files.append(str(relative_path))

    return "\n".join(sorted(files))