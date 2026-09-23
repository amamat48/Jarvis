from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_EXTENSIONS = {
    ".py",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".js",
    ".ts",
    ".java",
}


def search_files(query: str) -> str:
    """
    Search source-code files in the JARVIS project.

    Args:
        query: Text to search for.

    Returns:
        Matching file paths, line numbers, and lines of code.
    """

    matches = []

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

        if path.suffix.lower() not in SOURCE_EXTENSIONS:
            continue

        try:
            lines = path.read_text(
                encoding="utf-8",
                errors="replace"
            ).splitlines()

            for line_number, line in enumerate(lines, start=1):
                if query.lower() in line.lower():
                    matches.append(
                        f"{relative_path}:{line_number}: {line.strip()}"
                    )

        except Exception:
            continue

        if len(matches) >= 50:
            break

    if not matches:
        return f"No matches found for: {query}"

    return "\n".join(matches)