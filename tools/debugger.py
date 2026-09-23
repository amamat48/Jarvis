from tools.file_reader import read_file
from tools.code_runner import run_python_file


def debug_python_file(path: str) -> str:
    """
    Gather the source code and execution results for a Python file.

    The LLM uses this information to diagnose bugs.

    Args:
        path: Path to the Python file, relative to the JARVIS project.

    Returns:
        Source code and execution results.
    """

    source = read_file(path)

    if source.startswith("File not found:"):
        return source

    if source.startswith("Access denied:"):
        return source

    execution = run_python_file(path)

    return (
        "=== SOURCE CODE ===\n"
        f"{source}\n\n"
        "=== EXECUTION RESULTS ===\n"
        f"{execution}"
    )