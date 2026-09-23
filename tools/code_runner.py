from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_python_file(path: str) -> str:
    """
    Run a Python file inside the JARVIS project.

    Args:
        path: Path to a Python file, relative to the JARVIS project.

    Returns:
        The program's output or error information.
    """

    try:
        file_path = (PROJECT_ROOT / path).resolve()

        # Keep execution inside the JARVIS project.
        if not file_path.is_relative_to(PROJECT_ROOT):
            return "Execution denied: file is outside the JARVIS project."

        # Only allow Python source files for now.
        if file_path.suffix != ".py":
            return "Execution denied: only Python files can be run."

        # Don't allow execution of protected project areas.
        if ".env" in file_path.parts or ".git" in file_path.parts:
            return "Execution denied: protected file or directory."

        if not file_path.is_file():
            return f"File not found: {path}"

        result = subprocess.run(
            [sys.executable, str(file_path)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10
        )

        output = []

        if result.stdout:
            output.append(f"STDOUT:\n{result.stdout}")

        if result.stderr:
            output.append(f"STDERR:\n{result.stderr}")

        output.append(f"Exit code: {result.returncode}")

        return "\n".join(output)

    except subprocess.TimeoutExpired:
        return "Execution timed out after 10 seconds."

    except Exception as error:
        return f"Error running file: {error}"