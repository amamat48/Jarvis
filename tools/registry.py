import re

from tools.calculator import calculate
from tools.file_reader import read_file
from tools.file_manager import list_files
from tools.code_runner import run_python_file
from tools.code_search import search_files
from tools.memory_tools import remember_memory, recall_memory
from tools.debugger import debug_python_file


TOOLS = {
    "calculate": calculate,
    "read_file": read_file,
    "list_files": list_files,
    "run_python_file": run_python_file,
    "search_files": search_files,
    "remember_memory": remember_memory,
    "recall_memory": recall_memory,
    "debug_python_file": debug_python_file,
}


def select_tools(user_input: str) -> dict:
    """
    Select tools that are relevant to the user's request.

    This prevents the language model from seeing every tool
    on every turn.
    """

    text = user_input.lower()
    selected = {}

    # -------------------------
    # Memory tools
    # -------------------------

    recall_triggers = (
        "what do you remember",
        "what do you know about me",
        "recall my",
        "show my memories",
        "what have you remembered",
    )

    if any(trigger in text for trigger in recall_triggers):
        selected["recall_memory"] = TOOLS["recall_memory"]

    # Only expose remember_memory when the request appears
    # to contain a complete piece of information.
    remember_triggers = (
        "remember that",
        "remember this",
        "remember my",
        "save this",
        "store this",
        "keep this in mind",
    )

    complete_memory_request = re.search(
        r"\b("
        r"is|am|are|use|uses|like|likes|prefer|prefers|"
        r"called|named|have|has"
        r")\b\s+\S+",
        text,
    )

    if (
        any(trigger in text for trigger in remember_triggers)
        and complete_memory_request
    ):
        selected["remember_memory"] = TOOLS["remember_memory"]

    # -------------------------
    # Calculator
    # -------------------------

    calculation_triggers = (
        "calculate",
        "compute",
        "how much is",
        "what is",
        "multiply",
        "divide",
        "add",
        "subtract",
    )

    has_math_symbol = any(
        symbol in text
        for symbol in "+-*/="
    )

    has_number = any(
        character.isdigit()
        for character in text
    )

    if (
        any(trigger in text for trigger in calculation_triggers)
        and (has_number or has_math_symbol)
    ):
        selected["calculate"] = TOOLS["calculate"]

    # -------------------------
    # File tools
    # -------------------------

    file_reference = any(
        extension in text
        for extension in (
            ".py",
            ".cpp",
            ".c",
            ".h",
            ".hpp",
            ".js",
            ".ts",
            ".java",
        )
    )

    file_actions = (
        "read",
        "open",
        "explain",
        "inspect",
        "show",
        "look at",
    )

    project_listing = (
        "list files",
        "what files",
        "which files",
        "files in the project",
        "files are in",
        "show me the files",
    )

    search_request = (
        "find",
        "search",
        "locate",
        "where is",
        "where are",
        "look for",
    )

    code_reference = (
        "function",
        "class",
        "variable",
        "definition",
        "defined",
        "implementation",
        "source code",
        "code",
        "error message",
        "calculator tool",
        "tool definition",
    )   

    if file_reference and any(
        action in text
        for action in file_actions
    ):
        selected["read_file"] = TOOLS["read_file"]

    if any(
        phrase in text
        for phrase in project_listing
    ):
        selected["list_files"] = TOOLS["list_files"]

    if (
        any(action in text for action in search_request)
        and any(reference in text for reference in code_reference)
    ):
        selected["search_files"] = TOOLS["search_files"]

    # -------------------------
    # Python execution
    # -------------------------

    python_file = ".py" in text

    if (
        python_file
        and ("run" in text or "execute" in text)
    ):
        selected["run_python_file"] = TOOLS["run_python_file"]

    # -------------------------
    # Python debugging
    # -------------------------

    if (
        "debug" in text
        or "debugging" in text
        or "diagnose" in text
    ):
        if python_file or "python" in text:
            selected["debug_python_file"] = TOOLS["debug_python_file"]

    return selected