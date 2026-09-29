import re

from tools.calculator import calculate
from tools.file_reader import read_file
from tools.file_manager import list_files
from tools.file_writer import write_file
from tools.file_editor import edit_file
from tools.code_runner import run_python_file
from tools.code_search import search_files
from tools.memory_tools import remember_memory, recall_memory
from tools.debugger import debug_python_file
from tools.web_search import web_search


TOOLS = {
    "calculate": calculate,
    "read_file": read_file,
    "list_files": list_files,
    "write_file": write_file,
    "edit_file": edit_file,
    "run_python_file": run_python_file,
    "search_files": search_files,
    "remember_memory": remember_memory,
    "recall_memory": recall_memory,
    "debug_python_file": debug_python_file,
    "web_search": web_search,
}


def select_tools(user_input: str, context=None) -> dict:
    """
    Select tools that are relevant to the user's request.

    This prevents the language model from seeing every tool
    on every turn.
    """

    text = user_input.lower()
    selected = {}

    web_triggers = ("current", "latest", "today", "recent", "news", "on the web", "web search", "search the web", "search online", "online", "on the internet")
    if any(trigger in text for trigger in web_triggers):
        selected["web_search"] = TOOLS["web_search"]

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
        "what's inside",
        "what is inside",
        "show me what's inside",
        "show me what is inside",
    )

    directory_reference = re.search(r"\b(?:folder|directory|folders|directories)\b", text)
    filename_discovery = re.search(r"\b(?:find|locate|identify)\b.{0,60}\bfiles?\b", text)
    file_listing = re.search(
        r"\b(?:list|show)\b.{0,80}\bfiles?\b"
        r"|\b(?:what|which)\s+(?:are\s+)?(?:the\s+)?(?:python\s+)?files?\b",
        text,
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

    if (
        any(phrase in text for phrase in project_listing)
        or bool(directory_reference and ("file" in text or "inside" in text))
        or bool(file_listing)
        or bool(filename_discovery)
    ):
        selected["list_files"] = TOOLS["list_files"]

    if filename_discovery:
        # Discovery requests often require a dependent read after the model
        # learns the path from list_files or search_files.
        selected["read_file"] = TOOLS["read_file"]

    prior_listing = any(
        isinstance(message, dict)
        and message.get("role") == "tool"
        and message.get("tool_name") == "list_files"
        and isinstance(message.get("content"), str)
        for message in (context or ())
    )
    if prior_listing and re.search(r"\b(?:read|open|inspect|explain|summarize)\b", text):
        selected["read_file"] = TOOLS["read_file"]

    if (
        any(action in text for action in search_request)
        and any(reference in text for reference in code_reference)
    ):
        selected["search_files"] = TOOLS["search_files"]

    # -------------------------
    # File creation/editing
    # -------------------------

    write_triggers = (
        "create",
        "write",
        "save",
        "make a new",
        "new file",
        "add a file",
    )

    edit_triggers = (
        "edit",
        "modify",
        "change",
        "update",
        "fix",
        "replace",
        "patch",
    )

    # Determine if this looks like a file creation request
    file_creation_request = (
        any(trigger in text for trigger in write_triggers)
        and re.search(r"\b(?:file|script|module|class|function)\b", text)
    )

    # Determine if this looks like a file editing request
    file_edit_request = (
        any(trigger in text for trigger in edit_triggers)
        and (file_reference or re.search(r"\b(?:file|script|module|class|function|code)\b", text))
    )

    if file_creation_request:
        selected["write_file"] = TOOLS["write_file"]
        # Creation often requires reading existing files for context
        selected["read_file"] = TOOLS["read_file"]
        selected["list_files"] = TOOLS["list_files"]

    if file_edit_request:
        selected["edit_file"] = TOOLS["edit_file"]
        # Editing requires reading the file first
        selected["read_file"] = TOOLS["read_file"]

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

    explicit_python_path = re.search(r"(?:^|[\s/\\])[^\s/\\]+\.py\b", text)
    if ("debug" in text or "debugging" in text or "diagnose" in text):
        if explicit_python_path:
            selected["debug_python_file"] = TOOLS["debug_python_file"]

    return selected
