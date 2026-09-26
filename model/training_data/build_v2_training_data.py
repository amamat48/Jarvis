"""
Build the JARVIS v2 training dataset.

============================================================
PURPOSE
============================================================

JARVIS v1 showed an important training-data problem:

The model was often trained with only the tool it was expected to use
available in the example.

For example:

    User:
        Calculate 73 * 19

    Training tools:
        calculate

That makes tool selection artificially easy.

During evaluation, however, Qwen sees ALL JARVIS tools:

    calculate
    read_file
    list_files
    search_files
    run_python_file
    debug_python_file
    remember_memory
    recall_memory

Therefore, v1 was not being trained on exactly the same decision problem
that we were evaluating.

JARVIS v2 fixes that.

============================================================
V2 DATASET DESIGN
============================================================

The v2 dataset consists of:

    Original 113 training examples
        +
    24 new targeted behavior examples

The original 16 validation examples remain unchanged.

The original 16 test examples remain untouched.

The new 32-case evaluation set also remains untouched.

Therefore:

    training data
        -> used to learn

    validation data
        -> used during training

    32-case evaluation
        -> never used during training

============================================================
NEW TRAINING BEHAVIORS
============================================================

The new examples target the baseline weaknesses:

    - explicit memory writes
    - memory recall
    - incomplete memory requests
    - calculator selection
    - incomplete calculations
    - read_file vs list_files
    - search_files vs list_files
    - Python execution
    - debugging
    - no unnecessary follow-up tools
    - multi-step search -> read
    - failed search -> recovery
    - grounding after tool results
    - no-tool behavior while tools are available

============================================================
AVAILABLE TOOLS
============================================================

Each example receives an "available_tools" field.

Examples requiring no tool receive ALL tools.

Tool-use examples receive a realistic set of competing tools.

Examples involving file investigation receive:

    read_file
    list_files
    search_files
    run_python_file
    debug_python_file

Memory examples receive:

    remember_memory
    recall_memory

This makes the training problem much closer to what JARVIS experiences
during real operation.

============================================================
OUTPUT
============================================================

The script creates:

    model/training_data/v2/
        train.jsonl
        valid.jsonl

The original dataset files are not modified.
"""

import json
from copy import deepcopy
from pathlib import Path


# ============================================================================
# PROJECT PATHS
# ============================================================================
#
# This script lives at:
#
#     jarvis/model/training_data/build_v2_training_data.py
#
# Therefore:
#
#     parents[0] -> model/training_data
#     parents[1] -> model
#     parents[2] -> jarvis
#
# We use the project root to locate both the original training splits and
# the new v2 output directory.
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ORIGINAL_TRAIN_FILE = (
    PROJECT_ROOT
    / "model"
    / "training_data"
    / "splits"
    / "train.jsonl"
)

ORIGINAL_VALID_FILE = (
    PROJECT_ROOT
    / "model"
    / "training_data"
    / "splits"
    / "valid.jsonl"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "model"
    / "training_data"
    / "v2"
)

OUTPUT_TRAIN_FILE = OUTPUT_DIR / "train.jsonl"
OUTPUT_VALID_FILE = OUTPUT_DIR / "valid.jsonl"


# ============================================================================
# CURRENT JARVIS TOOL SET
# ============================================================================
#
# Keep this list synchronized with tools/registry.py.
#
# The order is not semantically important, but keeping it stable makes the
# generated datasets easier to inspect and compare.
# ============================================================================

ALL_TOOLS = [
    "calculate",
    "read_file",
    "list_files",
    "run_python_file",
    "search_files",
    "remember_memory",
    "recall_memory",
    "debug_python_file",
]


# ============================================================================
# TOOL COMPETITION GROUPS
# ============================================================================
#
# We do NOT always need to expose all eight tools.
#
# The goal is to expose the tools that could reasonably be confused with
# one another for the given task.
#
# This reduces unnecessary token overhead while still teaching actual
# selection behavior.
# ============================================================================

CALCULATION_TOOLS = [
    "calculate",
    "read_file",
    "search_files",
    "list_files",
]

MEMORY_TOOLS = [
    "remember_memory",
    "recall_memory",
]

FILE_TOOLS = [
    "read_file",
    "list_files",
    "search_files",
    "run_python_file",
    "debug_python_file",
]

PYTHON_TOOLS = [
    "read_file",
    "run_python_file",
    "debug_python_file",
]

SEARCH_TOOLS = [
    "search_files",
    "list_files",
    "read_file",
]


# ============================================================================
# NEW TARGETED TRAINING EXAMPLES
# ============================================================================
#
# These are intentionally different from the 32 evaluation cases.
#
# We do NOT want to simply memorize the evaluation questions.
#
# Instead, these examples teach the underlying behavior patterns.
# ============================================================================

NEW_EXAMPLES = [

    # ------------------------------------------------------------------------
    # 1. No-tool behavior while all tools are available.
    # ------------------------------------------------------------------------
    {
        "id": "v2_001",
        "available_tools": ALL_TOOLS,
        "messages": [
            {
                "role": "user",
                "content": (
                    "I usually work through equations by hand before "
                    "using a calculator."
                ),
            },
            {
                "role": "assistant",
                "content": "Understood.",
            },
        ],
    },

    # ------------------------------------------------------------------------
    # 2. No-tool conceptual question while all tools are available.
    # ------------------------------------------------------------------------
    {
        "id": "v2_002",
        "available_tools": ALL_TOOLS,
        "messages": [
            {
                "role": "user",
                "content": (
                    "What is the purpose of a PID controller?"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "A PID controller uses proportional, integral, and "
                    "derivative terms to reduce the difference between a "
                    "system's desired output and its measured output. "
                    "The three terms respond to present error, accumulated "
                    "error, and the rate of change of error."
                ),
            },
        ],
    },

    # ------------------------------------------------------------------------
    # 3. Casual statement should not become memory.
    # ------------------------------------------------------------------------
    {
        "id": "v2_003",
        "available_tools": ALL_TOOLS,
        "messages": [
            {
                "role": "user",
                "content": (
                    "I'm working on a C++ assignment this week."
                ),
            },
            {
                "role": "assistant",
                "content": "Understood.",
            },
        ],
    },

    # ------------------------------------------------------------------------
    # 4. Explicit memory write.
    #
    # Both memory tools are available so JARVIS must distinguish WRITE
    # from READ.
    # ------------------------------------------------------------------------
    {
        "id": "v2_004",
        "available_tools": MEMORY_TOOLS,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Please remember that I prefer examples with "
                    "engineering applications."
                ),
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "remember_memory",
                            "arguments": {
                                "key": "example_preference",
                                "value": (
                                    "engineering applications"
                                ),
                            },
                        }
                    }
                ],
            },
        ],
    },

    # ------------------------------------------------------------------------
    # 5. Explicit memory write using different wording.
    # ------------------------------------------------------------------------
    {
        "id": "v2_005",
        "available_tools": MEMORY_TOOLS,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Save this preference: keep explanations "
                    "technically detailed."
                ),
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "remember_memory",
                            "arguments": {
                                "key": "detail_preference",
                                "value": "technically detailed",
                            },
                        }
                    }
                ],
            },
        ],
    },

    # ------------------------------------------------------------------------
    # 6. Memory recall.
    # ------------------------------------------------------------------------
    {
        "id": "v2_006",
        "available_tools": MEMORY_TOOLS,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Can you tell me what you've stored in memory?"
                ),
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "recall_memory",
                            "arguments": {},
                        }
                    }
                ],
            },
        ],
    },

    # ------------------------------------------------------------------------
    # 7. Incomplete memory request.
    #
    # Both memory tools are available, but neither should be called because
    # the user has not actually supplied something complete to store.
    # ------------------------------------------------------------------------
    {
        "id": "v2_007",
        "available_tools": MEMORY_TOOLS,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Remember my preferred explanation style."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "What would you like me to remember about "
                    "your preferred explanation style?"
                ),
            },
        ],
    },

    # ------------------------------------------------------------------------
    # 8. Calculator selection with competing tools.
    # ------------------------------------------------------------------------
    {
        "id": "v2_008",
        "available_tools": CALCULATION_TOOLS,
        "messages": [
            {
                "role": "user",
                "content": "Calculate 73 * 19 - 20.",
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "calculate",
                            "arguments": {
                                "expression": "73 * 19 - 20"
                            },
                        }
                    }
                ],
            },
        ],
    },

    # ------------------------------------------------------------------------
    # 9. Incomplete calculation.
    # ------------------------------------------------------------------------
    {
        "id": "v2_009",
        "available_tools": CALCULATION_TOOLS,
        "messages": [
            {
                "role": "user",
                "content": "Divide 840 by",
            },
            {
                "role": "assistant",
                "content": (
                    "What number would you like me to divide "
                    "840 by?"
                ),
            },
        ],
    },

    # ------------------------------------------------------------------------
    # 10. Known file -> read_file.
    # ------------------------------------------------------------------------
    {
        "id": "v2_010",
        "available_tools": FILE_TOOLS,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Read tools/registry.py."
                ),
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "read_file",
                            "arguments": {
                                "path": "tools/registry.py"
                            },
                        }
                    }
                ],
            },
        ],
    },

    # ------------------------------------------------------------------------
    # 11. "Show files" -> list_files, not search_files.
    # ------------------------------------------------------------------------
    {
        "id": "v2_011",
        "available_tools": FILE_TOOLS,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Show me the files currently in the project."
                ),
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "list_files",
                            "arguments": {},
                        }
                    }
                ],
            },
        ],
    },

    # ------------------------------------------------------------------------
    # 12. Definition lookup -> search_files, not list_files.
    # ------------------------------------------------------------------------
    {
        "id": "v2_012",
        "available_tools": SEARCH_TOOLS,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Where is the implementation of the file reader?"
                ),
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "search_files",
                            "arguments": {
                                "query": "read_file"
                            },
                        }
                    }
                ],
            },
        ],
    },

    # ------------------------------------------------------------------------
    # 13. Search for a specific variable/function.
    # ------------------------------------------------------------------------
    {
        "id": "v2_013",
        "available_tools": SEARCH_TOOLS,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Find where PROJECT_ROOT is used."
                ),
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "search_files",
                            "arguments": {
                                "query": "PROJECT_ROOT"
                            },
                        }
                    }
                ],
            },
        ],
    },

    # ------------------------------------------------------------------------
    # 14. Execute Python.
    # ------------------------------------------------------------------------
    {
        "id": "v2_014",
        "available_tools": PYTHON_TOOLS,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Run model/evaluations/fixtures/"
                    "benchmark_python.py."
                ),
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "run_python_file",
                            "arguments": {
                                "path": (
                                    "model/evaluations/fixtures/"
                                    "benchmark_python.py"
                                ),
                            },
                        }
                    }
                ],
            },
        ],
    },

    # ------------------------------------------------------------------------
    # 15. Debug Python.
    # ------------------------------------------------------------------------
    {
        "id": "v2_015",
        "available_tools": PYTHON_TOOLS,
        "messages": [
            {
                "role": "user",
                "content": "Debug tools/code_runner.py.",
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "debug_python_file",
                            "arguments": {
                                "path": "tools/code_runner.py"
                            },
                        }
                    }
                ],
            },
        ],
    },

    # ------------------------------------------------------------------------
    # 16. Run means RUN, not run + debug.
    #
    # This targets the baseline behavior where Qwen called debug_python_file
    # after successfully choosing run_python_file.
    # ------------------------------------------------------------------------
    {
        "id": "v2_016",
        "available_tools": PYTHON_TOOLS,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Execute tools/calculator.py and report the result."
                ),
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "run_python_file",
                            "arguments": {
                                "path": "tools/calculator.py"
                            },
                        }
                    }
                ],
            },
            {
                "role": "tool",
                "content": (
                    "STDOUT:\n"
                    "42\n"
                    "Exit code: 0"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "The file completed with exit code 0 and "
                    "printed 42."
                ),
            },
        ],
    },

    # ------------------------------------------------------------------------
    # 17. Explicit debug request.
    # ------------------------------------------------------------------------
    {
        "id": "v2_017",
        "available_tools": PYTHON_TOOLS,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Diagnose the problem in test.py."
                ),
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "debug_python_file",
                            "arguments": {
                                "path": "test.py"
                            },
                        }
                    }
                ],
            },
        ],
    },

    # ------------------------------------------------------------------------
    # 18. Search before reading an unknown implementation.
    # ------------------------------------------------------------------------
    {
        "id": "v2_018",
        "available_tools": FILE_TOOLS,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Find where the memory storage function is "
                    "defined."
                ),
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "search_files",
                            "arguments": {
                                "query": "save_memory"
                            },
                        }
                    }
                ],
            },
        ],
    },

    # ------------------------------------------------------------------------
    # 19. Multi-step: search -> read -> explain.
    # ------------------------------------------------------------------------
    {
        "id": "v2_019",
        "available_tools": FILE_TOOLS,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Find the code that selects JARVIS tools, read "
                    "the file, and explain the selection logic."
                ),
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "search_files",
                            "arguments": {
                                "query": "select_tools"
                            },
                        }
                    }
                ],
            },
            {
                "role": "tool",
                "content": (
                    "tools/registry.py:19:"
                    "def select_tools(user_input: str) -> dict:"
                ),
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "read_file",
                            "arguments": {
                                "path": "tools/registry.py"
                            },
                        }
                    }
                ],
            },
            {
                "role": "tool",
                "content": (
                    "def select_tools(user_input: str) -> dict:\n"
                    "    text = user_input.lower()\n"
                    "    selected = {}\n"
                    "    # selection logic"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "The tool-selection logic is implemented in "
                    "tools/registry.py. The select_tools function "
                    "examines the user's input and builds the set of "
                    "tools that should be exposed for that request."
                ),
            },
        ],
    },

    # ------------------------------------------------------------------------
    # 20. Multi-step: find dependency -> read dependency.
    # ------------------------------------------------------------------------
    {
        "id": "v2_020",
        "available_tools": FILE_TOOLS,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Find the debugger implementation and determine "
                    "which execution function it uses."
                ),
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "search_files",
                            "arguments": {
                                "query": "debug_python_file"
                            },
                        }
                    }
                ],
            },
            {
                "role": "tool",
                "content": (
                    "tools/debugger.py:4:"
                    "def debug_python_file(path: str) -> str:"
                ),
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "read_file",
                            "arguments": {
                                "path": "tools/debugger.py"
                            },
                        }
                    }
                ],
            },
            {
                "role": "tool",
                "content": (
                    "from tools.code_runner import run_python_file\n"
                    "\n"
                    "def debug_python_file(path: str) -> str:\n"
                    "    execution = run_python_file(path)\n"
                    "    return execution"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "The debugger implementation is in "
                    "tools/debugger.py, and it uses run_python_file "
                    "to execute the target Python file."
                ),
            },
        ],
    },

    # ------------------------------------------------------------------------
    # 21. Failed search -> recover with a better query.
    #
    # This directly targets the remaining agentic weakness from the
    # evaluation, but uses different wording from the evaluation case.
    # ------------------------------------------------------------------------
    {
        "id": "v2_021",
        "available_tools": FILE_TOOLS,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Locate the implementation of the calculator."
                ),
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "search_files",
                            "arguments": {
                                "query": "calculator implementation"
                            },
                        }
                    }
                ],
            },
            {
                "role": "tool",
                "content": (
                    "No matches found for: "
                    "calculator implementation"
                ),
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "search_files",
                            "arguments": {
                                "query": "calculate"
                            },
                        }
                    }
                ],
            },
            {
                "role": "tool",
                "content": (
                    "tools/calculator.py:4:"
                    "def calculate(expression: str) -> str:"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "The calculator implementation is in "
                    "tools/calculator.py."
                ),
            },
        ],
    },

    # ------------------------------------------------------------------------
    # 22. Failed search -> no fabrication.
    # ------------------------------------------------------------------------
    {
        "id": "v2_022",
        "available_tools": FILE_TOOLS,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Find the nonexistent QuantumFoo function and "
                    "tell me what file it is in."
                ),
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "search_files",
                            "arguments": {
                                "query": "QuantumFoo"
                            },
                        }
                    }
                ],
            },
            {
                "role": "tool",
                "content": (
                    "No matches found for: QuantumFoo"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "I couldn't find a definition of QuantumFoo "
                    "in the search results, so I can't reliably "
                    "identify a file containing it."
                ),
            },
        ],
    },

    # ------------------------------------------------------------------------
    # 23. Search vs list distinction.
    # ------------------------------------------------------------------------
    {
        "id": "v2_023",
        "available_tools": FILE_TOOLS,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Which file contains the memory store function?"
                ),
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "search_files",
                            "arguments": {
                                "query": "save_memory"
                            },
                        }
                    }
                ],
            },
        ],
    },

    # ------------------------------------------------------------------------
    # 24. Execution completion should not trigger an unrelated tool.
    # ------------------------------------------------------------------------
    {
        "id": "v2_024",
        "available_tools": PYTHON_TOOLS,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Run the benchmark fixture and tell me whether "
                    "it finished successfully."
                ),
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "run_python_file",
                            "arguments": {
                                "path": (
                                    "model/evaluations/fixtures/"
                                    "benchmark_python.py"
                                ),
                            },
                        }
                    }
                ],
            },
            {
                "role": "tool",
                "content": (
                    "STDOUT:\n"
                    "Benchmark result: 42\n"
                    "Exit code: 0"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "Yes. The benchmark fixture completed "
                    "successfully with exit code 0."
                ),
            },
        ],
    },
]


# ============================================================================
# LOAD JSONL
# ============================================================================

def load_jsonl(path: Path) -> list[dict]:
    """
    Load a JSONL file and validate every line.

    We use this instead of simply calling json.loads on the entire file
    because JSONL is a sequence of independent JSON objects rather than
    one large JSON document.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}"
        )

    examples = []

    for line_number, line in enumerate(
        path.read_text().splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            examples.append(
                json.loads(line)
            )
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON in {path} at line "
                f"{line_number}: {error}"
            ) from error

    return examples


# ============================================================================
# DETERMINE USED TOOLS
# ============================================================================

def get_used_tools(example: dict) -> list[str]:
    """
    Return the unique tools called by an example.

    The order of first appearance is preserved.

    Example:

        search_files
        read_file
        read_file

    becomes:

        ["search_files", "read_file"]
    """

    used_tools = []

    for message in example.get("messages", []):

        if message.get("role") != "assistant":
            continue

        for tool_call in message.get("tool_calls", []):

            function = tool_call.get("function", {})
            name = function.get("name")

            if not name:
                raise ValueError(
                    "A tool call is missing its function name."
                )

            if name not in ALL_TOOLS:
                raise ValueError(
                    f"Unknown JARVIS tool: {name}"
                )

            if name not in used_tools:
                used_tools.append(name)

    return used_tools


# ============================================================================
# CHOOSE REALISTIC COMPETING TOOLS
# ============================================================================

def choose_available_tools(
    example: dict,
) -> list[str]:
    """
    Decide which tools should be visible during training.

    This function is one of the most important changes for v2.

    We want the model to actually make a selection decision.

    Examples:

        "What files are in the project?"
            -> list_files + related file tools

        "Where is calculate defined?"
            -> search_files + related file tools

        "Remember this preference."
            -> remember_memory + recall_memory

        "Hello."
            -> ALL tools

    No-tool examples receive every tool because we specifically want to
    teach JARVIS that having tools available does not imply that one
    should be used.
    """

    used_tools = get_used_tools(example)

    # ------------------------------------------------------------------------
    # No-tool examples
    #
    # This is crucial for preventing the behavior we saw in the v1 baseline,
    # where JARVIS sometimes called recall_memory, remember_memory, or
    # list_files when the user did not request an operation.
    # ------------------------------------------------------------------------

    if not used_tools:
        return list(ALL_TOOLS)

    # ------------------------------------------------------------------------
    # Memory behavior
    # ------------------------------------------------------------------------

    if any(
        tool in {"remember_memory", "recall_memory"}
        for tool in used_tools
    ):
        return list(MEMORY_TOOLS)

    # ------------------------------------------------------------------------
    # Calculator behavior
    # ------------------------------------------------------------------------

    if "calculate" in used_tools:
        return list(CALCULATION_TOOLS)

    # ------------------------------------------------------------------------
    # Python execution/debugging
    # ------------------------------------------------------------------------

    if any(
        tool in {
            "run_python_file",
            "debug_python_file",
        }
        for tool in used_tools
    ):
        return list(PYTHON_TOOLS)

    # ------------------------------------------------------------------------
    # General project/file investigation
    # ------------------------------------------------------------------------

    if any(
        tool in {
            "read_file",
            "list_files",
            "search_files",
        }
        for tool in used_tools
    ):
        return list(FILE_TOOLS)

    # ------------------------------------------------------------------------
    # Safety fallback
    #
    # If a future tool gets added and isn't yet assigned to one of the
    # groups above, exposing all tools is preferable to silently training
    # the model with an artificially restricted tool set.
    # ------------------------------------------------------------------------

    return list(ALL_TOOLS)


# ============================================================================
# ANNOTATE EXAMPLE
# ============================================================================

def annotate_example(example: dict) -> dict:
    """
    Add the v2 available_tools annotation.

    The original example object is copied rather than modified in place.

    This is important because our original training dataset is treated as
    an immutable reference corpus.
    """

    annotated = deepcopy(example)

    annotated["available_tools"] = choose_available_tools(
        annotated
    )

    return annotated


# ============================================================================
# DEDUPLICATION
# ============================================================================

def deduplicate_examples(
    examples: list[dict],
) -> list[dict]:
    """
    Remove exact duplicate examples.

    We serialize each example into canonical JSON so that dictionaries with
    the same contents produce the same comparison key.

    This protects the v2 corpus from accidental duplicate entries without
    trying to judge whether two differently worded examples are semantically
    similar.
    """

    unique = []
    seen = set()

    for example in examples:

        key = json.dumps(
            example,
            sort_keys=True,
            ensure_ascii=False,
        )

        if key in seen:
            continue

        seen.add(key)
        unique.append(example)

    return unique


# ============================================================================
# WRITE JSONL
# ============================================================================

def write_jsonl(
    path: Path,
    examples: list[dict],
) -> None:
    """
    Write examples as one JSON object per line.

    ensure_ascii=False preserves readable characters instead of converting
    everything into escaped Unicode sequences.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        "\n".join(
            json.dumps(
                example,
                ensure_ascii=False,
            )
            for example in examples
        )
        + "\n"
    )


# ============================================================================
# SUMMARY
# ============================================================================

def print_summary(
    train_examples: list[dict],
    valid_examples: list[dict],
) -> None:
    """
    Print a concise report describing the v2 dataset.
    """

    print()
    print("=== JARVIS V2 TRAINING DATA ===")
    print()
    print(f"Training examples:   {len(train_examples)}")
    print(f"Validation examples: {len(valid_examples)}")
    print()

    print("Training tool behavior:")

    counts = {}

    for example in train_examples:

        tools = tuple(
            example.get("available_tools", [])
        )

        counts[tools] = counts.get(tools, 0) + 1

    for tools, count in sorted(
        counts.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        print(
            f"  {count:3d} examples -> "
            f"{', '.join(tools)}"
        )

    print()
    print("The original 32-case evaluation set was NOT modified.")
    print("The original 16-case validation set was preserved.")
    print()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=== BUILDING JARVIS V2 TRAINING DATA ===")
    print()

    # ------------------------------------------------------------------------
    # Load only the ORIGINAL TRAINING split.
    #
    # This is extremely important.
    #
    # We deliberately do NOT load:
    #
    #     original test set
    #
    # because those examples must remain held out.
    # ------------------------------------------------------------------------

    original_train = load_jsonl(
        ORIGINAL_TRAIN_FILE
    )

    # The validation set is preserved rather than regenerated.
    #
    # This makes comparisons across training experiments easier because
    # v1 and v2 can use the same validation cases.
    original_valid = load_jsonl(
        ORIGINAL_VALID_FILE
    )

    print(
        f"Original training examples loaded: "
        f"{len(original_train)}"
    )

    print(
        f"Original validation examples loaded: "
        f"{len(original_valid)}"
    )

    print(
        f"New targeted examples: "
        f"{len(NEW_EXAMPLES)}"
    )

    # ------------------------------------------------------------------------
    # Add available-tool annotations to the original training examples.
    # ------------------------------------------------------------------------

    annotated_original_train = [
        annotate_example(example)
        for example in original_train
    ]

    # ------------------------------------------------------------------------
    # Add annotations to the new targeted examples.
    #
    # These examples already represent the behavior we want, so the same
    # annotation mechanism makes their metadata consistent with the original
    # training examples.
    # ------------------------------------------------------------------------

    annotated_new_examples = [
        annotate_example(example)
        for example in NEW_EXAMPLES
    ]

    # ------------------------------------------------------------------------
    # Combine.
    # ------------------------------------------------------------------------

    combined_train = (
        annotated_original_train
        + annotated_new_examples
    )

    # ------------------------------------------------------------------------
    # Deduplicate.
    #
    # The original corpus has already been cleaned, but doing this again after
    # combining v2 examples gives us an additional safeguard.
    # ------------------------------------------------------------------------

    combined_train = deduplicate_examples(
        combined_train
    )

    # ------------------------------------------------------------------------
    # Preserve the existing validation set.
    #
    # We annotate it with the same available-tools logic so that validation
    # uses the same realistic tool-selection context as v2 training.
    #
    # IMPORTANT:
    #
    # The examples themselves are unchanged. Only the available_tools
    # metadata is added.
    # ------------------------------------------------------------------------

    annotated_valid = [
        annotate_example(example)
        for example in original_valid
    ]

    # ------------------------------------------------------------------------
    # Write the v2 datasets.
    # ------------------------------------------------------------------------

    write_jsonl(
        OUTPUT_TRAIN_FILE,
        combined_train,
    )

    write_jsonl(
        OUTPUT_VALID_FILE,
        annotated_valid,
    )

    # ------------------------------------------------------------------------
    # Safety checks.
    # ------------------------------------------------------------------------

    train_ids = [
        example.get("id")
        for example in combined_train
        if example.get("id") is not None
    ]

    valid_ids = [
        example.get("id")
        for example in annotated_valid
        if example.get("id") is not None
    ]

    overlapping_ids = (
        set(train_ids)
        & set(valid_ids)
    )

    if overlapping_ids:
        raise ValueError(
            "Training and validation contain overlapping IDs: "
            f"{sorted(overlapping_ids)}"
        )

    # ------------------------------------------------------------------------
    # Print summary.
    # ------------------------------------------------------------------------

    print_summary(
        combined_train,
        annotated_valid,
    )

    print(
        "Training file:"
    )
    print(
        f"  {OUTPUT_TRAIN_FILE}"
    )

    print(
        "Validation file:"
    )
    print(
        f"  {OUTPUT_VALID_FILE}"
    )


if __name__ == "__main__":
    main()