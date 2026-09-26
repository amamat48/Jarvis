"""
Create the 32-case JARVIS evaluation set.

WHY THIS EXISTS
---------------

Our original dataset contains:

    113 training examples
    16 validation examples
    16 test examples

The original 16-case test set is useful, but a few individual decisions
can have a large effect on the overall percentage.

We therefore expand the evaluation set to 32 cases.

IMPORTANT:

    The additional 16 cases are evaluation-only examples.

They are NOT added to the training corpus.

This preserves the existing 113-example training set while giving us
a larger behavioral evaluation.

Final evaluation:

    16 original held-out test cases
    +
    16 new evaluation-only cases
    =
    32 total cases

The new cases target:

    - tool selection
    - no-tool decisions
    - missing information
    - memory behavior
    - code inspection
    - Python execution
    - debugging
    - multi-step investigation
    - failed-search recovery

The script converts the new cases into the same MLX/Qwen representation
used by the existing evaluation dataset.
"""

import json
import sys
from pathlib import Path


# ============================================================================
# PROJECT PATH SETUP
# ============================================================================
#
# This script is located at:
#
#     jarvis/model/evaluations/create_eval32.py
#
# Path.parents gives us:
#
#     parents[0] -> model/evaluations
#     parents[1] -> model
#     parents[2] -> jarvis
#
# We explicitly add the JARVIS project root to sys.path so that imports such
# as:
#
#     from model.training_data.mlx_tools import TOOLS_SCHEMA
#
# work correctly when this file is executed directly.
#
# This is the same import-path pattern used by our benchmark and evaluation
# scripts elsewhere in the project.
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# PROJECT IMPORTS
# ============================================================================

from model.training_data.mlx_tools import TOOLS_SCHEMA


# ============================================================================
# PROJECT PATHS
# ============================================================================

SOURCE_TEST_FILE = (
    PROJECT_ROOT
    / "model"
    / "training_data"
    / "mlx"
    / "test.jsonl"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "model"
    / "evaluations"
    / "jarvis_eval_32.jsonl"
)


# ============================================================================
# TOOL LOOKUP
# ============================================================================
#
# TOOLS_SCHEMA is a list of tool-definition dictionaries.
#
# We convert that list into a dictionary keyed by tool name so that looking
# up a schema is straightforward:
#
#     TOOL_MAP["calculate"]
#
# instead of repeatedly iterating through the entire list.
#
# This also lets us verify that evaluation cases only reference tools that
# actually exist in the JARVIS tool schema.
# ============================================================================

TOOL_MAP = {}

for tool in TOOLS_SCHEMA:
    function = tool.get("function", {})
    name = function.get("name")

    if not name:
        raise ValueError(
            "A tool schema is missing its function name."
        )

    TOOL_MAP[name] = tool


# ============================================================================
# NEW EVALUATION CASES
# ============================================================================
#
# These cases are intentionally NOT copied into the training dataset.
#
# They exist only to evaluate whether JARVIS can generalize the behaviors
# we've been teaching it.
#
# The cases are written in the same internal representation we've used for
# our behavior corpus:
#
#     assistant tool call:
#
#         {
#             "function": {
#                 "name": "...",
#                 "arguments": {...}
#             }
#         }
#
# and:
#
#     tool result:
#
#         {
#             "role": "tool",
#             "content": "..."
#         }
#
# They are converted into MLX/Qwen format below.
# ============================================================================

NEW_CASES = [

    # ------------------------------------------------------------------------
    # 1. No-tool technical explanation
    # ------------------------------------------------------------------------
    {
        "id": "extension_01",
        "messages": [
            {
                "role": "user",
                "content": (
                    "What does feedback mean in a control system?"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "Feedback means measuring a system's output and "
                    "using that information to influence the input. "
                    "In a feedback control system, the measured output "
                    "is compared with the desired value and the controller "
                    "uses the difference to adjust the system."
                ),
            },
        ],
    },

    # ------------------------------------------------------------------------
    # 2. Calculator selection
    # ------------------------------------------------------------------------
    {
        "id": "extension_02",
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
    # 3. Statement that should NOT automatically become memory
    # ------------------------------------------------------------------------
    {
        "id": "extension_03",
        "messages": [
            {
                "role": "user",
                "content": (
                    "I really prefer C++ over Python."
                ),
            },
            {
                "role": "assistant",
                "content": "Understood.",
            },
        ],
    },

    # ------------------------------------------------------------------------
    # 4. Known-file inspection
    # ------------------------------------------------------------------------
    {
        "id": "extension_04",
        "messages": [
            {
                "role": "user",
                "content": "Please inspect brain/prompt.py.",
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "read_file",
                            "arguments": {
                                "path": "brain/prompt.py"
                            },
                        }
                    }
                ],
            },
        ],
    },

    # ------------------------------------------------------------------------
    # 5. Find a definition
    # ------------------------------------------------------------------------
    {
        "id": "extension_05",
        "messages": [
            {
                "role": "user",
                "content": "Where is select_tools defined?",
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
        ],
    },

    # ------------------------------------------------------------------------
    # 6. Project listing
    # ------------------------------------------------------------------------
    {
        "id": "extension_06",
        "messages": [
            {
                "role": "user",
                "content": "Give me a listing of the JARVIS project files.",
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
    # 7. Python execution
    # ------------------------------------------------------------------------
    {
        "id": "extension_07",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Execute model/evaluations/qwen_baseline.py."
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
                                    "model/evaluations/"
                                    "qwen_baseline.py"
                                )
                            },
                        }
                    }
                ],
            },
        ],
    },

    # ------------------------------------------------------------------------
    # 8. Python debugging
    # ------------------------------------------------------------------------
    {
        "id": "extension_08",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Diagnose the problem in "
                    "model/evaluations/run_benchmark.py."
                ),
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "debug_python_file",
                            "arguments": {
                                "path": (
                                    "model/evaluations/"
                                    "run_benchmark.py"
                                )
                            },
                        }
                    }
                ],
            },
        ],
    },

    # ------------------------------------------------------------------------
    # 9. Missing file information
    # ------------------------------------------------------------------------
    {
        "id": "extension_09",
        "messages": [
            {
                "role": "user",
                "content": "Read the file.",
            },
            {
                "role": "assistant",
                "content": (
                    "Which file would you like me to read?"
                ),
            },
        ],
    },

    # ------------------------------------------------------------------------
    # 10. Incomplete calculation
    # ------------------------------------------------------------------------
    {
        "id": "extension_10",
        "messages": [
            {
                "role": "user",
                "content": "What is 3973 divided by",
            },
            {
                "role": "assistant",
                "content": (
                    "What number would you like me to divide "
                    "3973 by?"
                ),
            },
        ],
    },

    # ------------------------------------------------------------------------
    # 11. Explicit memory write
    # ------------------------------------------------------------------------
    {
        "id": "extension_11",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Remember that I prefer concise technical "
                    "explanations."
                ),
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "remember_memory",
                            "arguments": {
                                "key": "explanation_preference",
                                "value": "concise technical explanations",
                            },
                        }
                    }
                ],
            },
        ],
    },

    # ------------------------------------------------------------------------
    # 12. Memory recall followed by final response
    # ------------------------------------------------------------------------
    {
        "id": "extension_12",
        "messages": [
            {
                "role": "user",
                "content": "What memories do you have?",
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
            {
                "role": "tool",
                "content": (
                    "{\"favorite_language\":\"C++\","
                    "\"explanation_preference\":\"concise\"}"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "I remember that you prefer C++ and concise "
                    "explanations."
                ),
            },
        ],
    },

    # ------------------------------------------------------------------------
    # 13. Multi-step: find -> read -> explain
    # ------------------------------------------------------------------------
    {
        "id": "extension_13",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Find the definition of remember_memory, "
                    "read its file, and explain how it works."
                ),
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "search_files",
                            "arguments": {
                                "query": "remember_memory"
                            },
                        }
                    }
                ],
            },
            {
                "role": "tool",
                "content": (
                    "tools/memory_tools.py:4:"
                    "def remember_memory(key: str, value: str) -> str:"
                ),
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "read_file",
                            "arguments": {
                                "path": "tools/memory_tools.py"
                            },
                        }
                    }
                ],
            },
            {
                "role": "tool",
                "content": (
                    "from memory.store import remember as store_memory\n"
                    "\n"
                    "def remember_memory(key: str, value: str) -> str:\n"
                    "    return store_memory(key, value)"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "remember_memory is defined in "
                    "tools/memory_tools.py. It passes the supplied key "
                    "and value to the persistent memory storage function "
                    "in memory.store."
                ),
            },
        ],
    },

    # ------------------------------------------------------------------------
    # 14. Multi-step dependency investigation
    # ------------------------------------------------------------------------
    {
        "id": "extension_14",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Find debug_python_file, read its implementation, "
                    "and identify what execution tool it calls."
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
                    "from tools.file_reader import read_file\n"
                    "from tools.code_runner import run_python_file\n"
                    "\n"
                    "def debug_python_file(path: str) -> str:\n"
                    "    source = read_file(path)\n"
                    "    if source.startswith(\"File not found:\"):\n"
                    "        return source\n"
                    "    execution = run_python_file(path)\n"
                    "    return source + \"\\n\" + execution"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "debug_python_file uses run_python_file to execute "
                    "the target Python file after reading its source."
                ),
            },
        ],
    },

    # ------------------------------------------------------------------------
    # 15. Failed search -> recover -> read -> explain
    #
    # This deliberately resembles the agent_01 failure from our original
    # benchmark. The first search is unsuccessful, but another reasonable
    # search query can recover the task.
    # ------------------------------------------------------------------------
    {
        "id": "extension_15",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Find the calculator implementation, read the "
                    "file containing it, and explain how it works."
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
                    "No matches found for: calculator implementation"
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
                "tool_calls": [
                    {
                        "function": {
                            "name": "read_file",
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
                    "def calculate(expression: str) -> str:\n"
                    "    try:\n"
                    "        result = eval(\n"
                    "            expression,\n"
                    "            {\"__builtins__\": {}},\n"
                    "            {},\n"
                    "        )\n"
                    "        return str(result)\n"
                    "    except Exception as error:\n"
                    "        return f\"Error: {error}\""
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "The calculator implementation is in "
                    "tools/calculator.py. The calculate function accepts "
                    "a string expression, evaluates it in a restricted "
                    "environment, converts the result to a string, and "
                    "returns an error message if evaluation fails."
                ),
            },
        ],
    },

    # ------------------------------------------------------------------------
    # 16. Execute -> report, without unnecessary follow-up
    # ------------------------------------------------------------------------
    {
        "id": "extension_16",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Run model/evaluations/fixtures/"
                    "benchmark_python.py and tell me whether it "
                    "completed without errors."
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
                                )
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
                    "Yes. The script completed without a runtime "
                    "error and exited with code 0."
                ),
            },
        ],
    },
]


# ============================================================================
# TOOL DISCOVERY
# ============================================================================

def get_tools_used(messages):
    """
    Return the unique tools called anywhere in one evaluation case.

    This allows the generated case to contain only the schemas for tools
    actually used by that case.

    Example:

        search_files -> read_file

    produces schemas for:

        search_files
        read_file

    rather than all eight JARVIS tools.
    """

    used_tools = []

    for message in messages:
        if message.get("role") != "assistant":
            continue

        for tool_call in message.get("tool_calls", []):
            function = tool_call.get("function", {})
            name = function.get("name")

            if not name:
                raise ValueError(
                    "A tool call is missing its function name."
                )

            if name not in TOOL_MAP:
                raise ValueError(
                    f"Unknown tool in evaluation case: {name}"
                )

            if name not in used_tools:
                used_tools.append(name)

    return used_tools


# ============================================================================
# TOOL-CALL CONVERSION
# ============================================================================

def convert_tool_call(
    tool_call,
    case_index,
    call_index,
):
    """
    Convert one internal JARVIS tool call into MLX/Qwen format.

    Internal form:

        "arguments": {
            "expression": "73 * 19"
        }

    Converted form:

        "arguments": "{\"expression\": \"73 * 19\"}"

    Qwen's MLX training format represents function arguments as a JSON
    string inside the function-call structure.

    A unique call ID is also generated so that tool responses can be
    associated with the corresponding tool call.
    """

    function = tool_call.get("function", {})

    name = function.get("name")
    arguments = function.get("arguments", {})

    if name not in TOOL_MAP:
        raise ValueError(
            f"Unknown tool: {name}"
        )

    if not isinstance(arguments, dict):
        raise ValueError(
            f"Arguments for {name} must be a dictionary."
        )

    return {
        "id": (
            f"eval_{case_index:03d}_"
            f"call_{call_index:03d}"
        ),
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(
                arguments,
                ensure_ascii=False,
            ),
        },
    }


# ============================================================================
# CASE CONVERSION
# ============================================================================

def convert_case(case, case_index):
    """
    Convert one internal evaluation case into MLX/Qwen format.

    Tool calls are converted into structured function calls.

    Tool-result messages are assigned the name of the corresponding
    preceding tool call.

    A queue is used so that if a future example contains multiple tool
    calls, their results can still be associated in order.
    """

    messages = case.get("messages", [])

    if not messages:
        raise ValueError(
            f"Evaluation case {case_index} contains no messages."
        )

    converted_messages = []

    pending_tool_names = []
    call_counter = 0

    for message in messages:
        role = message.get("role")

        # --------------------------------------------------------------------
        # Assistant tool call
        # --------------------------------------------------------------------
        if role == "assistant" and "tool_calls" in message:

            converted_calls = []

            for tool_call in message.get("tool_calls", []):
                call_counter += 1

                converted_call = convert_tool_call(
                    tool_call,
                    case_index,
                    call_counter,
                )

                converted_calls.append(
                    converted_call
                )

                pending_tool_names.append(
                    tool_call["function"]["name"]
                )

            assistant_message = {
                "role": "assistant",
                "tool_calls": converted_calls,
            }

            if "content" in message:
                assistant_message["content"] = (
                    message["content"]
                )

            converted_messages.append(
                assistant_message
            )

        # --------------------------------------------------------------------
        # Tool result
        # --------------------------------------------------------------------
        elif role == "tool":

            if not pending_tool_names:
                raise ValueError(
                    f"Evaluation case {case_index} contains a tool "
                    "result with no preceding tool call."
                )

            tool_name = pending_tool_names.pop(0)

            converted_messages.append({
                "role": "tool",
                "name": tool_name,
                "content": message.get("content", ""),
            })

        # --------------------------------------------------------------------
        # Ordinary user / assistant message
        # --------------------------------------------------------------------
        else:
            converted_messages.append(message)

    result = {
        "id": case.get(
            "id",
            f"evaluation_{case_index:03d}",
        ),
        "messages": converted_messages,
    }

    # Only attach schemas for tools actually involved in this case.
    used_tools = get_tools_used(messages)

    if used_tools:
        result["tools"] = [
            TOOL_MAP[name]
            for name in used_tools
        ]

    return result


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=== CREATING 32-CASE JARVIS EVALUATION ===")
    print()

    # ------------------------------------------------------------------------
    # Verify the existing 16-case test set.
    # ------------------------------------------------------------------------

    if not SOURCE_TEST_FILE.exists():
        raise FileNotFoundError(
            f"Existing test set not found:\n"
            f"{SOURCE_TEST_FILE}"
        )

    existing_cases = []

    for line_number, line in enumerate(
        SOURCE_TEST_FILE.read_text().splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            existing_cases.append(
                json.loads(line)
            )
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON in existing test set at "
                f"line {line_number}: {error}"
            ) from error

    print(
        f"Existing held-out cases: "
        f"{len(existing_cases)}"
    )

    if len(existing_cases) != 16:
        raise ValueError(
            "Expected the original test set to contain "
            f"16 examples, but found {len(existing_cases)}."
        )

    # ------------------------------------------------------------------------
    # Convert the new 16 cases.
    # ------------------------------------------------------------------------

    converted_new_cases = []

    for index, case in enumerate(
        NEW_CASES,
        start=len(existing_cases) + 1,
    ):
        converted_new_cases.append(
            convert_case(
                case,
                index,
            )
        )

    print(
        f"New evaluation-only cases: "
        f"{len(converted_new_cases)}"
    )

    # ------------------------------------------------------------------------
    # Combine old and new cases.
    #
    # The original cases remain untouched.
    # The additional 16 cases are appended.
    # ------------------------------------------------------------------------

    combined = (
        existing_cases
        + converted_new_cases
    )

    if len(combined) != 32:
        raise ValueError(
            f"Expected 32 evaluation cases, "
            f"created {len(combined)}."
        )

    # ------------------------------------------------------------------------
    # Check case IDs.
    #
    # Duplicate IDs would make result tracking ambiguous.
    # ------------------------------------------------------------------------

    ids = [
        case.get("id")
        for case in combined
    ]

    non_null_ids = [
        case_id
        for case_id in ids
        if case_id is not None
    ]

    if len(non_null_ids) != len(set(non_null_ids)):
        raise ValueError(
            "Duplicate evaluation case IDs were detected."
        )

    # ------------------------------------------------------------------------
    # Write the 32-case evaluation set.
    # ------------------------------------------------------------------------

    OUTPUT_FILE.write_text(
        "\n".join(
            json.dumps(
                case,
                ensure_ascii=False,
            )
            for case in combined
        )
        + "\n"
    )

    # ------------------------------------------------------------------------
    # Count cases that involve tools versus cases that require no tool.
    # ------------------------------------------------------------------------

    tool_case_count = 0
    no_tool_case_count = 0

    for case in combined:
        has_tool_call = any(
            "tool_calls" in message
            for message in case.get("messages", [])
            if message.get("role") == "assistant"
        )

        if has_tool_call:
            tool_case_count += 1
        else:
            no_tool_case_count += 1

    # ------------------------------------------------------------------------
    # Final report.
    # ------------------------------------------------------------------------

    print()
    print("=== EVALUATION CREATED ===")
    print(f"Total cases:       {len(combined)}")
    print(f"Tool-use cases:    {tool_case_count}")
    print(f"No-tool cases:     {no_tool_case_count}")
    print()
    print("The original training and validation datasets were NOT modified.")
    print()
    print("Evaluation file:")
    print(f"  {OUTPUT_FILE}")


if __name__ == "__main__":
    main()