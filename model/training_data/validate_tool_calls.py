import json
import inspect
from pathlib import Path

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.registry import TOOLS

DATA_FILE = PROJECT_ROOT / "model" / "training_data" / "jarvis_behavior.jsonl"


def validate_tool_call(tool_name, arguments, location):
    errors = []

    if tool_name not in TOOLS:
        return [f"{location}: unknown tool '{tool_name}'"]

    tool = TOOLS[tool_name]
    signature = inspect.signature(tool)

    expected_parameters = set(signature.parameters.keys())
    provided_parameters = set(arguments.keys())

    unexpected = provided_parameters - expected_parameters

    if unexpected:
        errors.append(
            f"{location}: tool '{tool_name}' received unexpected "
            f"arguments: {sorted(unexpected)}"
        )

    missing = {
        name
        for name, parameter in signature.parameters.items()
        if (
            parameter.default is inspect.Parameter.empty
            and name not in provided_parameters
        )
    }

    if missing:
        errors.append(
            f"{location}: tool '{tool_name}' is missing required "
            f"arguments: {sorted(missing)}"
        )

    return errors


def main():
    if not DATA_FILE.exists():
        print(f"ERROR: File not found: {DATA_FILE}")
        return

    errors = []
    tool_counts = {}

    lines = DATA_FILE.read_text().splitlines()

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue

        entry = json.loads(line)

        for message_index, message in enumerate(
            entry.get("messages", [])
        ):
            tool_calls = message.get("tool_calls", [])

            for call_index, call in enumerate(tool_calls):
                function = call.get("function", {})
                tool_name = function.get("name")
                arguments = function.get("arguments", {})

                if not isinstance(arguments, dict):
                    errors.append(
                        f"Line {line_number}, message {message_index}, "
                        f"tool call {call_index}: arguments must be an object"
                    )
                    continue

                tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1

                errors.extend(
                    validate_tool_call(
                        tool_name,
                        arguments,
                        (
                            f"Line {line_number}, "
                            f"message {message_index}, "
                            f"tool call {call_index}"
                        ),
                    )
                )

    print("=== JARVIS TOOL-CALL VALIDATION ===")
    print(f"Dataset: {DATA_FILE}")
    print()

    print("Tool usage in dataset:")
    for tool_name, count in sorted(tool_counts.items()):
        print(f"  {tool_name}: {count}")

    print()

    if errors:
        print("=== ERRORS ===")
        for error in errors:
            print(error)

        print()
        print(f"Total tool-call errors: {len(errors)}")
    else:
        print("No tool-call schema errors found.")
        print("All tool calls match the current Python tool signatures.")


if __name__ == "__main__":
    main()