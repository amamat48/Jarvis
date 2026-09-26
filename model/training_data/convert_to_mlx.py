"""
Convert JARVIS training data into the JSONL format expected by MLX-LM.

V2 change:
    Each training example can contain an "available_tools" field.

That field tells the converter which tools the model should see for that
example. This is important because JARVIS V1 was trained with only the
expected tool(s) exposed, while evaluation gave the model all tools.

For V2:
    - "no tool" examples can expose all tools.
    - memory examples expose both memory tools.
    - file examples expose competing file tools.
    - Python examples expose read/run/debug tools.
    - search examples expose search/list/read tools.

The converter also generates tool schemas from the real functions in
tools.registry so the training data stays synchronized with the actual
JARVIS tools.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from typing import Any, get_args, get_origin


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "model" / "training_data" / "v2" / "train.jsonl"
VALID_FILE = PROJECT_ROOT / "model" / "training_data" / "v2" / "valid.jsonl"

OUTPUT_DIR = PROJECT_ROOT / "model" / "training_data" / "v2" / "mlx"
OUTPUT_TRAIN = OUTPUT_DIR / "train.jsonl"
OUTPUT_VALID = OUTPUT_DIR / "valid.jsonl"


# ---------------------------------------------------------------------------
# Import the real JARVIS tools.
#
# Because this script is inside the project, adding the project root to
# sys.path lets us reuse the actual tool registry instead of duplicating
# tool definitions here.
# ---------------------------------------------------------------------------

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.registry import TOOLS  # noqa: E402


# ---------------------------------------------------------------------------
# Basic Python-type -> JSON-schema conversion.
#
# Most of the current JARVIS tools use simple string arguments, so we do not
# need a large schema-generation framework here.
# ---------------------------------------------------------------------------

def annotation_to_schema(annotation: Any) -> dict[str, Any]:
    """
    Convert a Python type annotation into a small JSON Schema fragment.

    This intentionally handles only the common types JARVIS currently uses.
    Keeping this simple is preferable to introducing a dependency just for
    schema generation.
    """

    if annotation is inspect.Signature.empty:
        return {"type": "string"}

    # Handle plain built-in types.
    if annotation is str:
        return {"type": "string"}

    if annotation is int:
        return {"type": "integer"}

    if annotation is float:
        return {"type": "number"}

    if annotation is bool:
        return {"type": "boolean"}

    # Handle simple Optional[T] / Union[T, None].
    origin = get_origin(annotation)

    if origin is not None:
        args = [arg for arg in get_args(annotation) if arg is not type(None)]

        if len(args) == 1:
            return annotation_to_schema(args[0])

    # Safe fallback for anything more complicated.
    return {"type": "string"}


# ---------------------------------------------------------------------------
# Build one MLX-compatible tool schema from a real Python function.
# ---------------------------------------------------------------------------

def build_tool_schema(name: str, function: Any) -> dict[str, Any]:
    """
    Create the tool schema MLX-LM expects from a registered JARVIS tool.

    The schema is intentionally generated at runtime so that changes to
    the actual tool function signature automatically flow into training.
    """

    signature = inspect.signature(function)

    properties: dict[str, Any] = {}
    required: list[str] = []

    for parameter_name, parameter in signature.parameters.items():
        # Ignore *args and **kwargs because they are not useful for the
        # structured function-calling interface.
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        parameter_schema = annotation_to_schema(parameter.annotation)

        properties[parameter_name] = parameter_schema

        # A parameter without a default value is required.
        if parameter.default is inspect.Signature.empty:
            required.append(parameter_name)

    description = inspect.getdoc(function) or f"JARVIS tool: {name}"

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


# ---------------------------------------------------------------------------
# Build all tool schemas once.
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = {
    name: build_tool_schema(name, function)
    for name, function in TOOLS.items()
}


# ---------------------------------------------------------------------------
# Helper: determine which tools are actually used by an example.
#
# This is used as a fallback for older datasets that do not contain
# "available_tools".
# ---------------------------------------------------------------------------

def tools_used_by_example(example: dict[str, Any]) -> list[str]:
    """
    Find tool names referenced by assistant tool calls.

    Supports the normal OpenAI-style:
        assistant.message["tool_calls"]

    This makes the converter backward-compatible with the older V1 data.
    """

    used: list[str] = []

    for message in example.get("messages", []):
        tool_calls = message.get("tool_calls", [])

        for call in tool_calls:
            function = call.get("function", {})
            name = function.get("name")

            if name and name in TOOLS and name not in used:
                used.append(name)

    return used


# ---------------------------------------------------------------------------
# Helper: normalize tool-call arguments.
#
# MLX-LM expects function arguments represented as JSON text in the
# tool-call structure.
# ---------------------------------------------------------------------------

def normalize_tool_calls(message: dict[str, Any]) -> dict[str, Any]:
    """
    Ensure assistant tool calls have JSON-string arguments.

    If arguments are already strings, they are left alone.
    If they are Python dictionaries, they are serialized.
    """

    if "tool_calls" not in message:
        return message

    normalized = dict(message)
    normalized_calls = []

    for call in message["tool_calls"]:
        call_copy = dict(call)
        function = dict(call_copy.get("function", {}))

        arguments = function.get("arguments", {})

        if not isinstance(arguments, str):
            function["arguments"] = json.dumps(
                arguments,
                separators=(",", ":"),
            )

        call_copy["function"] = function
        normalized_calls.append(call_copy)

    normalized["tool_calls"] = normalized_calls

    return normalized


# ---------------------------------------------------------------------------
# Helper: make tool result messages consistent.
#
# Some earlier training data did not explicitly attach the tool name to the
# tool-result message. We recover it from the preceding assistant tool call.
# ---------------------------------------------------------------------------

def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Normalize tool-call and tool-result messages.

    The tool-result message should contain:
        role = "tool"
        name = "<tool name>"
        content = "<result>"
    """

    normalized: list[dict[str, Any]] = []

    # Maps tool_call_id -> tool name.
    pending_tools: dict[str, str] = {}

    for message in messages:
        message_copy = dict(message)
        role = message_copy.get("role")

        # Track tool calls made by the assistant.
        if role == "assistant":
            message_copy = normalize_tool_calls(message_copy)

            for call in message_copy.get("tool_calls", []):
                call_id = call.get("id")
                function_name = call.get("function", {}).get("name")

                if call_id and function_name:
                    pending_tools[call_id] = function_name

        # Recover the tool name on tool-result messages when necessary.
        elif role == "tool":
            call_id = message_copy.get("tool_call_id")

            if not message_copy.get("name") and call_id:
                tool_name = pending_tools.get(call_id)

                if tool_name:
                    message_copy["name"] = tool_name

        normalized.append(message_copy)

    return normalized


# ---------------------------------------------------------------------------
# Convert one example.
# ---------------------------------------------------------------------------

def convert_example(example: dict[str, Any]) -> dict[str, Any]:
    """
    Convert one JARVIS example to MLX-LM's chat/tool format.

    V2 behavior:
        1. Prefer the example's "available_tools".
        2. Fall back to the tools actually used for older data.
        3. Omit the "tools" field entirely when no tools are available.
    """

    converted = {
        "messages": normalize_messages(example.get("messages", []))
    }

    available_tools = example.get("available_tools")

    if available_tools is None:
        # Backward compatibility with older V1 data.
        available_tools = tools_used_by_example(example)

    # Remove duplicate names while preserving order.
    available_tools = list(dict.fromkeys(available_tools))

    # Verify that every requested tool exists in the live registry.
    unknown_tools = [
        name for name in available_tools
        if name not in TOOL_SCHEMAS
    ]

    if unknown_tools:
        raise ValueError(
            f"Unknown tools in training example: {unknown_tools}"
        )

    # MLX does not need a tools field when the example has no available
    # tools. This is especially important for genuine no-tool examples.
    if available_tools:
        converted["tools"] = [
            TOOL_SCHEMAS[name]
            for name in available_tools
        ]

    return converted


# ---------------------------------------------------------------------------
# Read JSONL.
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dictionaries."""

    examples: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} on line {line_number}: {exc}"
                ) from exc

    return examples


# ---------------------------------------------------------------------------
# Write JSONL.
# ---------------------------------------------------------------------------

def write_jsonl(path: Path, examples: list[dict[str, Any]]) -> None:
    """Write dictionaries as one JSON object per line."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for example in examples:
            file.write(
                json.dumps(
                    example,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )


# ---------------------------------------------------------------------------
# Basic sanity checks.
# ---------------------------------------------------------------------------

def validate_examples(
    original: list[dict[str, Any]],
    converted: list[dict[str, Any]],
) -> None:
    """
    Run simple structural checks before writing the dataset.

    This is intentionally lightweight. The dedicated MLX validator will do
    deeper validation afterward.
    """

    if len(original) != len(converted):
        raise ValueError(
            f"Example count changed: {len(original)} -> {len(converted)}"
        )

    for index, example in enumerate(converted):
        messages = example.get("messages")

        if not isinstance(messages, list):
            raise ValueError(
                f"Example {index} does not contain a messages list."
            )

        if "tools" in example:
            if not isinstance(example["tools"], list):
                raise ValueError(
                    f"Example {index} has an invalid tools field."
                )

            for tool in example["tools"]:
                function = tool.get("function", {})
                name = function.get("name")

                if name not in TOOL_SCHEMAS:
                    raise ValueError(
                        f"Example {index} contains unknown tool: {name}"
                    )


# ---------------------------------------------------------------------------
# Convert one dataset split.
# ---------------------------------------------------------------------------

def convert_split(input_path: Path, output_path: Path) -> int:
    """
    Convert a single JSONL split and return the number of examples written.
    """

    original = load_jsonl(input_path)

    converted = [
        convert_example(example)
        for example in original
    ]

    validate_examples(original, converted)
    write_jsonl(output_path, converted)

    return len(converted)


# ---------------------------------------------------------------------------
# Main program.
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== CONVERTING JARVIS V2 DATA TO MLX FORMAT ===")
    print()

    print(f"Training source:   {INPUT_FILE}")
    print(f"Validation source: {VALID_FILE}")
    print(f"MLX output:        {OUTPUT_DIR}")
    print()

    train_count = convert_split(
        INPUT_FILE,
        OUTPUT_TRAIN,
    )

    valid_count = convert_split(
        VALID_FILE,
        OUTPUT_VALID,
    )

    print("=== CONVERSION COMPLETE ===")
    print()
    print(f"Training examples:   {train_count}")
    print(f"Validation examples: {valid_count}")
    print()
    print(f"Training output:   {OUTPUT_TRAIN}")
    print(f"Validation output: {OUTPUT_VALID}")
    print()
    print("The V2 available_tools metadata was used when present.")
    print("Tool schemas were generated from the live JARVIS tool registry.")
    print()
    print("Next step:")
    print("    python model/training_data/validate_mlx_data.py")


if __name__ == "__main__":
    main()