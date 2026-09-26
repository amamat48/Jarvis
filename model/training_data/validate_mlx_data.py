import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "model" / "training_data" / "mlx"

EXPECTED_TOOLS = {
    "calculate",
    "read_file",
    "list_files",
    "run_python_file",
    "search_files",
    "remember_memory",
    "recall_memory",
    "debug_python_file",
}


def main():
    total = 0
    tool_calls = 0
    tool_messages = 0
    errors = []

    for filename in (
        "train.jsonl",
        "valid.jsonl",
        "test.jsonl",
    ):
        path = DATA_DIR / filename

        for line_number, line in enumerate(
            path.read_text().splitlines(),
            start=1
        ):
            if not line.strip():
                continue

            total += 1

            try:
                example = json.loads(line)
            except json.JSONDecodeError as error:
                errors.append(
                    f"{filename}:{line_number}: {error}"
                )
                continue

            tools = example.get("tools", [])

            for tool in tools:
                function = tool.get("function", {})
                name = function.get("name")

                if name not in EXPECTED_TOOLS:
                    errors.append(
                        f"{filename}:{line_number}: "
                        f"unknown tool schema '{name}'"
                    )

            for message in example.get("messages", []):
                if message.get("role") == "tool":
                    tool_messages += 1

                    if not message.get("name"):
                        errors.append(
                            f"{filename}:{line_number}: "
                            "tool message missing name"
                        )

                for call in message.get("tool_calls", []):
                    tool_calls += 1

                    function = call.get("function", {})

                    if not call.get("id"):
                        errors.append(
                            f"{filename}:{line_number}: "
                            "tool call missing id"
                        )

                    if function.get("name") not in EXPECTED_TOOLS:
                        errors.append(
                            f"{filename}:{line_number}: "
                            f"unknown tool call "
                            f"'{function.get('name')}'"
                        )

                    arguments = function.get("arguments")

                    if not isinstance(arguments, str):
                        errors.append(
                            f"{filename}:{line_number}: "
                            "tool-call arguments are not JSON strings"
                        )
                    else:
                        try:
                            json.loads(arguments)
                        except json.JSONDecodeError:
                            errors.append(
                                f"{filename}:{line_number}: "
                                "tool-call arguments contain invalid JSON"
                            )

    print("=== MLX DATASET VALIDATION ===")
    print(f"Examples:       {total}")
    print(f"Tool calls:     {tool_calls}")
    print(f"Tool messages:  {tool_messages}")

    if errors:
        print()
        print("=== ERRORS ===")

        for error in errors:
            print(error)

        print()
        print(f"Total errors: {len(errors)}")
    else:
        print()
        print("No MLX format errors found.")


if __name__ == "__main__":
    main()