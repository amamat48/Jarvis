import json
from pathlib import Path
from collections import Counter

DATA_FILE = Path(__file__).resolve().parent / "jarvis_behavior.jsonl"

VALID_ROLES = {"system", "user", "assistant", "tool"}

def validate():
    if not DATA_FILE.exists():
        print(f"ERROR: File not found: {DATA_FILE}")
        return

    errors = []
    warnings = []
    entries = []
    raw_lines = []

    for line_number, raw_line in enumerate(
        DATA_FILE.read_text().splitlines(), start=1
    ):
        if not raw_line.strip():
            warnings.append(f"Line {line_number}: blank line")
            continue

        raw_lines.append(raw_line)

        try:
            entry = json.loads(raw_line)
        except json.JSONDecodeError as error:
            errors.append(
                f"Line {line_number}: invalid JSON: {error}"
            )
            continue

        if not isinstance(entry, dict):
            errors.append(
                f"Line {line_number}: top-level value must be an object"
            )
            continue

        messages = entry.get("messages")

        if not isinstance(messages, list) or not messages:
            errors.append(
                f"Line {line_number}: missing or empty 'messages' list"
            )
            continue

        for message_index, message in enumerate(messages):
            if not isinstance(message, dict):
                errors.append(
                    f"Line {line_number}, message {message_index}: "
                    "message must be an object"
                )
                continue

            role = message.get("role")

            if role not in VALID_ROLES:
                errors.append(
                    f"Line {line_number}, message {message_index}: "
                    f"invalid role '{role}'"
                )

            if role in {"user", "assistant", "system"}:
                if "content" not in message and "tool_calls" not in message:
                    errors.append(
                        f"Line {line_number}, message {message_index}: "
                        "assistant/user/system message needs content "
                        "or tool_calls"
                    )

            if role == "tool" and "content" not in message:
                errors.append(
                    f"Line {line_number}, message {message_index}: "
                    "tool message is missing content"
                )

            if "tool_calls" in message:
                tool_calls = message["tool_calls"]

                if not isinstance(tool_calls, list):
                    errors.append(
                        f"Line {line_number}, message {message_index}: "
                        "tool_calls must be a list"
                    )
                    continue

                for call_index, call in enumerate(tool_calls):
                    if not isinstance(call, dict):
                        errors.append(
                            f"Line {line_number}, message {message_index}, "
                            f"tool call {call_index}: must be an object"
                        )
                        continue

                    function = call.get("function")

                    if not isinstance(function, dict):
                        errors.append(
                            f"Line {line_number}, message {message_index}, "
                            f"tool call {call_index}: missing function object"
                        )
                        continue

                    if not function.get("name"):
                        errors.append(
                            f"Line {line_number}, message {message_index}, "
                            f"tool call {call_index}: missing function name"
                        )

                    arguments = function.get("arguments")

                    if arguments is None:
                        errors.append(
                            f"Line {line_number}, message {message_index}, "
                            f"tool call {call_index}: missing arguments"
                        )

                    elif not isinstance(arguments, dict):
                        errors.append(
                            f"Line {line_number}, message {message_index}, "
                            f"tool call {call_index}: arguments must be an object"
                        )

        entries.append(entry)

    duplicate_count = len(raw_lines) - len(set(raw_lines))

    role_counts = Counter()

    for entry in entries:
        for message in entry.get("messages", []):
            role_counts[message.get("role")] += 1

    print("=== JARVIS TRAINING DATA VALIDATION ===")
    print(f"Dataset: {DATA_FILE}")
    print(f"Examples: {len(entries)}")
    print(f"Messages: {sum(role_counts.values())}")
    print(f"User messages: {role_counts['user']}")
    print(f"Assistant messages: {role_counts['assistant']}")
    print(f"Tool messages: {role_counts['tool']}")
    print(f"System messages: {role_counts['system']}")
    print(f"Exact duplicate lines: {duplicate_count}")

    if errors:
        print("\n=== ERRORS ===")
        for error in errors:
            print(error)
    else:
        print("\nNo structural errors found.")

    if warnings:
        print("\n=== WARNINGS ===")
        for warning in warnings:
            print(warning)

    if duplicate_count:
        print(
            "\nWARNING: Exact duplicate examples exist. "
            "We should review them before training."
        )

    if not errors:
        print("\nDataset structure looks valid.")

if __name__ == "__main__":
    validate()