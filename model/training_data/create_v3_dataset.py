import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

V2_TRAIN = ROOT / "model" / "training_data" / "v2" / "mlx" / "train.jsonl"
V2_VALID = ROOT / "model" / "training_data" / "v2" / "mlx" / "valid.jsonl"

V3_DIR = ROOT / "model" / "training_data" / "v3" / "mlx"
V3_TRAIN = V3_DIR / "train.jsonl"
V3_VALID = V3_DIR / "valid.jsonl"


def tool_call(name, arguments):
    return {
        "function": {
            "name": name,
            "arguments": json.dumps(
                arguments,
                separators=(",", ":"),
            ),
        }
    }


def make_example(template, messages):
    example = copy.deepcopy(template)
    example["messages"] = messages

    # Avoid accidentally inheriting an example-specific ID.
    if "id" in example:
        example["id"] = None

    return example


def assistant_tool(name, arguments):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [tool_call(name, arguments)],
    }


def user(text):
    return {
        "role": "user",
        "content": text,
    }


def assistant(text):
    return {
        "role": "assistant",
        "content": text,
    }


def tool(content):
    return {
        "role": "tool",
        "content": content,
    }


def build_examples(template):
    examples = []

    # ============================================================
    # MEMORY BOUNDARIES
    # ============================================================

    # Incomplete memory request -> clarify
    examples.append(make_example(template, [
        user("Remember that."),
        assistant("What would you like me to remember?"),
    ]))

    # Complete memory request -> store
    examples.append(make_example(template, [
        user("Remember that my favorite editor is VS Code."),
        assistant_tool(
            "remember_memory",
            {
                "key": "favorite_editor",
                "value": "VS Code",
            },
        ),
        tool("Stored memory: favorite_editor = VS Code"),
        assistant("I'll remember that your favorite editor is VS Code."),
    ]))

    # Memory question -> recall, not store
    examples.append(make_example(template, [
        user("What memories do you have?"),
        assistant_tool("recall_memory", {}),
        tool('{"favorite_language":"C++","explanation_preference":"concise"}'),
        assistant(
            "I remember that you prefer C++ and concise explanations."
        ),
    ]))

    # Incomplete memory request -> clarify
    examples.append(make_example(template, [
        user("Remember my favorite color."),
        assistant("What is your favorite color?"),
    ]))

    # Memory question -> recall
    examples.append(make_example(template, [
        user("What's my favorite programming language?"),
        assistant_tool("recall_memory", {}),
        tool('{"favorite_language":"C++"}'),
        assistant("Your favorite programming language is C++."),
    ]))

    # Complete preference -> store
    examples.append(make_example(template, [
        user("Save this preference: use engineering examples."),
        assistant_tool(
            "remember_memory",
            {
                "key": "example_preference",
                "value": "engineering examples",
            },
        ),
        tool(
            "Stored memory: example_preference = engineering examples"
        ),
        assistant(
            "I'll remember that you prefer engineering examples."
        ),
    ]))

    # ============================================================
    # FILE-SELECTION BOUNDARIES
    # ============================================================

    # Missing filename -> clarify
    examples.append(make_example(template, [
        user("Read the file."),
        assistant("Which file would you like me to read?"),
    ]))

    # Specific file -> read
    examples.append(make_example(template, [
        user("Read tools/file_reader.py."),
        assistant_tool(
            "read_file",
            {
                "path": "tools/file_reader.py",
            },
        ),
        tool(
            "def read_file(path: str) -> str:\n"
            "    # file reader implementation\n"
            "    ..."
        ),
        assistant(
            "I read tools/file_reader.py. It implements the project's "
            "file-reading functionality."
        ),
    ]))

    # Project-wide file request -> list
    examples.append(make_example(template, [
        user("What files are in the project?"),
        assistant_tool("list_files", {}),
        tool(
            "brain/\n"
            "tools/\n"
            "model/\n"
            "main.py"
        ),
        assistant(
            "The project contains the brain, tools, and model directories, "
            "along with main.py."
        ),
    ]))

    # Missing filename -> clarify
    examples.append(make_example(template, [
        user("Explain this file."),
        assistant("Which file would you like me to explain?"),
    ]))

    # ============================================================
    # CALCULATION COMPLETENESS
    # ============================================================

    # Incomplete expression -> clarify
    examples.append(make_example(template, [
        user("What is 3973 divided by"),
        assistant(
            "What number would you like me to divide 3973 by?"
        ),
    ]))

    # Complete expression -> calculate
    examples.append(make_example(template, [
        user("What is 3973 divided by 17?"),
        assistant_tool(
            "calculate",
            {
                "expression": "3973 / 17",
            },
        ),
        tool("233.7058823529412"),
        assistant("3973 divided by 17 is approximately 233.706."),
    ]))

    # ============================================================
    # DEBUGGING BOUNDARIES
    # ============================================================

    # Missing filename -> clarify
    examples.append(make_example(template, [
        user("Debug this."),
        assistant("Which Python file would you like me to debug?"),
    ]))

    # Specific Python file -> debugger
    examples.append(make_example(template, [
        user("Debug test.py."),
        assistant_tool(
            "debug_python_file",
            {
                "path": "test.py",
            },
        ),
        tool(
            "No runtime error detected. "
            "The script completed successfully."
        ),
        assistant(
            "I debugged test.py. It completed successfully without a "
            "runtime error."
        ),
    ]))

    return examples


def main():
    if not V2_TRAIN.exists():
        raise FileNotFoundError(f"Missing V2 training file: {V2_TRAIN}")

    with V2_TRAIN.open() as f:
        original = [json.loads(line) for line in f if line.strip()]

    if len(original) != 137:
        raise ValueError(
            f"Expected 137 V2 training examples, found {len(original)}"
        )

    with V2_VALID.open() as f:
        validation = [json.loads(line) for line in f if line.strip()]

    new_examples = build_examples(original[0])

    # Prevent accidental duplicate examples.
    existing = {
        json.dumps(example["messages"], sort_keys=True)
        for example in original
    }

    unique_new = []
    for example in new_examples:
        key = json.dumps(example["messages"], sort_keys=True)

        if key not in existing:
            unique_new.append(example)
            existing.add(key)

    combined = original + unique_new

    V3_DIR.mkdir(parents=True, exist_ok=True)

    with V3_TRAIN.open("w") as f:
        for example in combined:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    with V3_VALID.open("w") as f:
        for example in validation:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print("=== JARVIS V3 DATASET CREATED ===")
    print(f"V2 training examples: {len(original)}")
    print(f"New targeted examples: {len(unique_new)}")
    print(f"V3 training examples: {len(combined)}")
    print(f"Validation examples: {len(validation)}")
    print()
    print(f"Train: {V3_TRAIN}")
    print(f"Valid: {V3_VALID}")


if __name__ == "__main__":
    main()
