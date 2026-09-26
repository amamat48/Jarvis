import json
import random
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILE = PROJECT_ROOT / "model" / "training_data" / "jarvis_behavior.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "model" / "training_data" / "splits"

SEED = 42


def get_group(example):
    """
    Group examples by their first tool call so that different
    tool-use behaviors are represented across train/valid/test.
    """
    messages = example.get("messages", [])

    for message in messages:
        tool_calls = message.get("tool_calls", [])

        if tool_calls:
            function = tool_calls[0].get("function", {})
            return function.get("name", "unknown_tool")

    return "no_tool"


def load_examples():
    examples = []

    for line_number, line in enumerate(
        SOURCE_FILE.read_text().splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            examples.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON at line {line_number}: {error}"
            ) from error

    return examples


def write_jsonl(path, examples):
    with path.open("w") as file:
        for example in examples:
            file.write(json.dumps(example, ensure_ascii=False) + "\n")


def main():
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(SOURCE_FILE)

    examples = load_examples()

    groups = defaultdict(list)

    for example in examples:
        groups[get_group(example)].append(example)

    rng = random.Random(SEED)

    train = []
    valid = []
    test = []

    for group_name, group_examples in sorted(groups.items()):
        rng.shuffle(group_examples)

        count = len(group_examples)

        # Aim for approximately 80/10/10.
        test_count = max(1, round(count * 0.10))
        valid_count = max(1, round(count * 0.10))

        if count < 5:
            # Tiny groups need most of their examples for training.
            test_count = 1
            valid_count = 1

        train_count = count - valid_count - test_count

        if train_count < 1:
            train_count = 1

        group_train = group_examples[:train_count]
        group_valid = group_examples[
            train_count:train_count + valid_count
        ]
        group_test = group_examples[
            train_count + valid_count:
        ]

        train.extend(group_train)
        valid.extend(group_valid)
        test.extend(group_test)

    rng.shuffle(train)
    rng.shuffle(valid)
    rng.shuffle(test)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    write_jsonl(OUTPUT_DIR / "train.jsonl", train)
    write_jsonl(OUTPUT_DIR / "valid.jsonl", valid)
    write_jsonl(OUTPUT_DIR / "test.jsonl", test)

    print("=== JARVIS DATASET SPLIT ===")
    print(f"Total examples: {len(examples)}")
    print(f"Training:       {len(train)}")
    print(f"Validation:     {len(valid)}")
    print(f"Test:           {len(test)}")
    print()
    print(f"Output directory: {OUTPUT_DIR}")

    print("\nGroups:")
    for group_name in sorted(groups):
        print(f"  {group_name}: {len(groups[group_name])}")


if __name__ == "__main__":
    main()