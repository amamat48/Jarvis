import json
from pathlib import Path

from mlx_lm import load

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "model" / "training_data" / "mlx"
MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"


def main():
    print("Loading tokenizer...")

    _, tokenizer = load(MODEL)

    lengths = []

    for filename in ("train.jsonl", "valid.jsonl", "test.jsonl"):
        path = DATA_DIR / filename

        for line_number, line in enumerate(
            path.read_text().splitlines(),
            start=1,
        ):
            if not line.strip():
                continue

            example = json.loads(line)

            tokens = tokenizer.apply_chat_template(
                example["messages"],
                tools=example.get("tools"),
                tokenize=True,
                add_generation_prompt=False,
            )

            length = len(tokens)

            lengths.append(
                (length, filename, line_number)
            )

    lengths.sort(reverse=True)

    values = [item[0] for item in lengths]

    print("\n=== TOKEN LENGTH ANALYSIS ===")
    print(f"Examples: {len(values)}")
    print(f"Shortest: {min(values)}")
    print(f"Longest:  {max(values)}")
    print(f"Average:  {sum(values) / len(values):.1f}")

    for limit in (512, 640, 768, 1024):
        over = sum(length > limit for length in values)

        print(
            f"\nOver {limit} tokens: "
            f"{over}/{len(values)} "
            f"({over / len(values) * 100:.1f}%)"
        )

    print("\n=== LONGEST EXAMPLES ===")

    for length, filename, line_number in lengths[:10]:
        print(
            f"{length:4d} tokens | "
            f"{filename}:{line_number}"
        )


if __name__ == "__main__":
    main()