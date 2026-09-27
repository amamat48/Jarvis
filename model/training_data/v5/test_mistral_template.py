import json
from pathlib import Path

from transformers import MistralCommonBackend


DATASET = Path(__file__).resolve().parent / "train.jsonl"
MODEL_NAME = "mistralai/Ministral-3-8B-Reasoning-2512"
MAX_LENGTH = 1024


rows = [
    json.loads(line)
    for line in DATASET.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

print(f"Examples: {len(rows)}")
print("Loading MistralCommonBackend...")

backend = MistralCommonBackend.from_pretrained(MODEL_NAME)

print("Backend loaded successfully.")
print("Testing chat templates...")

failures = []
total_tokens = 0
truncated = 0

for i, example in enumerate(rows):
    try:
        messages = example["messages"]
        final_role = messages[-1].get("role")

        result = backend.apply_chat_template(
            messages,
            tools=example.get("tools", []),
            add_generation_prompt=False,
            continue_final_message=(final_role == "assistant"),
            tokenize=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
            return_dict=True,
        )

        if "input_ids" not in result:
            raise ValueError("No input_ids returned")

        token_count = result["input_ids"].shape[-1]
        total_tokens += token_count

        if token_count >= MAX_LENGTH:
            truncated += 1

    except Exception as exc:
        failures.append(
            {
                "example": i,
                "error": type(exc).__name__,
                "message": str(exc),
            }
        )


print()
print("========== RESULTS ==========")
print(f"Examples tested: {len(rows)}")
print(f"Failures:        {len(failures)}")
print(f"Total tokens:    {total_tokens}")
print(f"Average tokens:  {total_tokens / len(rows):.2f}")
print(f"At max length:   {truncated}")

if failures:
    print()
    print("First failures:")
    for failure in failures[:10]:
        print(failure)

    raise SystemExit(1)

print()
print("PASS: All V5 examples are compatible with the Mistral chat template.")