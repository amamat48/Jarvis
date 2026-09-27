import json
from pathlib import Path

DATASET = Path(__file__).parent / "train.jsonl"

FIXES = {
    16: (
        "The calculation function is implemented in `tools/calculator.py`. "
        "It evaluates the supplied expression with Python's `eval()` while "
        "disabling built-in functions, returns the result as a string, and "
        "returns an error message if evaluation raises an exception."
    ),
    50: (
        "The first search for `calculation` returned no matches. "
        "The follow-up search for `calculate` found the function in "
        "`tools/calculator.py`."
    ),
    59: (
        "I searched for both `FooBar class` and `FooBar`, and neither search "
        "returned a match. I can't identify a FooBar implementation from "
        "the available search results."
    ),
}

with DATASET.open("r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f if line.strip()]

for index, response in FIXES.items():
    messages = data[index]["messages"]

    if messages[-1].get("role") != "tool":
        raise ValueError(
            f"Example {index} does not end with a tool result; refusing to modify it."
        )

    messages.append({
        "role": "assistant",
        "content": response,
    })

with DATASET.open("w", encoding="utf-8") as f:
    for example in data:
        f.write(json.dumps(example, ensure_ascii=False) + "\n")

print("Fixed examples:", ", ".join(map(str, FIXES)))
print("Total examples:", len(data))
print("No other examples were modified.")