import json
from pathlib import Path

DATASET = Path(__file__).parent / "train.jsonl"

with DATASET.open("r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f if line.strip()]

print("=" * 70)
print("V5 EDGE-CASE INSPECTION")
print("=" * 70)

for i, ex in enumerate(data):
    messages = ex.get("messages", [])

    if not messages:
        continue

    last = messages[-1]

    # Assistant tool call is final
    if last.get("role") == "assistant" and last.get("tool_calls"):
        print(f"\n{'=' * 70}")
        print(f"EXAMPLE {i} — ASSISTANT TOOL CALL IS FINAL")
        print(f"{'=' * 70}")

        for j, m in enumerate(messages):
            print(f"\n[{j}] ROLE: {m.get('role')}")

            if m.get("role") == "assistant" and m.get("tool_calls"):
                for call in m["tool_calls"]:
                    fn = call.get("function", {})
                    print(f"TOOL: {fn.get('name')}")
                    print(f"ARGS: {fn.get('arguments')}")

            elif m.get("role") == "tool":
                print(f"RESULT: {m.get('content')}")

            else:
                content = m.get("content", "")
                print(f"CONTENT: {content}")

    # Tool result is final
    elif last.get("role") == "tool":
        print(f"\n{'=' * 70}")
        print(f"EXAMPLE {i} — TOOL RESULT IS FINAL")
        print(f"{'=' * 70}")

        for j, m in enumerate(messages):
            print(f"\n[{j}] ROLE: {m.get('role')}")

            if m.get("role") == "assistant" and m.get("tool_calls"):
                for call in m["tool_calls"]:
                    fn = call.get("function", {})
                    print(f"TOOL: {fn.get('name')}")
                    print(f"ARGS: {fn.get('arguments')}")

            elif m.get("role") == "tool":
                print(f"RESULT: {m.get('content')}")

            else:
                print(f"CONTENT: {m.get('content', '')}")