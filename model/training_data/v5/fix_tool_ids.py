import json
from pathlib import Path


DATASET = Path(__file__).resolve().parent / "train.jsonl"


rows = [
    json.loads(line)
    for line in DATASET.read_text(encoding="utf-8").splitlines()
    if line.strip()
]


total_calls = 0
total_results = 0


for example_index, example in enumerate(rows):
    messages = example["messages"]

    # Map tool-call IDs to the tool calls in this conversation.
    pending_tool_ids = []

    for message_index, message in enumerate(messages):
        if message.get("role") == "assistant":
            for call_index, call in enumerate(message.get("tool_calls", [])):
                tool_id = call.get("id")

                if not tool_id:
                    tool_id = (
                        f"jarvis_tool_{message_index}_{call_index}"
                    )
                    call["id"] = tool_id

                # Ministral expects the OpenAI-style function-call type.
                call.setdefault("type", "function")

                pending_tool_ids.append(tool_id)
                total_calls += 1

        elif message.get("role") == "tool":
            tool_call_id = message.get("tool_call_id")

            if not tool_call_id:
                if pending_tool_ids:
                    tool_call_id = pending_tool_ids.pop(0)
                else:
                    tool_call_id = (
                        f"jarvis_tool_result_{message_index}"
                    )

                message["tool_call_id"] = tool_call_id

            total_results += 1


# Write every example as exactly one JSON object per line.
DATASET.write_text(
    "".join(
        json.dumps(
            example,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
        for example in rows
    ),
    encoding="utf-8",
)


print(f"Repaired examples: {len(rows)}")
print(f"Tool calls:        {total_calls}")
print(f"Tool results:      {total_results}")
print(f"Wrote:             {DATASET}")