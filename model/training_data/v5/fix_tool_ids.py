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

    # Only pair results with calls that actually precede them.
    pending_tool_ids = []
    seen_tool_ids = set()

    for message_index, message in enumerate(messages):
        if message.get("role") == "assistant":
            for call_index, call in enumerate(message.get("tool_calls", [])):
                tool_id = call.get("id")

                if not tool_id:
                    tool_id = (
                        f"jarvis_tool_{example_index}_{message_index}_{call_index}"
                    )
                    call["id"] = tool_id

                if not isinstance(tool_id, str):
                    raise ValueError(
                        f"Example {example_index + 1} has a non-string "
                        "tool-call id."
                    )

                if tool_id in seen_tool_ids:
                    raise ValueError(
                        f"Example {example_index + 1} has duplicate "
                        f"tool-call id {tool_id!r}."
                    )

                seen_tool_ids.add(tool_id)

                # Ministral expects the OpenAI-style function-call type.
                call.setdefault("type", "function")

                pending_tool_ids.append(tool_id)
                total_calls += 1

        elif message.get("role") == "tool":
            tool_call_id = message.get("tool_call_id")

            if not tool_call_id:
                if not pending_tool_ids:
                    raise ValueError(
                        f"Example {example_index + 1} has a tool result "
                        "without a preceding unmatched tool call."
                    )

                tool_call_id = pending_tool_ids[0]
                message["tool_call_id"] = tool_call_id

            if tool_call_id not in pending_tool_ids:
                raise ValueError(
                    f"Example {example_index + 1} has a tool result that "
                    f"does not match a preceding call: {tool_call_id!r}."
                )

            pending_tool_ids.remove(tool_call_id)

            total_results += 1

    if pending_tool_ids and not (
        messages
        and messages[-1].get("role") == "assistant"
        and messages[-1].get("tool_calls")
    ):
        raise ValueError(
            f"Example {example_index + 1} has tool calls without results "
            "before its final training target."
        )


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
