import copy


def serialize_assistant_message(message, message_index, seen_call_ids):
    """Return a plain assistant-message dict with unique tool-call IDs."""
    if isinstance(message, dict):
        serialized = copy.deepcopy(message)
    elif hasattr(message, "model_dump"):
        serialized = message.model_dump(exclude_none=True)
    elif hasattr(message, "dict"):
        serialized = message.dict(exclude_none=True)
    else:
        raise TypeError(
            "Assistant message must be a mapping or a supported model object."
        )

    if not isinstance(serialized, dict):
        raise TypeError("Serialized assistant message must be a mapping.")

    for call_index, tool_call in enumerate(serialized.get("tool_calls") or []):
        if not isinstance(tool_call, dict):
            raise TypeError("Serialized tool calls must be mappings.")

        call_id = tool_call.get("id")

        if not call_id:
            call_id = f"jarvis_tool_{message_index}_{call_index}"
            tool_call["id"] = call_id

        if not isinstance(call_id, str):
            raise ValueError("Tool-call IDs must be non-empty strings.")

        if call_id in seen_call_ids:
            raise ValueError(f"Duplicate tool-call id: {call_id}")

        seen_call_ids.add(call_id)

    return serialized
