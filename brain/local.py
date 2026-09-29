import os
import json
import re
import ollama
from types import SimpleNamespace


MODEL = os.getenv("JARVIS_LOCAL_MODEL", "llama3.2:3b")


_QWEN_TOOL_CALL_PATTERN = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```",
    re.DOTALL | re.IGNORECASE,
)


def _extract_argument_value(arg_value):
    """Extract actual value from model output that may include schema metadata."""
    if isinstance(arg_value, dict):
        # Handle case where model outputs schema with value field
        if "value" in arg_value and len(arg_value) <= 3:
            return arg_value["value"]
        # Handle nested schema structures
        for key, val in arg_value.items():
            if isinstance(val, dict) and "value" in val:
                arg_value[key] = val["value"]
    return arg_value


def _normalize_arguments(arguments):
    """Normalize arguments by extracting values from schema-like structures."""
    if not isinstance(arguments, dict):
        return arguments
    normalized = {}
    for key, value in arguments.items():
        normalized[key] = _extract_argument_value(value)
    return normalized


def _parse_qwen_tool_calls(text: str):
    """Parse Qwen-style tool calls from JSON code blocks or raw JSON in content."""
    if not text:
        return []
    calls = []

    # First try code block format
    for match in _QWEN_TOOL_CALL_PATTERN.finditer(text):
        try:
            parsed = json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict) or not parsed.get("name"):
            continue
        arguments = parsed.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                continue
        if not isinstance(arguments, dict):
            continue
        arguments = _normalize_arguments(arguments)
        calls.append({"name": parsed["name"], "arguments": arguments})

    if calls:
        return calls

    # Fallback: try to parse raw JSON object from content
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, dict) and parsed.get("name"):
            arguments = parsed.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    return []
            if isinstance(arguments, dict):
                arguments = _normalize_arguments(arguments)
                calls.append({"name": parsed["name"], "arguments": arguments})

    return calls


def _normalize_response(response):
    """Ensure response has tool_calls in the expected format."""
    message = response.message
    if isinstance(message, dict):
        tool_calls = message.get("tool_calls")
        content = message.get("content", "") or ""
    else:
        tool_calls = getattr(message, "tool_calls", None)
        content = getattr(message, "content", "") or ""

    if tool_calls:
        return response

    # Try to parse Qwen-style tool calls from content
    parsed_calls = _parse_qwen_tool_calls(content)
    if not parsed_calls:
        return response

    # Convert to expected format - let the serializer handle ID generation
    new_tool_calls = []
    for call in parsed_calls:
        new_tool_calls.append({
            "type": "function",
            "function": {
                "name": call["name"],
                "arguments": call["arguments"],
            },
        })

    # Create normalized response with dict message (not SimpleNamespace)
    if isinstance(message, dict):
        normalized_message = dict(message)
        normalized_message["tool_calls"] = new_tool_calls
        normalized_message["content"] = None
    else:
        normalized_message = {
            "role": getattr(message, "role", "assistant"),
            "content": None,
            "tool_calls": new_tool_calls,
        }

    return SimpleNamespace(message=normalized_message)


def chat(messages, tools=None):
    response = ollama.chat(
        model=MODEL,
        messages=messages,
        tools=tools or []
    )

    return _normalize_response(response)