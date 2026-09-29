"""Parsing helpers for the Mistral native tool-call formats used by V6."""
import json
import re


def parse_arguments(raw_arguments):
    if raw_arguments is None:
        return {}
    if isinstance(raw_arguments, dict):
        return raw_arguments
    if not isinstance(raw_arguments, str):
        return None
    raw_arguments = raw_arguments.strip()
    if not raw_arguments:
        return {}
    try:
        parsed = json.loads(raw_arguments)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def parse_tool_calls_bracket_format(text):
    if not text:
        return []
    pattern = re.compile(
        r"\[TOOL_CALLS\]\s*"
        r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
        r"\s*\[ARGS\]\s*"
        r"(?P<arguments>\{.*?\})",
        re.DOTALL,
    )
    calls = []
    for match in pattern.finditer(text):
        arguments = parse_arguments(match.group("arguments"))
        if arguments is not None:
            calls.append({"name": match.group("name"), "arguments": arguments})
    return calls


def parse_xml_tool_calls(text):
    if not text:
        return []
    pattern = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
    calls = []
    for match in pattern.finditer(text):
        try:
            parsed = json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict) or not parsed.get("name"):
            continue
        arguments = parsed.get("arguments", {})
        if isinstance(arguments, str):
            arguments = parse_arguments(arguments)
        if arguments is None or not isinstance(arguments, dict):
            continue
        call = {"name": parsed["name"], "arguments": arguments}
        if parsed.get("id") is not None:
            call["id"] = parsed["id"]
        calls.append(call)
    return calls


def parse_model_tool_calls(text):
    bracket_calls = parse_tool_calls_bracket_format(text)
    return bracket_calls if bracket_calls else parse_xml_tool_calls(text)
