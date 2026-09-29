import re

from brain.router import chat
from tasks.manager import TaskExecutionHooks, TaskExecutionRequest, TaskSuspended
from tools.message_serialization import serialize_assistant_message
from tools.registry import select_tools
from tools.schemas import schemas_for
from tools.security import ApprovalRequired, AuthorizationError


_TOOL_FAILURE_PREFIXES = (
    "Tool '",
    "Execution unavailable:",
    "Execution denied",
    "Authorization denied",
    "Access denied:",
    "File not found:",
    "Error reading file:",
    "Web search unavailable:",
    "Web search failed:",
    "Web search returned no usable results.",
    "Error:",
    "Error reading memory:",
)
_ACTION_CLAIM = re.compile(
    r"\b(?:i|we)\s+(?:(?:have|has|did)\s+)?"
    r"(?P<action>listed|read|searched|looked\s+up|calculated|remembered|saved|stored|"
    r"debugged|ran|executed|checked|opened|used|found)\b",
    re.IGNORECASE,
)
_ACTION_TOOLS = {
    "listed": {"list_files"},
    "read": {"read_file"},
    "searched": {"search_files", "web_search"},
    "looked up": {"search_files", "web_search", "recall_memory"},
    "calculated": {"calculate"},
    "remembered": {"remember_memory", "recall_memory"},
    "saved": {"remember_memory"},
    "stored": {"remember_memory"},
    "debugged": {"debug_python_file"},
    "ran": {"run_python_file", "debug_python_file"},
    "executed": {"run_python_file", "debug_python_file"},
    "checked": {"read_file", "list_files", "search_files", "web_search", "debug_python_file"},
    "opened": {"read_file"},
    "used": {"calculate", "read_file", "list_files", "run_python_file", "search_files", "remember_memory", "recall_memory", "debug_python_file", "web_search"},
    "found": {"read_file", "list_files", "search_files", "recall_memory", "web_search"},
}


def _tool_succeeded(result: str) -> bool:
    if not isinstance(result, str):
        return False
    return not any(
        line.strip().startswith(_TOOL_FAILURE_PREFIXES)
        for line in result.splitlines() or [result]
    )


def _has_unsupported_action_claim(content: str, tool_outcomes: list[tuple[str, bool, str]]) -> bool:
    successful_tools = {name for name, succeeded, _ in tool_outcomes if succeeded}
    return any(
        not (_ACTION_TOOLS[match.group("action").casefold()] & successful_tools)
        for match in _ACTION_CLAIM.finditer(content)
    )


def execute_task_turn(
    request: TaskExecutionRequest,
    hooks: TaskExecutionHooks,
) -> str:
    """Run the existing model/tool loop for one task-owned conversation turn."""
    messages = request.messages
    available_tools = select_tools(request.user_message, context=messages)
    step_number = 0
    tool_outcomes = []

    def next_assistant_message(require_tool=False):
        nonlocal step_number
        step_number += 1
        hooks.update_step(f"model-{step_number}", "Prepare or review the response", "running")
        hooks.update_progress(min(90, max(5, step_number * 10)))
        hooks.checkpoint()
        hooks.update_activity("Generating a response")
        tool_schemas = schemas_for(available_tools)
        response = chat(messages, tools=tool_schemas)
        assistant = serialize_assistant_message(response.message, len(messages), request.seen_call_ids)
        if require_tool and available_tools and not assistant.get("tool_calls"):
            correction_context = messages + [
                assistant,
                {
                    "role": "system",
                    "content": (
                        "The request requires one of the supplied tools, but your last response did not "
                        "contain a structured tool call. Do not describe an action as completed. Make the "
                        "appropriate structured tool call now, or state that no action was performed."
                    ),
                },
            ]
            response = chat(correction_context, tools=tool_schemas)
            assistant = serialize_assistant_message(response.message, len(messages), request.seen_call_ids)
            if not assistant.get("tool_calls"):
                assistant = {
                    "role": "assistant",
                    "content": "I did not complete the requested operation because no tool was executed.",
                }
        messages.append(assistant)
        # Keep a tool-call message paired with its eventual result.
        hooks.checkpoint(honor_pause=False)
        return assistant

    def process_tool_call(call, remaining_calls=()):
        hooks.checkpoint(honor_pause=False)
        function = call.get("function", {})
        tool_name = function.get("name")
        arguments = function.get("arguments", {})
        call_id = call.get("id", "")
        nonlocal step_number
        step_number += 1
        step_id = f"tool-{step_number}"
        hooks.update_step(step_id, f"Use {tool_name}", "running")
        hooks.update_progress(min(90, max(10, step_number * 15)))
        hooks.update_activity(f"Checking {tool_name}")
        try:
            if tool_name not in available_tools:
                result = f"Tool '{tool_name}' is not available for this request. Use only supplied tools."
            else:
                normalized, decision = hooks.security_gate.inspect(tool_name, arguments)
                if not decision.allowed:
                    result = f"Authorization denied for '{tool_name}': {decision.reason}"
                elif decision.requires_approval:
                    request.deferred_tool_calls = list(remaining_calls)
                    hooks.request_approval(tool_name, normalized, call_id)
                    result = "Operation is awaiting approval."
                else:
                    context = hooks.task_context()
                    operation, current_decision = hooks.security_gate.propose(
                        context, tool_name, normalized
                    )
                    if not current_decision.allowed:
                        result = f"Authorization denied for '{tool_name}': {current_decision.reason}"
                    else:
                        authorized = hooks.security_gate.authorize(context, operation)
                        hooks.tool_started(tool_name)
                        result = hooks.execute_authorized(authorized)
                    hooks.tool_finished(tool_name)
        except (ApprovalRequired, TaskSuspended):
            raise
        except AuthorizationError as error:
            result = f"Authorization denied for '{tool_name}': {error}"
        except Exception as error:
            result = f"Tool '{tool_name}' failed: {error}"

        succeeded = _tool_succeeded(result)
        tool_outcomes.append((tool_name, succeeded, result))
        hooks.update_step(step_id, f"Use {tool_name}", "completed" if succeeded else "failed")
        messages.append({
            "role": "tool", "tool_name": tool_name,
            "tool_call_id": call_id, "content": result,
        })

    if request.approved_operation is not None:
        authorized = request.approved_operation
        call_id = request.approved_call_id or ""
        hooks.tool_started(authorized.tool_name)
        result = hooks.execute_authorized(authorized)
        hooks.tool_finished(authorized.tool_name)
        succeeded = _tool_succeeded(result)
        tool_outcomes.append((authorized.tool_name, succeeded, result))
        hooks.update_step(
            "approved-tool",
            f"Use {authorized.tool_name}",
            "completed" if succeeded else "failed",
        )
        messages.append({
            "role": "tool", "tool_name": authorized.tool_name,
            "tool_call_id": call_id, "content": result,
        })
        request.approved_operation = None
        request.approved_call_id = None
        deferred = request.deferred_tool_calls
        request.deferred_tool_calls = []
        for index, call in enumerate(deferred):
            process_tool_call(call, deferred[index + 1:])
        assistant_message = next_assistant_message()
    elif request.rejected_tool_result is not None:
        tool_name, call_id, result = request.rejected_tool_result
        tool_outcomes.append((tool_name, False, result))
        hooks.update_step(
            f"rejected-tool-{len(messages)}",
            f"Use {tool_name}",
            "failed",
        )
        messages.append({
            "role": "tool", "tool_name": tool_name,
            "tool_call_id": call_id, "content": result,
        })
        request.rejected_tool_result = None
        deferred = request.deferred_tool_calls
        request.deferred_tool_calls = []
        for index, call in enumerate(deferred):
            process_tool_call(call, deferred[index + 1:])
        assistant_message = next_assistant_message()
    else:
        assistant_message = next_assistant_message(require_tool=True)

    while assistant_message.get("tool_calls"):
        calls = assistant_message["tool_calls"]
        for index, call in enumerate(calls):
            process_tool_call(call, calls[index + 1:])

        # The full tool-call batch is paired before honoring a pause request.
        hooks.checkpoint()
        hooks.update_activity("Incorporating tool results")
        response = chat(messages, tools=schemas_for(available_tools))
        assistant_message = serialize_assistant_message(
            response.message,
            message_index=len(messages),
            seen_call_ids=request.seen_call_ids,
        )
        messages.append(assistant_message)
        hooks.checkpoint(honor_pause=False)

    if tool_outcomes and not any(succeeded for _, succeeded, _ in tool_outcomes):
        failures = [result for _, succeeded, result in tool_outcomes if not succeeded]
        if any(result.startswith("Authorization denied") for result in failures):
            return "The requested operation was not completed because the resource was not accessible. " + " ".join(failures)
        return "The requested operation was not completed because its tool did not succeed. " + " ".join(failures)
    content = assistant_message.get("content") or ""
    if _has_unsupported_action_claim(content, tool_outcomes):
        return "I cannot claim that action was completed because no matching tool succeeded."
    return content
