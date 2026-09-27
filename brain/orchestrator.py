from brain.router import chat
from tasks.manager import TaskExecutionHooks, TaskExecutionRequest, TaskSuspended
from tools.message_serialization import serialize_assistant_message
from tools.registry import select_tools
from tools.security import ApprovalRequired, AuthorizationError


def execute_task_turn(
    request: TaskExecutionRequest,
    hooks: TaskExecutionHooks,
) -> str:
    """Run the existing model/tool loop for one task-owned conversation turn."""
    messages = request.messages
    available_tools = select_tools(request.user_message)

    def next_assistant_message():
        hooks.checkpoint()
        hooks.update_activity("Generating a response")
        response = chat(messages, tools=list(available_tools.values()))
        assistant = serialize_assistant_message(
            response.message,
            message_index=len(messages),
            seen_call_ids=request.seen_call_ids,
        )
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
        hooks.update_activity(f"Checking {tool_name}")
        try:
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
        assistant_message = next_assistant_message()

    while assistant_message.get("tool_calls"):
        calls = assistant_message["tool_calls"]
        for index, call in enumerate(calls):
            process_tool_call(call, calls[index + 1:])

        # The full tool-call batch is paired before honoring a pause request.
        hooks.checkpoint()
        hooks.update_activity("Incorporating tool results")
        response = chat(messages, tools=list(available_tools.values()))
        assistant_message = serialize_assistant_message(
            response.message,
            message_index=len(messages),
            seen_call_ids=request.seen_call_ids,
        )
        messages.append(assistant_message)
        hooks.checkpoint(honor_pause=False)

    return assistant_message.get("content") or ""
