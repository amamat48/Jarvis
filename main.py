import queue
import threading

from brain.orchestrator import execute_task_turn
from brain.prompt import SYSTEM_PROMPT
from tasks.manager import ResyncRequired, TaskManager
from tasks.models import TaskState
from voice_input import listen
from voice_output import speak


def _read_console(input_queue):
    while True:
        try:
            line = input("You: ")
        except EOFError:
            input_queue.put(("exit", "", False))
            return

        if line.strip().lower() == "voice":
            print("Listening...")
            input_queue.put(("message", listen(), True))
            continue

        if line.strip().lower() == "exit":
            input_queue.put(("exit", "", False))
            return

        input_queue.put(("line", line, False))


def _short_id(task_id):
    return task_id[:8]


def _resolve_task_id(manager, prefix):
    prefix = prefix.strip()
    if not prefix:
        return None
    matches = [
        task.task_id
        for task in manager.get_snapshot().tasks
        if task.task_id.startswith(prefix)
    ]
    return matches[0] if len(matches) == 1 else None


def _render_event(event, voice_tasks):
    task_label = f"[{_short_id(event.task_id)}] " if event.task_id else ""
    payload = event.payload

    if event.event_type == "task_created":
        print(f"{task_label}Created: {payload.get('objective', '')}")
    elif event.event_type == "task_state_changed":
        new_state = payload.get("to_state", "")
        if new_state != TaskState.QUEUED.value:
            print(f"{task_label}State: {new_state.replace('_', ' ')}")
    elif event.event_type == "activity_updated":
        print(f"{task_label}{payload.get('current_activity', '')}")
    elif event.event_type == "tool_started":
        print(f"{task_label}Using {payload.get('tool_name', 'tool')}")
    elif event.event_type == "user_input_requested":
        print(f"{task_label}Question: {payload.get('prompt', '')}")
        print(f"Answer with: answer {_short_id(event.task_id)} {payload.get('question_id', '')} <your response>")
    elif event.event_type == "approval_requested":
        print(
            f"{task_label}Approval required: {payload.get('summary', '')} "
            f"(approval ID: {payload.get('approval_id', '')})"
        )
        print(
            f"Use approve {_short_id(event.task_id)} {payload.get('approval_id', '')} "
            f"or reject {_short_id(event.task_id)} {payload.get('approval_id', '')}."
        )
    elif event.event_type == "focus_changed":
        focused = payload.get("focused_task_id")
        print(f"Focused task: {_short_id(focused) if focused else 'none'}")
    elif event.event_type == "task_finished":
        state = payload.get("state")
        result = payload.get("result", "")
        if state == TaskState.COMPLETED.value:
            print(f"{task_label}JARVIS: {result}")
            if voice_tasks.pop(event.task_id, False):
                speak(result)
        elif state == TaskState.FAILED.value:
            print(f"{task_label}{result}")
        elif state == TaskState.CANCELLED.value:
            print(f"{task_label}Cancelled.")


def _print_tasks(manager):
    snapshot = manager.get_snapshot()
    tasks = snapshot.tasks
    if not tasks:
        print("No tasks yet.")
        return
    for task in tasks:
        focused = " (focused)" if task.task_id == snapshot.focused_task_id else ""
        print(
            f"{_short_id(task.task_id)}  {task.state.value}{focused}  "
            f"{task.objective}  — {task.current_activity}"
        )


def _report_command(result):
    if result.accepted:
        if result.task_id:
            print(f"[{_short_id(result.task_id)}] {result.message or 'Command accepted.'}")
        elif result.message:
            print(result.message)
        if result.duplicate:
            print("Duplicate command ignored.")
    else:
        print(f"Command rejected: {result.message}")


def _handle_line(line, manager, conversation_id, voice_tasks):
    stripped = line.strip()
    if not stripped:
        return conversation_id

    command, _, arguments = stripped.partition(" ")
    command = command.lower()

    if command in {"clear", "new"}:
        conversation_id = manager.create_conversation()
        print("Started a new conversation. Existing tasks continue in their original contexts.")
        return conversation_id
    if command == "help":
        print(
            "Commands: clear/new, tasks, focus <task>, background <task>, "
            "pause <task>, resume <task>, cancel <task>, "
            "answer <task> <question-id> <response>, "
            "approve <task> <approval-id>, reject <task> <approval-id>, exit, voice"
        )
        return conversation_id
    if command == "tasks":
        _print_tasks(manager)
        return conversation_id
    if command in {"focus", "background", "pause", "resume", "cancel"}:
        task_id = _resolve_task_id(manager, arguments.strip())
        if task_id is None:
            print("Task ID is missing, unknown, or ambiguous.")
            return conversation_id
        operation = {
            "focus": manager.focus_task,
            "background": manager.background_task,
            "pause": manager.pause_task,
            "resume": manager.resume_task,
            "cancel": manager.cancel_task,
        }[command]
        _report_command(operation(task_id))
        return conversation_id
    if command == "answer":
        parts = arguments.split(maxsplit=2)
        if len(parts) != 3:
            print("Usage: answer <task> <question-id> <response>")
            return conversation_id
        task_id = _resolve_task_id(manager, parts[0])
        if task_id is None:
            print("Task ID is missing, unknown, or ambiguous.")
            return conversation_id
        _report_command(manager.answer_question(task_id, parts[1], parts[2]))
        return conversation_id
    if command in {"approve", "reject"}:
        parts = arguments.split(maxsplit=1)
        if len(parts) != 2:
            print(f"Usage: {command} <task> <approval-id>")
            return conversation_id
        task_id = _resolve_task_id(manager, parts[0])
        if task_id is None:
            print("Task ID is missing, unknown, or ambiguous.")
            return conversation_id
        task = manager.get_task(task_id)
        operation = manager.approve_action if command == "approve" else manager.reject_action
        _report_command(operation(task_id, parts[1], expected_revision=task.revision))
        return conversation_id

    result = manager.submit_message(stripped, conversation_id=conversation_id)
    _report_command(result)
    return conversation_id


def main():
    manager = TaskManager(executor=execute_task_turn, system_prompt=SYSTEM_PROMPT)
    conversation_id = manager.default_conversation_id
    voice_tasks = {}
    snapshot = manager.get_snapshot()
    subscription = manager.subscribe(snapshot.last_sequence, snapshot.manager_epoch)
    input_queue = queue.Queue()
    input_thread = threading.Thread(
        target=_read_console,
        args=(input_queue,),
        name="jarvis-console-input",
        daemon=True,
    )

    print("JARVIS is online.")
    print("Type 'help' for console/task commands. Submit another message while work runs to queue it.\n")
    input_thread.start()

    try:
        while True:
            event = subscription.get(timeout=0.05)
            if isinstance(event, ResyncRequired):
                snapshot = manager.get_snapshot()
                subscription.close()
                subscription = manager.subscribe(snapshot.last_sequence, snapshot.manager_epoch)
                print("Task event history refreshed from the current snapshot.")
            elif event is not None:
                _render_event(event, voice_tasks)

            try:
                kind, line, using_voice = input_queue.get_nowait()
            except queue.Empty:
                continue

            if kind == "exit":
                active_task_id = manager.get_snapshot().active_task_id
                if active_task_id:
                    print("Stopping queued work and waiting for the active operation to reach a safe boundary...")
                break
            if kind == "message":
                print(f"You (voice): {line}")
                result = manager.submit_message(line, conversation_id=conversation_id)
                _report_command(result)
                if result.accepted and result.task_id and using_voice:
                    voice_tasks[result.task_id] = True
            else:
                conversation_id = _handle_line(line, manager, conversation_id, voice_tasks)
    finally:
        subscription.close()
        manager.close(wait=True)

    print("JARVIS is offline.")


if __name__ == "__main__":
    main()
