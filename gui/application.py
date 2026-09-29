"""Application boundary between the GUI and JARVIS task/runtime services."""
from __future__ import annotations

import re
import time
from typing import Any
from uuid import uuid4

from tasks.manager import TaskExecutionHooks, TaskExecutionRequest, TaskManager
from tasks.models import TaskCommand, TaskState
from tools.security import SecurityGate


def _runtime_executor(request: TaskExecutionRequest, hooks: TaskExecutionHooks) -> str:
    # Keep heavyweight/local model imports out of the GUI's demo startup path.
    from brain.orchestrator import execute_task_turn
    return execute_task_turn(request, hooks)


def _demo_executor(request: TaskExecutionRequest, hooks: TaskExecutionHooks) -> str:
    """Deterministic demo using the same task and Security Gate boundaries."""
    text = request.user_message.casefold()
    if "demo failure" in text or "simulate failure" in text:
        raise RuntimeError("Demonstration failure requested.")
    if request.approved_operation is not None:
        operation = request.approved_operation
        hooks.tool_started(operation.tool_name)
        hooks.execute_authorized(operation)
        hooks.tool_finished(operation.tool_name)
        hooks.update_step("demo-approved", "Complete the approved operation", "completed")
        request.approved_operation = None
        return "The approval flow completed. The configured search provider is unavailable, so no web result was retrieved."
    if request.rejected_tool_result is not None:
        request.rejected_tool_result = None
        return "The requested operation was rejected; no network request was made."
    answered = bool(request.messages and request.messages[-1].get("role") == "user" and request.messages[-1].get("content") != request.user_message)
    if ("demo input" in text or "wait for input" in text) and not answered:
        hooks.request_user_input("Which project file should the demonstration inspect?")

    if "demo approval" in text or "show approval" in text:
        hooks.request_approval("web_search", {"query": "JARVIS demo approval flow"}, f"demo-{request.task_id}")

    # A few bounded, repeatable steps make pause/resume/cancel and side questions visible.
    for index, (progress, activity) in enumerate((
        (10, "Scanning project inventory"),
        (30, "Inspecting the requested topic"),
        (55, "Checking a safe local calculation"),
        (80, "Summarizing findings"),
    ), start=1):
        hooks.checkpoint()
        hooks.update_activity(activity)
        hooks.update_progress(progress)
        hooks.update_step(f"demo-{index}", activity, "running")
        if progress == 10:
            context = hooks.task_context()
            operation, decision = hooks.security_gate.propose(context, "list_files", {})
            if not decision.allowed or decision.requires_approval:
                hooks.security_gate.reject(operation)
                return "The demo file inventory was denied by policy."
            authorized = hooks.security_gate.authorize(context, operation)
            hooks.tool_started("list_files")
            inventory = hooks.execute_authorized(authorized)
            hooks.tool_finished("list_files")
            if not isinstance(inventory, str):
                return "The demo file inventory returned an invalid result."
        if progress == 55:
            context = hooks.task_context()
            operation, decision = hooks.security_gate.propose(context, "calculate", {"expression": "2 + 2"})
            if not decision.allowed or decision.requires_approval:
                hooks.security_gate.reject(operation)
                return "The demo calculation was denied by policy."
            authorized = hooks.security_gate.authorize(context, operation)
            hooks.tool_started("calculate")
            tool_result = hooks.execute_authorized(authorized)
            hooks.tool_finished("calculate")
            if tool_result != "4":
                return f"The demo calculation did not complete: {tool_result}"
        # Short intervals keep the demo responsive to safe-boundary controls.
        for _ in range(3):
            time.sleep(0.08)
            hooks.checkpoint()
    if "fail at end" in text:
        raise RuntimeError("Demonstration failure after simulated work.")
    return "Demo analysis completed. It exercised task progress, a Security Gate-authorized calculator call, and task-local status without loading a model."


class JarvisApplication:
    """GUI-facing application API; callers never access model/tool internals."""

    def __init__(self, max_concurrent_tasks: int = 2):
        self._max_concurrent_tasks = max_concurrent_tasks
        self._managers: dict[str, TaskManager] = {}
        self._managers["demo"] = self._new_manager("demo")

    def _new_manager(self, mode: str) -> TaskManager:
        from brain.prompt import SYSTEM_PROMPT
        executor = _demo_executor if mode == "demo" else _runtime_executor
        return TaskManager(
            executor=executor,
            system_prompt=SYSTEM_PROMPT,
            security_gate=SecurityGate(),
            max_concurrent_tasks=self._max_concurrent_tasks,
        )

    def manager(self, mode: str) -> TaskManager:
        if mode not in {"demo", "real"}:
            raise ValueError("Mode must be 'demo' or 'real'.")
        if mode not in self._managers:
            self._managers[mode] = self._new_manager(mode)
        return self._managers[mode]

    def snapshot(self, mode: str) -> dict[str, Any]:
        return _asdict(self.manager(mode).get_view_snapshot())

    def default_conversation(self, mode: str) -> dict[str, str]:
        return {"conversation_id": self.manager(mode).default_conversation_id}

    def events(self, mode: str, after_sequence: int, epoch: str | None = None) -> dict[str, Any]:
        manager = self.manager(mode)
        subscription = manager.subscribe_views(after_sequence, epoch)
        events = []
        try:
            first = subscription.get(timeout=0.05)
            if first is not None:
                events.append(_asdict(first))
            while len(events) < 100:
                event = subscription.get(timeout=0)
                if event is None:
                    break
                events.append(_asdict(event))
        finally:
            subscription.close()
        snapshot = manager.get_view_snapshot()
        return {"manager_epoch": snapshot.manager_epoch, "last_sequence": snapshot.last_sequence, "events": events}

    def submit(self, mode: str, text: str, conversation_id: str | None = None, priority: str = "normal") -> dict[str, Any]:
        manager = self.manager(mode)
        if not isinstance(text, str) or not text.strip():
            return {"accepted": False, "message": "Enter a message first."}
        stripped = text.strip()

        command_result = self._task_control_command(manager, stripped)
        if command_result is not None:
            return command_result
        if stripped.casefold() in {"what are you doing?", "what are you doing", "status"}:
            return {"accepted": True, "message": self.current_status(mode), "status_answer": True}

        conversation_id = conversation_id or manager.default_conversation_id
        conversation_id, side_task = self._separate_if_busy(manager, conversation_id)
        result = manager.send_command(TaskCommand(
            command_id=uuid4().hex,
            command_type="submit_message",
            conversation_id=conversation_id,
            payload={"text": stripped, "priority": priority},
        ))
        return {
            "accepted": result.accepted,
            "task_id": result.task_id,
            "conversation_id": conversation_id,
            "message": result.message,
            "side_task": side_task,
        }

    @staticmethod
    def _separate_if_busy(manager: TaskManager, conversation_id: str) -> tuple[str, bool]:
        busy_states = {
            TaskState.QUEUED, TaskState.RUNNING,
            TaskState.WAITING_FOR_USER_INPUT, TaskState.WAITING_FOR_APPROVAL,
            TaskState.PAUSED,
        }
        if any(task.conversation_id == conversation_id and task.state in busy_states for task in manager.get_snapshot().tasks):
            return manager.create_conversation(), True
        return conversation_id, False

    def _task_control_command(self, manager: TaskManager, text: str) -> dict[str, Any] | None:
        match = re.fullmatch(r"\s*(pause|resume|cancel)\s+(?:task\s+)?(.+?)\s*[.!]?\s*", text, re.I)
        if not match:
            return None
        verb, target = match.groups()
        tasks = manager.get_snapshot().tasks
        target_folded = re.sub(r"^(the|my)\s+", "", target.casefold())
        ordinal = re.fullmatch(r"(?:task\s+)?(\d+)", target_folded)
        if ordinal:
            ordered = sorted(tasks, key=lambda item: item.created_at)
            index = int(ordinal.group(1)) - 1
            matches = [ordered[index]] if 0 <= index < len(ordered) else []
        else:
            matches = [task for task in tasks if task.task_id.startswith(target) or task.title.casefold() == target_folded]
        if not matches:
            matches = [task for task in tasks if target_folded in task.title.casefold()]
        if len(matches) != 1:
            return {"accepted": False, "message": "I could not identify one matching task. Select its task card or provide its short ID."}
        task = matches[0]
        result = {"pause": manager.pause_task, "resume": manager.resume_task, "cancel": manager.cancel_task}[verb.casefold()](task.task_id)
        return {"accepted": result.accepted, "task_id": task.task_id, "message": result.message or f"{verb.title()} requested for {task.title}."}

    def current_status(self, mode: str, task_id: str | None = None) -> str:
        snapshot = self.manager(mode).get_snapshot()
        task = next((item for item in snapshot.tasks if item.task_id == task_id), None)
        if task is None and snapshot.focused_task_id:
            task = next((item for item in snapshot.tasks if item.task_id == snapshot.focused_task_id), None)
        if task is None and snapshot.active_task_id:
            task = next((item for item in snapshot.tasks if item.task_id == snapshot.active_task_id), None)
        if task is None:
            return "There are no active JARVIS tasks right now."
        return f"{task.title}: {task.current_activity}. Status: {task.state.value.replace('_', ' ')}."

    def control(self, mode: str, task_id: str, action: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        manager = self.manager(mode)
        payload = payload or {}
        task = manager.get_task(task_id)
        if task is None:
            return {"accepted": False, "message": "Task not found."}
        if action == "focus":
            result = manager.focus_task(task_id)
        elif action == "pause":
            result = manager.pause_task(task_id)
        elif action == "resume":
            result = manager.resume_task(task_id)
        elif action == "cancel":
            result = manager.cancel_task(task_id)
        elif action == "answer":
            result = manager.answer_question(task_id, payload.get("question_id", ""), payload.get("answer", ""))
        elif action in {"approve", "reject"}:
            approval = task.pending_approval
            if approval is None:
                return {"accepted": False, "message": "No pending approval."}
            fn = manager.approve_action if action == "approve" else manager.reject_action
            result = fn(task_id, payload.get("approval_id", approval.approval_id), task.revision)
        else:
            return {"accepted": False, "message": "Unknown task action."}
        return {"accepted": result.accepted, "task_id": task_id, "message": result.message}

    def close(self) -> None:
        for manager in self._managers.values():
            manager.close(wait=True)

    # MathNotebook methods
    def _get_math_notebook(self, notebook_id: str):
        """Get or create a MathNotebook by ID."""
        if not hasattr(self, "_math_notebooks"):
            self._math_notebooks = {}
            self._math_engine = None
        
        if notebook_id not in self._math_notebooks:
            from mathnotebook.engine import MathEngine
            from mathnotebook.notebook import Notebook
            if self._math_engine is None:
                self._math_engine = MathEngine()
            self._math_notebooks[notebook_id] = Notebook(self._math_engine)
        return self._math_notebooks[notebook_id]

    def math_notebook_new(self, notebook_id: str | None = None) -> dict[str, Any]:
        """Create a new MathNotebook."""
        if notebook_id is None:
            notebook_id = uuid4().hex[:8]
        nb = self._get_math_notebook(notebook_id)
        return {"notebook_id": notebook_id, "cells": []}

    def math_notebook_get(self, notebook_id: str) -> dict[str, Any]:
        """Get a MathNotebook's state."""
        nb = self._get_math_notebook(notebook_id)
        cells = []
        for cid in nb.cell_order:
            cell = nb.cells[cid]
            cells.append({
                "id": cell.id,
                "cell_type": cell.cell_type.value,
                "source": cell.source,
                "status": cell.status.value,
                "execution_count": cell.execution_count,
                "outputs": [{"output_type": o.output_type, "data": o.data} for o in cell.outputs],
            })
        return {"notebook_id": notebook_id, "cells": cells}

    def math_notebook_add_cell(self, notebook_id: str, cell_type: str, source: str, index: int | None = None) -> dict[str, Any]:
        """Add a cell to a MathNotebook."""
        from mathnotebook.notebook import Cell, CellType
        nb = self._get_math_notebook(notebook_id)
        ct = CellType.CODE if cell_type == "code" else CellType.MARKDOWN
        cell = Cell.new_code(source) if ct == CellType.CODE else Cell.new_markdown(source)
        nb.add_cell(cell, index)
        return {"cell_id": cell.id, "status": "added"}

    def math_notebook_eval_cell(self, notebook_id: str, cell_id: str) -> dict[str, Any]:
        """Evaluate a cell in a MathNotebook."""
        nb = self._get_math_notebook(notebook_id)
        cell = nb.evaluate_cell(cell_id)
        return {
            "cell_id": cell.id,
            "status": cell.status.value,
            "execution_count": cell.execution_count,
            "outputs": [{"output_type": o.output_type, "data": o.data} for o in cell.outputs],
        }

    def math_notebook_eval_all(self, notebook_id: str) -> dict[str, Any]:
        """Evaluate all cells in a MathNotebook."""
        nb = self._get_math_notebook(notebook_id)
        results = nb.evaluate_all()
        return {
            "cells": [
                {
                    "cell_id": c.id,
                    "status": c.status.value,
                    "execution_count": c.execution_count,
                    "outputs": [{"output_type": o.output_type, "data": o.data} for o in c.outputs],
                }
                for c in results
            ]
        }

    def math_notebook_delete_cell(self, notebook_id: str, cell_id: str) -> dict[str, Any]:
        """Delete a cell from a MathNotebook."""
        nb = self._get_math_notebook(notebook_id)
        nb.remove_cell(cell_id)
        return {"status": "deleted"}

    def math_notebook_save(self, notebook_id: str, path: str) -> dict[str, Any]:
        """Save a MathNotebook to disk."""
        nb = self._get_math_notebook(notebook_id)
        nb.save(path)
        return {"status": "saved", "path": path}

    def math_notebook_load(self, notebook_id: str, path: str) -> dict[str, Any]:
        """Load a MathNotebook from disk."""
        from mathnotebook.notebook import Notebook
        if self._math_engine is None:
            from mathnotebook.engine import MathEngine
            self._math_engine = MathEngine()
        nb = Notebook.load(path, self._math_engine)
        self._math_notebooks[notebook_id] = nb
        return {"status": "loaded", "notebook_id": notebook_id}


def _asdict(value):
    from dataclasses import asdict, is_dataclass
    if hasattr(value, "manager_epoch") and hasattr(value, "current_sequence"):
        return {"resync_required": True, "manager_epoch": value.manager_epoch, "current_sequence": value.current_sequence}
    if is_dataclass(value):
        return asdict(value)
    return value
