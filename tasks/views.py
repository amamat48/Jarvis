"""User-facing projections of TaskManager state and events.

These views intentionally omit model messages and authorization/execution objects.
They are a presentation boundary, not a general-purpose PII detector.
"""
from dataclasses import dataclass
from typing import Any

from tasks.models import (
    ResyncRequired,
    Task,
    TaskEvent,
    TaskSnapshot,
    TaskState,
)


@dataclass(frozen=True)
class TaskStepView:
    step_id: str
    title: str
    state: str


@dataclass(frozen=True)
class TaskStatisticsView:
    steps_completed: int
    tool_calls: int
    retries: int


@dataclass(frozen=True)
class TimelineEntryView:
    sequence: int
    timestamp: str
    event_type: str
    label: str


@dataclass(frozen=True)
class UserQuestionView:
    question_id: str
    prompt: str
    options: tuple[str, ...]


@dataclass(frozen=True)
class PendingApprovalView:
    approval_id: str
    summary: str


@dataclass(frozen=True)
class TaskView:
    task_id: str
    conversation_id: str
    state: str
    revision: int
    objective: str
    title: str
    description: str
    priority: str
    progress: int
    current_activity: str
    created_at: str
    started_at: str | None
    completed_at: str | None
    updated_at: str
    plan: tuple[TaskStepView, ...]
    result: str | None
    error: str | None
    statistics: TaskStatisticsView
    timeline: tuple[TimelineEntryView, ...]
    pending_question: UserQuestionView | None
    pending_approval: PendingApprovalView | None
    pause_requested: bool
    cancel_requested: bool


@dataclass(frozen=True)
class TaskSnapshotView:
    manager_epoch: str
    last_sequence: int
    focused_task_id: str | None
    active_task_id: str | None
    queued_task_ids: tuple[str, ...]
    tasks: tuple[TaskView, ...]
    active_task_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskEventView:
    manager_epoch: str
    sequence: int
    event_id: str
    timestamp: str
    event_type: str
    task_id: str | None
    task_revision: int | None
    payload: dict[str, Any]


def _safe_text(value: Any, limit: int = 512) -> str:
    if not isinstance(value, str):
        return ""
    printable = "".join(
        character
        for character in value
        if character in "\t\n" or (ord(character) >= 32 and ord(character) != 127)
    )
    return " ".join(printable.split())[:limit]


def _safe_activity(value: Any) -> str:
    if not isinstance(value, str):
        return "Working"
    fixed_labels = {
        "Queued for execution": "Queued",
        "Paused before execution": "Paused before execution",
        "Pause requested; waiting for a safe boundary": "Pause requested; waiting for the current action",
        "Cancellation requested; waiting for a safe boundary": "Cancellation requested; waiting for the current action",
        "Processing request": "Processing request",
        "Generating a response": "Preparing a response",
        "Incorporating tool results": "Reviewing operation results",
        "Paused at a safe boundary": "Paused",
        "Waiting for user input": "Waiting for your input",
        "Waiting for approval": "Waiting for your approval",
    }
    if value in fixed_labels:
        return fixed_labels[value]
    if value.startswith("Checking "):
        return "Reviewing a requested operation"
    if value.startswith("Using "):
        return "Executing an authorized operation"
    return "Working"


_EVENT_LABELS = {
    "task_created": "Task created",
    "task_queued": "Task queued",
    "task_state_changed": "Task state changed",
    "activity_updated": "Activity updated",
    "plan_updated": "Plan updated",
    "statistics_updated": "Statistics updated",
    "progress_updated": "Progress updated",
    "tool_started": "Authorized operation started",
    "tool_finished": "Authorized operation finished",
    "user_input_requested": "User input requested",
    "approval_requested": "Approval requested",
    "approval_decided": "Approval decided",
    "result_updated": "Result updated",
    "task_finished": "Task finished",
    "task_control_requested": "Task control requested",
    "focus_changed": "Focus changed",
}


def _safe_count(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _safe_state(value: Any) -> str | None:
    if isinstance(value, TaskState):
        return value.value
    if isinstance(value, str) and value in {state.value for state in TaskState}:
        return value
    return None


def _safe_step_views(steps: Any) -> tuple[TaskStepView, ...]:
    if not isinstance(steps, (list, tuple)):
        return ()
    return tuple(
        TaskStepView(
            _safe_text(getattr(step, "step_id", ""), 80),
            _safe_text(getattr(step, "title", ""), 200),
            getattr(getattr(step, "state", None), "value", "pending"),
        )
        for step in steps
    )


def project_task(task: Task) -> TaskView:
    """Project one internal task using an explicit user-facing field allowlist."""
    state = _safe_state(task.state) or TaskState.FAILED.value
    question = task.pending_question
    approval = task.pending_approval
    result = _safe_text(task.result, 8000) if state == TaskState.COMPLETED.value else None
    error = "Task failed." if state == TaskState.FAILED.value else None
    return TaskView(
        task_id=task.task_id,
        conversation_id=task.conversation_id,
        state=state,
        revision=_safe_count(task.revision),
        objective=_safe_text(task.objective, 500),
        title=_safe_text(task.title or task.objective, 120),
        description=_safe_text(task.description or task.objective, 1000),
        priority=task.priority if task.priority in {"low", "normal", "high"} else "normal",
        progress=min(100, _safe_count(task.progress)),
        current_activity=_safe_activity(task.current_activity),
        created_at=_safe_text(task.created_at, 80),
        started_at=_safe_text(task.started_at, 80) if task.started_at else None,
        completed_at=_safe_text(task.completed_at, 80) if task.completed_at else None,
        updated_at=_safe_text(task.updated_at, 80),
        plan=_safe_step_views(task.plan),
        result=result,
        error=error,
        statistics=TaskStatisticsView(
            _safe_count(task.statistics.steps_completed),
            _safe_count(task.statistics.tool_calls),
            _safe_count(task.statistics.retries),
        ),
        timeline=tuple(
            TimelineEntryView(
                _safe_count(entry.sequence),
                _safe_text(entry.timestamp, 80),
                _safe_text(entry.event_type, 80),
                _EVENT_LABELS.get(entry.event_type, "Task updated"),
            )
            for entry in task.timeline
        ),
        pending_question=(
            UserQuestionView(
                question.question_id,
                _safe_text(question.prompt, 1000),
                tuple(_safe_text(option, 160) for option in question.options),
            )
            if question is not None
            else None
        ),
        pending_approval=(
            PendingApprovalView(
                approval.approval_id,
                "Sensitive operation requires approval.",
            )
            if approval is not None
            else None
        ),
        pause_requested=bool(task.pause_requested),
        cancel_requested=bool(task.cancel_requested),
    )


def project_snapshot(snapshot: TaskSnapshot) -> TaskSnapshotView:
    return TaskSnapshotView(
        manager_epoch=snapshot.manager_epoch,
        last_sequence=snapshot.last_sequence,
        focused_task_id=snapshot.focused_task_id,
        active_task_id=snapshot.active_task_id,
        active_task_ids=tuple(snapshot.active_task_ids),
        queued_task_ids=tuple(snapshot.queued_task_ids),
        tasks=tuple(project_task(task) for task in snapshot.tasks),
    )


def _safe_event_payload(event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    if event_type == "task_created":
        return {
            "conversation_id": _safe_text(payload.get("conversation_id"), 80),
            "objective": _safe_text(payload.get("objective"), 500),
        }
    if event_type == "task_queued":
        return {"queue_position": _safe_count(payload.get("queue_position"))}
    if event_type == "task_state_changed":
        return {
            "from_state": _safe_state(payload.get("from_state")),
            "to_state": _safe_state(payload.get("to_state")),
        }
    if event_type == "activity_updated":
        return {"current_activity": _safe_activity(payload.get("current_activity"))}
    if event_type == "plan_updated":
        return {
            "steps": tuple(
                {
                    "step_id": step.step_id,
                    "title": step.title,
                    "state": step.state,
                }
                for step in _safe_step_views(payload.get("steps"))
            )
        }
    if event_type == "progress_updated":
        return {"progress": min(100, _safe_count(payload.get("progress")))}
    if event_type == "statistics_updated":
        return {
            key: _safe_count(payload[key])
            for key in ("steps_completed", "tool_calls", "retries")
            if key in payload
        }
    if event_type in {"tool_started", "tool_finished"}:
        result = {"activity": _EVENT_LABELS[event_type]}
        tool_name = payload.get("tool_name")
        if isinstance(tool_name, str) and tool_name in {"calculate", "read_file", "list_files", "search_files", "run_python_file", "debug_python_file", "remember_memory", "recall_memory", "web_search"}:
            result["tool_name"] = tool_name
        if event_type == "tool_finished":
            status = payload.get("status")
            result["status"] = (
                status
                if isinstance(status, str) and status in {"finished", "failed", "denied"}
                else "finished"
            )
        return result
    if event_type == "user_input_requested":
        options = payload.get("options", ())
        return {
            "question_id": _safe_text(payload.get("question_id"), 80),
            "prompt": _safe_text(payload.get("prompt"), 1000),
            "options": tuple(_safe_text(item, 160) for item in options if isinstance(item, str))
            if isinstance(options, (list, tuple))
            else (),
        }
    if event_type == "approval_requested":
        return {
            "approval_id": _safe_text(payload.get("approval_id"), 80),
            "summary": "Sensitive operation requires approval.",
        }
    if event_type == "approval_decided":
        decision = payload.get("decision")
        return {
            "approval_id": _safe_text(payload.get("approval_id"), 80),
            "decision": (
                decision
                if isinstance(decision, str) and decision in {"approved", "rejected"}
                else "decided"
            ),
        }
    if event_type == "result_updated":
        return {"available": True}
    if event_type == "task_finished":
        state = _safe_state(payload.get("state")) or "failed"
        result = {"state": state}
        if state == TaskState.COMPLETED.value:
            result["result"] = _safe_text(payload.get("result"), 8000)
        elif state == TaskState.FAILED.value:
            result["error"] = "Task failed."
        elif state == TaskState.CANCELLED.value:
            result["result"] = "Cancelled."
        return result
    if event_type == "task_control_requested":
        control = payload.get("control")
        return {
            "control": control if control in {"pause", "cancel"} else "update",
            "current_activity": _safe_activity(payload.get("current_activity")),
        }
    if event_type == "focus_changed":
        focused = payload.get("focused_task_id")
        return {"focused_task_id": focused if isinstance(focused, str) else None}
    return {}


def project_event(event: TaskEvent) -> TaskEventView:
    return TaskEventView(
        manager_epoch=event.manager_epoch,
        sequence=event.sequence,
        event_id=event.event_id,
        timestamp=_safe_text(event.timestamp, 80),
        event_type=event.event_type,
        task_id=event.task_id,
        task_revision=event.task_revision,
        payload=_safe_event_payload(event.event_type, event.payload),
    )


class TaskViewSubscription:
    """Subscription facade that never returns raw TaskEvent payloads."""

    def __init__(self, subscription: Any):
        self._subscription = subscription

    def get(self, timeout: float | None = None) -> TaskEventView | ResyncRequired | None:
        event = self._subscription.get(timeout)
        return project_event(event) if isinstance(event, TaskEvent) else event

    def close(self) -> None:
        self._subscription.close()
