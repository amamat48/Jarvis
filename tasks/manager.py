import copy
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import threading
import time
from typing import Any, Callable
from uuid import uuid4

from tasks.models import (
    ApprovalRequest,
    CommandResult,
    ResyncRequired,
    StepState,
    Task,
    TaskCommand,
    TaskEvent,
    TaskSnapshot,
    TaskState,
    TaskStep,
    TimelineEntry,
    UserQuestion,
)
from tools.runner import (
    DEFAULT_SECURITY_GATE,
    claim as claim_authorized_operation,
    execute_claimed,
)
from tasks.views import (
    TaskSnapshotView,
    TaskView,
    TaskViewSubscription,
    project_snapshot,
    project_task,
)
from tools.security import AuthorizedOperation, Operation, SecurityGate, TaskContext


_ALLOWED_TRANSITIONS = {
    TaskState.QUEUED: {TaskState.RUNNING, TaskState.PAUSED, TaskState.CANCELLED},
    TaskState.RUNNING: {
        TaskState.WAITING_FOR_USER_INPUT,
        TaskState.WAITING_FOR_APPROVAL,
        TaskState.PAUSED,
        TaskState.COMPLETED,
        TaskState.FAILED,
        TaskState.CANCELLED,
    },
    TaskState.WAITING_FOR_USER_INPUT: {TaskState.QUEUED, TaskState.CANCELLED},
    TaskState.WAITING_FOR_APPROVAL: {
        TaskState.QUEUED,
        TaskState.FAILED,
        TaskState.CANCELLED,
    },
    TaskState.PAUSED: {TaskState.QUEUED, TaskState.CANCELLED},
    TaskState.COMPLETED: set(),
    TaskState.FAILED: set(),
    TaskState.CANCELLED: set(),
}


class _TaskPaused(Exception):
    pass


class _TaskCancelled(Exception):
    pass


class TaskSuspended(Exception):
    pass


@dataclass
class TaskExecutionRequest:
    task_id: str
    conversation_id: str
    user_message: str
    messages: list[dict[str, Any]]
    seen_call_ids: set[str]
    base_message_count: int
    pending_operation: Operation | None = None
    pending_call_id: str | None = None
    approved_operation: AuthorizedOperation | None = None
    approved_call_id: str | None = None
    rejected_tool_result: tuple[str, str, str] | None = None
    deferred_tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class TaskExecutionHooks:
    checkpoint: Callable[..., None]
    update_activity: Callable[[str], None]
    tool_started: Callable[[str], None]
    tool_finished: Callable[[str], None]
    request_user_input: Callable[[str, tuple[str, ...]], None]
    task_context: Callable[[], TaskContext]
    execute_authorized: Callable[[AuthorizedOperation], str]
    request_approval: Callable[[str, dict[str, Any], str], None]
    security_gate: SecurityGate


TaskExecutor = Callable[[TaskExecutionRequest, TaskExecutionHooks], str]


@dataclass
class _Conversation:
    conversation_id: str
    messages: list[dict[str, Any]]
    seen_call_ids: set[str] = field(default_factory=set)


class TaskSubscription:
    def __init__(self, manager: "TaskManager", queue_limit: int):
        self._manager = manager
        self._events: deque[TaskEvent] = deque()
        self._queue_limit = queue_limit
        self._resync_required = False
        self._resync_reported = False
        self._closed = False

    def _push_locked(self, event: TaskEvent) -> None:
        if self._closed or self._resync_required:
            return
        if len(self._events) >= self._queue_limit:
            self._events.clear()
            self._resync_required = True
        else:
            self._events.append(copy.deepcopy(event))
        self._manager._condition.notify_all()

    def get(self, timeout: float | None = None) -> TaskEvent | ResyncRequired | None:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._manager._condition:
            while not self._events and not self._closed and not self._resync_required:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._manager._condition.wait(remaining)

            if self._resync_required and not self._resync_reported:
                self._resync_reported = True
                return ResyncRequired(
                    manager_epoch=self._manager.manager_epoch,
                    current_sequence=self._manager._sequence,
                )
            if self._events:
                return self._events.popleft()
            return None

    def close(self) -> None:
        with self._manager._condition:
            if not self._closed:
                self._closed = True
                self._manager._subscriptions.discard(self)
                self._manager._condition.notify_all()


class TaskManager:
    """Owns task lifecycle, task contexts, the serial worker, and task events."""

    def __init__(
        self,
        executor: TaskExecutor,
        system_prompt: str = "",
        event_history_limit: int = 512,
        security_gate: SecurityGate = DEFAULT_SECURITY_GATE,
    ):
        if event_history_limit < 1:
            raise ValueError("event_history_limit must be at least 1")

        self.manager_epoch = str(uuid4())
        self._executor = executor
        self._security_gate = security_gate
        self._system_prompt = system_prompt
        self._condition = threading.Condition()
        self._tasks: dict[str, Task] = {}
        self._contexts: dict[str, TaskExecutionRequest] = {}
        self._conversations: dict[str, _Conversation] = {}
        self._initialized_contexts: set[str] = set()
        self._committed_tasks: set[str] = set()
        self._queue: deque[str] = deque()
        self._active_task_id: str | None = None
        self._focused_task_id: str | None = None
        self._sequence = 0
        self._events: deque[TaskEvent] = deque(maxlen=event_history_limit)
        self._subscriptions: set[TaskSubscription] = set()
        self._event_history_limit = event_history_limit
        self._command_results: dict[str, CommandResult] = {}
        self._closed = False
        self.default_conversation_id = self.create_conversation()
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="jarvis-task-worker",
            daemon=True,
        )
        self._worker.start()

    def create_conversation(self) -> str:
        conversation_id = str(uuid4())
        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        with self._condition:
            self._conversations[conversation_id] = _Conversation(
                conversation_id=conversation_id,
                messages=messages,
            )
        return conversation_id

    def submit_message(
        self,
        text: str,
        conversation_id: str | None = None,
        command_id: str | None = None,
    ) -> CommandResult:
        return self.send_command(
            TaskCommand(
                command_id=command_id or str(uuid4()),
                command_type="submit_message",
                conversation_id=conversation_id or self.default_conversation_id,
                payload={"text": text},
            )
        )

    def focus_task(
        self,
        task_id: str,
        command_id: str | None = None,
        expected_revision: int | None = None,
    ) -> CommandResult:
        return self.send_command(
            TaskCommand(
                command_id or str(uuid4()),
                "focus_task",
                task_id=task_id,
                expected_revision=expected_revision,
            )
        )

    def background_task(
        self,
        task_id: str,
        command_id: str | None = None,
        expected_revision: int | None = None,
    ) -> CommandResult:
        return self.send_command(
            TaskCommand(
                command_id or str(uuid4()),
                "background_task",
                task_id=task_id,
                expected_revision=expected_revision,
            )
        )

    def pause_task(
        self,
        task_id: str,
        command_id: str | None = None,
        expected_revision: int | None = None,
    ) -> CommandResult:
        return self.send_command(
            TaskCommand(
                command_id or str(uuid4()),
                "pause_task",
                task_id=task_id,
                expected_revision=expected_revision,
            )
        )

    def resume_task(
        self,
        task_id: str,
        command_id: str | None = None,
        expected_revision: int | None = None,
    ) -> CommandResult:
        return self.send_command(
            TaskCommand(
                command_id or str(uuid4()),
                "resume_task",
                task_id=task_id,
                expected_revision=expected_revision,
            )
        )

    def cancel_task(
        self,
        task_id: str,
        command_id: str | None = None,
        expected_revision: int | None = None,
    ) -> CommandResult:
        return self.send_command(
            TaskCommand(
                command_id or str(uuid4()),
                "cancel_task",
                task_id=task_id,
                expected_revision=expected_revision,
            )
        )

    def answer_question(
        self,
        task_id: str,
        question_id: str,
        answer: str,
        command_id: str | None = None,
        expected_revision: int | None = None,
    ) -> CommandResult:
        return self.send_command(
            TaskCommand(
                command_id or str(uuid4()),
                "answer_question",
                task_id=task_id,
                payload={"question_id": question_id, "answer": answer},
                expected_revision=expected_revision,
            )
        )

    def approve_action(
        self,
        task_id: str,
        approval_id: str,
        expected_revision: int,
        command_id: str | None = None,
    ) -> CommandResult:
        return self.send_command(TaskCommand(
            command_id or str(uuid4()), "approve_action", task_id=task_id,
            payload={"approval_id": approval_id}, expected_revision=expected_revision,
        ))

    def reject_action(
        self,
        task_id: str,
        approval_id: str,
        expected_revision: int,
        command_id: str | None = None,
    ) -> CommandResult:
        return self.send_command(TaskCommand(
            command_id or str(uuid4()), "reject_action", task_id=task_id,
            payload={"approval_id": approval_id}, expected_revision=expected_revision,
        ))

    def send_command(self, command: TaskCommand) -> CommandResult:
        """Apply an idempotent command with an optional task-state revision guard.

        The manager epoch and global event sequence identify an event replay cursor;
        task revision rejects commands based on a stale task snapshot. SecurityGate
        authorization separately binds a sensitive operation to the task revision at
        approval time, which is why approval-related events can preserve that revision.
        """
        with self._condition:
            previous = self._command_results.get(command.command_id)
            if previous is not None:
                return CommandResult(
                    command_id=previous.command_id,
                    accepted=previous.accepted,
                    task_id=previous.task_id,
                    message=previous.message,
                    duplicate=True,
                )

            if command.expected_revision is not None and command.task_id:
                task = self._tasks.get(command.task_id)
                if task is not None and task.revision != command.expected_revision:
                    result = CommandResult(
                        command.command_id,
                        False,
                        task_id=task.task_id,
                        message="Task revision changed; refresh the task before retrying.",
                    )
                    self._command_results[command.command_id] = result
                    return result

            try:
                result = self._dispatch_locked(command)
            except (KeyError, TypeError, ValueError) as error:
                result = CommandResult(
                    command_id=command.command_id,
                    accepted=False,
                    task_id=command.task_id,
                    message=str(error),
                )
            self._command_results[command.command_id] = result
            return result

    def _dispatch_locked(self, command: TaskCommand) -> CommandResult:
        if self._closed:
            return CommandResult(command.command_id, False, message="Task Manager is shutting down.")

        if command.command_type == "submit_message":
            return self._submit_locked(command)
        if command.command_type == "focus_task":
            return self._focus_locked(command)
        if command.command_type == "background_task":
            return self._background_locked(command)
        if command.command_type == "pause_task":
            return self._pause_locked(command)
        if command.command_type == "resume_task":
            return self._resume_locked(command)
        if command.command_type == "cancel_task":
            return self._cancel_locked(command)
        if command.command_type == "answer_question":
            return self._answer_locked(command)
        if command.command_type == "approve_action":
            return self._approve_action_locked(command)
        if command.command_type == "reject_action":
            return self._reject_action_locked(command)
        return CommandResult(command.command_id, False, message="Unknown command.")

    def _submit_locked(self, command: TaskCommand) -> CommandResult:
        text = command.payload.get("text")
        if not isinstance(text, str) or not text.strip():
            return CommandResult(command.command_id, False, message="Message must not be empty.")
        conversation_id = command.conversation_id
        conversation = self._conversations.get(conversation_id or "")
        if conversation is None:
            return CommandResult(command.command_id, False, message="Conversation not found.")

        task_id = str(uuid4())
        messages = copy.deepcopy(conversation.messages)
        base_message_count = len(messages)
        messages.append({"role": "user", "content": text})
        task = Task(
            task_id=task_id,
            conversation_id=conversation.conversation_id,
            state=TaskState.QUEUED,
            revision=0,
            objective=text.strip(),
            current_activity="Queued for execution",
            plan=[TaskStep("request", "Handle request")],
            created_at=self._now(),
            updated_at=self._now(),
        )
        request = TaskExecutionRequest(
            task_id=task_id,
            conversation_id=conversation.conversation_id,
            user_message=text,
            messages=messages,
            # IDs are reserved per conversation while message histories remain task-local.
            seen_call_ids=conversation.seen_call_ids,
            base_message_count=base_message_count,
        )
        self._tasks[task_id] = task
        self._contexts[task_id] = request
        self._queue.append(task_id)
        self._emit_locked(
            "task_created",
            task,
            {"conversation_id": task.conversation_id, "objective": task.objective},
            f"Task created: {task.objective}",
        )
        self._emit_locked(
            "task_queued",
            task,
            {"queue_position": len(self._queue)},
            "Task queued",
        )
        self._condition.notify_all()
        return CommandResult(command.command_id, True, task_id=task_id, message="Task queued.")

    def _focus_locked(self, command: TaskCommand) -> CommandResult:
        task = self._tasks.get(command.task_id or "")
        if task is None:
            return CommandResult(command.command_id, False, message="Task not found.")
        self._focused_task_id = task.task_id
        self._emit_locked(
            "focus_changed",
            None,
            {"focused_task_id": task.task_id},
            None,
        )
        return CommandResult(command.command_id, True, task_id=task.task_id)

    def _background_locked(self, command: TaskCommand) -> CommandResult:
        task = self._tasks.get(command.task_id or "")
        if task is None:
            return CommandResult(command.command_id, False, message="Task not found.")
        if self._focused_task_id == task.task_id:
            self._focused_task_id = None
            self._emit_locked("focus_changed", None, {"focused_task_id": None}, None)
        return CommandResult(command.command_id, True, task_id=task.task_id)

    def _pause_locked(self, command: TaskCommand) -> CommandResult:
        task = self._tasks.get(command.task_id or "")
        if task is None:
            return CommandResult(command.command_id, False, message="Task not found.")
        if task.state == TaskState.QUEUED:
            self._remove_queued_locked(task.task_id)
            task.pause_requested = False
            self._set_activity_locked(task, "Paused before execution")
            self._transition_locked(task, TaskState.PAUSED, "Paused before execution")
            return CommandResult(command.command_id, True, task_id=task.task_id)
        if task.state == TaskState.RUNNING and not task.pause_requested:
            task.pause_requested = True
            self._set_activity_locked(task, "Pause requested; waiting for a safe boundary")
            self._emit_locked(
                "task_control_requested",
                task,
                {"control": "pause", "current_activity": task.current_activity},
                "Pause requested",
            )
            return CommandResult(command.command_id, True, task_id=task.task_id)
        return CommandResult(command.command_id, False, task_id=task.task_id, message="Task cannot be paused in its current state.")

    def _resume_locked(self, command: TaskCommand) -> CommandResult:
        task = self._tasks.get(command.task_id or "")
        if task is None:
            return CommandResult(command.command_id, False, message="Task not found.")
        if task.state != TaskState.PAUSED:
            return CommandResult(command.command_id, False, task_id=task.task_id, message="Only paused tasks can be resumed.")
        task.pause_requested = False
        self._transition_locked(task, TaskState.QUEUED, "Resume requested")
        self._queue.append(task.task_id)
        self._emit_locked("task_queued", task, {"queue_position": len(self._queue)}, "Task queued to resume")
        self._condition.notify_all()
        return CommandResult(command.command_id, True, task_id=task.task_id)

    def _cancel_locked(self, command: TaskCommand) -> CommandResult:
        task = self._tasks.get(command.task_id or "")
        if task is None:
            return CommandResult(command.command_id, False, message="Task not found.")
        if task.state in {TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED}:
            return CommandResult(command.command_id, False, task_id=task.task_id, message="Task is already finished.")
        self._security_gate.cancel_task(task.task_id)
        if task.state == TaskState.RUNNING:
            task.cancel_requested = True
            self._set_activity_locked(task, "Cancellation requested; waiting for a safe boundary")
            self._emit_locked(
                "task_control_requested",
                task,
                {"control": "cancel", "current_activity": task.current_activity},
                "Cancellation requested",
            )
        else:
            self._remove_queued_locked(task.task_id)
            self._finish_locked(task, TaskState.CANCELLED, "Cancelled by user.")
        return CommandResult(command.command_id, True, task_id=task.task_id)

    def _answer_locked(self, command: TaskCommand) -> CommandResult:
        task = self._tasks.get(command.task_id or "")
        question_id = command.payload.get("question_id")
        answer = command.payload.get("answer")
        if task is None:
            return CommandResult(command.command_id, False, message="Task not found.")
        if task.state != TaskState.WAITING_FOR_USER_INPUT or task.pending_question is None:
            return CommandResult(command.command_id, False, task_id=task.task_id, message="Task is not waiting for input.")
        if question_id != task.pending_question.question_id:
            return CommandResult(command.command_id, False, task_id=task.task_id, message="Question ID does not match.")
        if not isinstance(answer, str) or not answer.strip():
            return CommandResult(command.command_id, False, task_id=task.task_id, message="Answer must not be empty.")

        self._contexts[task.task_id].messages.append({"role": "user", "content": answer})
        task.pending_question = None
        self._transition_locked(task, TaskState.QUEUED, "User answered the pending question")
        self._queue.append(task.task_id)
        self._emit_locked("task_queued", task, {"queue_position": len(self._queue)}, "Task queued to continue")
        self._condition.notify_all()
        return CommandResult(command.command_id, True, task_id=task.task_id)

    def _approve_action_locked(self, command: TaskCommand) -> CommandResult:
        task = self._tasks.get(command.task_id or "")
        if task is None:
            return CommandResult(command.command_id, False, message="Task not found.")
        if command.expected_revision is None:
            return CommandResult(command.command_id, False, task_id=task.task_id, message="Expected task revision is required.")
        approval_id = command.payload.get("approval_id")
        request = self._contexts[task.task_id]
        operation = request.pending_operation
        if task.state != TaskState.WAITING_FOR_APPROVAL or operation is None or task.pending_approval is None:
            return CommandResult(command.command_id, False, task_id=task.task_id, message="No pending approval.")
        if approval_id != task.pending_approval.approval_id or approval_id != operation.operation_id:
            return CommandResult(command.command_id, False, task_id=task.task_id, message="Approval ID does not match.")
        try:
            context = TaskContext(task.task_id, task.revision, task.cancel_requested)
            grant = self._security_gate.approve(context, operation)
            authorized = self._security_gate.authorize(context, operation, grant)
        except (ValueError, PermissionError) as error:
            return CommandResult(command.command_id, False, task_id=task.task_id, message=str(error))

        request.pending_operation = None
        request.approved_operation = authorized
        request.approved_call_id = request.pending_call_id
        request.pending_call_id = None
        task.pending_approval = None
        # Approval lifecycle events do not change the exact operation revision.
        # Their event sequence advances normally; a later task change invalidates it.
        self._emit_locked(
            "approval_decided", task,
            {"approval_id": approval_id, "decision": "approved"},
            "Operation approved", increment_revision=False,
        )
        self._transition_locked(task, TaskState.QUEUED, "Approved operation queued", increment_revision=False)
        self._queue.append(task.task_id)
        self._emit_locked(
            "task_queued", task, {"queue_position": len(self._queue)},
            "Approved operation queued", increment_revision=False,
        )
        self._condition.notify_all()
        return CommandResult(command.command_id, True, task_id=task.task_id, message="Operation approved.")

    def _reject_action_locked(self, command: TaskCommand) -> CommandResult:
        task = self._tasks.get(command.task_id or "")
        if task is None:
            return CommandResult(command.command_id, False, message="Task not found.")
        if command.expected_revision is None:
            return CommandResult(command.command_id, False, task_id=task.task_id, message="Expected task revision is required.")
        approval_id = command.payload.get("approval_id")
        request = self._contexts[task.task_id]
        operation = request.pending_operation
        if task.state != TaskState.WAITING_FOR_APPROVAL or operation is None or task.pending_approval is None:
            return CommandResult(command.command_id, False, task_id=task.task_id, message="No pending approval.")
        if approval_id != task.pending_approval.approval_id:
            return CommandResult(command.command_id, False, task_id=task.task_id, message="Approval ID does not match.")

        self._security_gate.reject(operation)
        request.rejected_tool_result = (
            operation.tool_name, request.pending_call_id or "", "Operation rejected by user."
        )
        request.pending_operation = None
        request.pending_call_id = None
        task.pending_approval = None
        self._emit_locked(
            "approval_decided", task,
            {"approval_id": approval_id, "decision": "rejected"},
            "Operation rejected", increment_revision=False,
        )
        self._transition_locked(task, TaskState.QUEUED, "Rejected operation returned to the task", increment_revision=False)
        self._queue.append(task.task_id)
        self._emit_locked(
            "task_queued", task, {"queue_position": len(self._queue)},
            "Task queued after rejection", increment_revision=False,
        )
        self._condition.notify_all()
        return CommandResult(command.command_id, True, task_id=task.task_id, message="Operation rejected.")

    def request_user_input(
        self,
        task_id: str,
        prompt: str,
        options: tuple[str, ...] = (),
    ) -> None:
        with self._condition:
            task = self._require_task_locked(task_id)
            if task.state != TaskState.RUNNING:
                raise ValueError("Only a running task can request user input.")
            question = UserQuestion(str(uuid4()), prompt, options)
            task.pending_question = question
            self._contexts[task_id].messages.append({"role": "assistant", "content": prompt})
            self._transition_locked(task, TaskState.WAITING_FOR_USER_INPUT, "Waiting for user input")
            self._emit_locked(
                "user_input_requested",
                task,
                {"question_id": question.question_id, "prompt": prompt, "options": options},
                "Waiting for user input",
            )
        raise TaskSuspended()

    def request_approval(self, task_id: str, tool_name: str, arguments: dict[str, Any], call_id: str) -> None:
        """Suspend a task with a Gate-bound operation and user-safe approval event."""
        with self._condition:
            task = self._require_task_locked(task_id)
            if task.state != TaskState.RUNNING:
                raise ValueError("Only a running task can request approval.")
            if task.cancel_requested:
                raise _TaskCancelled()
            self._transition_locked(task, TaskState.WAITING_FOR_APPROVAL, "Waiting for approval")
            context = TaskContext(task.task_id, task.revision, task.cancel_requested)
            try:
                operation, decision = self._security_gate.propose(
                    context, tool_name, arguments
                )
                if not decision.requires_approval:
                    self._security_gate.reject(operation)
                    raise ValueError(
                        "The Security Gate did not require approval for this operation."
                    )
            except Exception:
                # A failed proposal must be terminal and invalidate any Gate-side
                # record that may have been created before the failure.
                self._security_gate.cancel_task(task.task_id)
                self._finish_locked(
                    task,
                    TaskState.FAILED,
                    "Approval could not be prepared. No action was executed.",
                )
                raise TaskSuspended()

            summary = f"{decision.tool_name} ({decision.risk})"
            if decision.target:
                summary += f" — {decision.target}"
            approval = ApprovalRequest(operation.operation_id, summary)
            task.pending_approval = approval
            request = self._contexts[task_id]
            request.pending_operation = operation
            request.pending_call_id = call_id
            self._emit_locked(
                "approval_requested",
                task,
                {
                    "approval_id": approval.approval_id,
                    "operation_id": operation.operation_id,
                    "tool_name": decision.tool_name,
                    "risk": decision.risk,
                    "target": decision.target,
                    "summary": summary,
                },
                "Waiting for approval",
                increment_revision=False,
            )
        raise TaskSuspended()

    def get_snapshot(self) -> TaskSnapshot:
        with self._condition:
            return TaskSnapshot(
                manager_epoch=self.manager_epoch,
                last_sequence=self._sequence,
                focused_task_id=self._focused_task_id,
                active_task_id=self._active_task_id,
                queued_task_ids=tuple(self._queue),
                tasks=tuple(copy.deepcopy(tuple(self._tasks.values()))),
            )

    def get_view_snapshot(self) -> TaskSnapshotView:
        return project_snapshot(self.get_snapshot())

    def get_task_view(self, task_id: str) -> TaskView | None:
        task = self.get_task(task_id)
        return project_task(task) if task is not None else None

    def subscribe_views(
        self,
        after_sequence: int,
        manager_epoch: str | None = None,
    ) -> TaskViewSubscription:
        return TaskViewSubscription(self.subscribe(after_sequence, manager_epoch))

    def get_task(self, task_id: str) -> Task | None:
        with self._condition:
            task = self._tasks.get(task_id)
            return copy.deepcopy(task) if task else None

    def subscribe(
        self,
        after_sequence: int,
        manager_epoch: str | None = None,
    ) -> TaskSubscription:
        subscription = TaskSubscription(self, self._event_history_limit)
        with self._condition:
            history_is_stale = bool(
                self._events and after_sequence < self._events[0].sequence - 1
            )
            invalid_cursor = after_sequence < 0 or after_sequence > self._sequence
            wrong_epoch = manager_epoch is not None and manager_epoch != self.manager_epoch
            if history_is_stale or invalid_cursor or wrong_epoch:
                subscription._resync_required = True
                return subscription

            for event in self._events:
                if event.sequence > after_sequence:
                    subscription._events.append(copy.deepcopy(event))
            self._subscriptions.add(subscription)
            return subscription

    def close(self, wait: bool = True) -> None:
        with self._condition:
            if not self._closed:
                self._closed = True
                for task in self._tasks.values():
                    if task.task_id == self._active_task_id or task.state in {
                        TaskState.COMPLETED,
                        TaskState.FAILED,
                        TaskState.CANCELLED,
                    }:
                        continue
                    self._remove_queued_locked(task.task_id)
                    self._finish_locked(
                        task,
                        TaskState.CANCELLED,
                        "Cancelled because JARVIS is shutting down.",
                    )
                if self._active_task_id:
                    active = self._tasks[self._active_task_id]
                    active.cancel_requested = True
                    self._emit_locked(
                        "task_control_requested",
                        active,
                        {"control": "cancel", "reason": "shutdown"},
                        "Cancellation requested for shutdown",
                    )
                self._condition.notify_all()
        if wait and threading.current_thread() is not self._worker:
            self._worker.join()
        with self._condition:
            for subscription in list(self._subscriptions):
                subscription.close()

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                while not self._queue and not self._closed:
                    self._condition.wait()
                if self._closed and not self._queue:
                    return
                task_id = self._queue.popleft()
                task = self._tasks[task_id]
                if task.state != TaskState.QUEUED:
                    continue
                request = self._contexts[task_id]
                if task_id not in self._initialized_contexts:
                    conversation = self._conversations[task.conversation_id]
                    request.messages = copy.deepcopy(conversation.messages)
                    request.base_message_count = len(request.messages)
                    request.messages.append(
                        {"role": "user", "content": request.user_message}
                    )
                    request.seen_call_ids = conversation.seen_call_ids
                    self._initialized_contexts.add(task_id)
                self._active_task_id = task_id
                resuming_approval = request.approved_operation is not None or request.rejected_tool_result is not None
                self._transition_locked(
                    task, TaskState.RUNNING, "Execution started",
                    increment_revision=not resuming_approval,
                )
                if task.plan and not resuming_approval:
                    task.plan[0].state = StepState.RUNNING
                    self._emit_locked("plan_updated", task, self._plan_payload(task), "Plan step started")
                self._set_activity_locked(
                    task, "Processing request", increment_revision=not resuming_approval,
                )
                hooks = self._make_hooks(task_id)

            outcome = None
            error = None
            try:
                outcome = self._executor(request, hooks)
            except TaskSuspended:
                pass
            except (_TaskPaused, _TaskCancelled) as control:
                outcome = control
            except Exception as exception:
                error = exception

            with self._condition:
                task = self._tasks[task_id]
                if error is not None and task.state == TaskState.RUNNING:
                    self._finish_locked(task, TaskState.FAILED, f"Task failed: {error}")
                elif isinstance(outcome, _TaskCancelled) or (
                    task.cancel_requested
                    and task.state
                    not in {TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED}
                ):
                    self._finish_locked(task, TaskState.CANCELLED, "Cancelled by user.")
                elif isinstance(outcome, _TaskPaused):
                    task.pause_requested = False
                    self._set_activity_locked(task, "Paused at a safe boundary")
                    self._transition_locked(task, TaskState.PAUSED, "Paused at a safe boundary")
                elif task.state == TaskState.RUNNING:
                    self._finish_locked(task, TaskState.COMPLETED, str(outcome or ""))
                self._active_task_id = None
                self._condition.notify_all()

    def _make_hooks(self, task_id: str) -> TaskExecutionHooks:
        return TaskExecutionHooks(
            checkpoint=lambda honor_pause=True: self._checkpoint(task_id, honor_pause),
            update_activity=lambda activity: self._update_activity(task_id, activity),
            tool_started=lambda name: self._tool_started(task_id, name),
            tool_finished=lambda name: self._tool_finished(task_id, name),
            request_user_input=lambda prompt, options=(): self.request_user_input(task_id, prompt, options),
            task_context=lambda: self._task_context(task_id),
            execute_authorized=lambda authorized: self._execute_authorized(task_id, authorized),
            request_approval=lambda name, arguments, call_id: self.request_approval(
                task_id, name, arguments, call_id
            ),
            security_gate=self._security_gate,
        )

    def _task_context(self, task_id: str) -> TaskContext:
        with self._condition:
            task = self._require_task_locked(task_id)
            return TaskContext(task.task_id, task.revision, task.cancel_requested or task.state == TaskState.CANCELLED)

    def _execute_authorized(self, task_id: str, authorized: AuthorizedOperation) -> str:
        # Claim and consume the one-use authorization atomically with task state.
        # The runner detaches arguments from the shallow-frozen authorization object.
        with self._condition:
            task = self._require_task_locked(task_id)
            context = TaskContext(
                task.task_id,
                task.revision,
                task.cancel_requested or task.state == TaskState.CANCELLED,
            )
            if task.state != TaskState.RUNNING or context.cancelled:
                return "Execution denied: task is cancelled or not running."
            if (authorized.task_id, authorized.revision) != (task.task_id, task.revision):
                self._security_gate.invalidate_task_revision(task.task_id, task.revision)
                return "Execution denied: task revision changed."
            claimed = claim_authorized_operation(
                context, authorized, gate=self._security_gate
            )
            if isinstance(claimed, str):
                return claimed

        # Cancellation and UI state operations can acquire the manager lock while the
        # authorized tool performs potentially slow work. Cancellation remains a
        # safe-boundary request after this one-use claim has been consumed.
        return execute_claimed(claimed)

    def _checkpoint(self, task_id: str, honor_pause: bool = True) -> None:
        with self._condition:
            task = self._require_task_locked(task_id)
            if task.cancel_requested:
                raise _TaskCancelled()
            if task.pause_requested and honor_pause:
                task.pause_requested = False
                raise _TaskPaused()

    def _update_activity(self, task_id: str, activity: str) -> None:
        with self._condition:
            task = self._require_task_locked(task_id)
            if task.state == TaskState.RUNNING:
                self._set_activity_locked(task, activity)

    def _tool_started(self, task_id: str, name: str) -> None:
        with self._condition:
            task = self._require_task_locked(task_id)
            if task.state != TaskState.RUNNING:
                return
            task.statistics.tool_calls += 1
            self._emit_locked(
                "statistics_updated",
                task,
                {"tool_calls": task.statistics.tool_calls},
                f"Tool call started: {name}",
                increment_revision=False,
            )
            self._emit_locked("tool_started", task, {"tool_name": name}, None, increment_revision=False)

    def _tool_finished(self, task_id: str, name: str) -> None:
        with self._condition:
            task = self._require_task_locked(task_id)
            self._emit_locked(
                "tool_finished",
                task,
                {"tool_name": name, "status": "finished"},
                f"Tool call finished: {name}",
                increment_revision=False,
            )

    def _finish_locked(self, task: Task, state: TaskState, result: str) -> None:
        if state == TaskState.CANCELLED:
            self._security_gate.cancel_task(task.task_id)
        task.result = result
        task.pause_requested = False
        task.cancel_requested = False
        task.pending_question = None
        task.pending_approval = None
        request = self._contexts.get(task.task_id)
        if request is not None:
            request.pending_operation = None
            request.pending_call_id = None
            request.approved_operation = None
            request.approved_call_id = None
            request.rejected_tool_result = None
            request.deferred_tool_calls.clear()
        if task.plan:
            task.plan[0].state = {
                TaskState.COMPLETED: StepState.COMPLETED,
                TaskState.FAILED: StepState.FAILED,
                TaskState.CANCELLED: StepState.CANCELLED,
            }[state]
            if state == TaskState.COMPLETED:
                task.statistics.steps_completed += 1
            self._emit_locked("plan_updated", task, self._plan_payload(task), "Plan updated")
        self._commit_context_locked(task, completed=state == TaskState.COMPLETED)
        self._emit_locked("result_updated", task, {"result": result}, "Task result updated")
        self._transition_locked(task, state, result if state != TaskState.COMPLETED else "Task completed")
        self._emit_locked(
            "task_finished",
            task,
            {"state": task.state.value, "result": result},
            f"Task {state.value}",
        )

    def _commit_context_locked(self, task: Task, completed: bool = False) -> None:
        if task.task_id in self._committed_tasks:
            return
        request = self._contexts[task.task_id]
        conversation = self._conversations[task.conversation_id]
        if completed:
            conversation.messages.extend(
                copy.deepcopy(request.messages[request.base_message_count:])
            )
            conversation.seen_call_ids.update(request.seen_call_ids)
        else:
            conversation.messages.extend(
                [
                    {"role": "user", "content": request.user_message},
                    {"role": "assistant", "content": task.result or ""},
                ]
            )
        self._committed_tasks.add(task.task_id)

    def _transition_locked(
        self, task: Task, state: TaskState, reason: str, increment_revision: bool = True,
    ) -> None:
        if state not in _ALLOWED_TRANSITIONS[task.state]:
            raise ValueError(f"Invalid task transition: {task.state.value} -> {state.value}")
        previous = task.state
        task.state = state
        if state in {TaskState.WAITING_FOR_USER_INPUT, TaskState.WAITING_FOR_APPROVAL}:
            task.current_activity = reason
        self._emit_locked(
            "task_state_changed",
            task,
            {"from_state": previous.value, "to_state": state.value, "reason": reason},
            reason,
            increment_revision=increment_revision,
        )

    def _set_activity_locked(self, task: Task, activity: str, increment_revision: bool = True) -> None:
        if task.current_activity == activity:
            return
        task.current_activity = activity
        self._emit_locked(
            "activity_updated",
            task,
            {"current_activity": activity},
            activity,
            increment_revision=increment_revision,
        )

    def _emit_locked(
        self,
        event_type: str,
        task: Task | None,
        payload: dict[str, Any],
        timeline_summary: str | None,
        increment_revision: bool = True,
    ) -> TaskEvent:
        self._sequence += 1
        timestamp = self._now()
        if task is not None:
            if increment_revision:
                task.revision += 1
            task.updated_at = timestamp
            if timeline_summary:
                task.timeline.append(
                    TimelineEntry(self._sequence, timestamp, event_type, timeline_summary)
                )
        event = TaskEvent(
            manager_epoch=self.manager_epoch,
            sequence=self._sequence,
            event_id=str(uuid4()),
            timestamp=timestamp,
            event_type=event_type,
            task_id=task.task_id if task else None,
            task_revision=task.revision if task else None,
            payload=copy.deepcopy(payload),
        )
        self._events.append(event)
        for subscription in tuple(self._subscriptions):
            subscription._push_locked(event)
        self._condition.notify_all()
        return event

    def _plan_payload(self, task: Task) -> dict[str, Any]:
        return {
            "steps": [
                {"step_id": step.step_id, "title": step.title, "state": step.state.value}
                for step in task.plan
            ]
        }

    def _require_task_locked(self, task_id: str) -> Task:
        task = self._tasks.get(task_id)
        if task is None:
            raise KeyError("Task not found.")
        return task

    def _remove_queued_locked(self, task_id: str) -> None:
        self._queue = deque(queued for queued in self._queue if queued != task_id)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()
