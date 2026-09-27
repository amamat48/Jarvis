from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_FOR_USER_INPUT = "waiting_for_user_input"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskStep:
    step_id: str
    title: str
    state: StepState = StepState.PENDING


@dataclass
class TaskStatistics:
    steps_completed: int = 0
    tool_calls: int = 0
    retries: int = 0


@dataclass
class TimelineEntry:
    sequence: int
    timestamp: str
    event_type: str
    summary: str


@dataclass
class UserQuestion:
    question_id: str
    prompt: str
    options: tuple[str, ...] = ()


@dataclass
class ApprovalRequest:
    approval_id: str
    summary: str
    status: str = "pending"


@dataclass
class Task:
    task_id: str
    conversation_id: str
    state: TaskState
    revision: int
    objective: str
    current_activity: str
    plan: list[TaskStep]
    result: str | None = None
    statistics: TaskStatistics = field(default_factory=TaskStatistics)
    timeline: list[TimelineEntry] = field(default_factory=list)
    pending_question: UserQuestion | None = None
    pending_approval: ApprovalRequest | None = None
    pause_requested: bool = False
    cancel_requested: bool = False
    created_at: str = ""
    updated_at: str = ""


@dataclass(frozen=True)
class TaskEvent:
    manager_epoch: str
    sequence: int
    event_id: str
    timestamp: str
    event_type: str
    task_id: str | None
    task_revision: int | None
    payload: dict[str, Any]


@dataclass(frozen=True)
class TaskCommand:
    command_id: str
    command_type: str
    task_id: str | None = None
    conversation_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    expected_revision: int | None = None


@dataclass(frozen=True)
class CommandResult:
    command_id: str
    accepted: bool
    task_id: str | None = None
    message: str = ""
    duplicate: bool = False


@dataclass(frozen=True)
class TaskSnapshot:
    manager_epoch: str
    last_sequence: int
    focused_task_id: str | None
    active_task_id: str | None
    queued_task_ids: tuple[str, ...]
    tasks: tuple[Task, ...]


@dataclass(frozen=True)
class ResyncRequired:
    manager_epoch: str
    current_sequence: int
