import copy
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tasks.manager import TaskExecutionRequest, TaskManager
from tasks.models import (
    ApprovalRequest,
    ResyncRequired,
    Task,
    TaskCommand,
    TaskEvent,
    TaskState,
)
from tasks.views import project_event, project_task
from tools.security import (
    AuthorizedOperation,
    AuthorizationError,
    SecurityGate,
    TaskContext,
    ToolSecurity,
)


def wait_for(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class TaskManagerTests(unittest.TestCase):
    def make_manager(self, executor, **kwargs):
        kwargs.setdefault("max_concurrent_tasks", 1)
        manager = TaskManager(executor=executor, system_prompt="system", **kwargs)
        self.addCleanup(manager.close)
        return manager

    def test_task_creation_and_result(self):
        manager = self.make_manager(lambda request, hooks: "answer")
        conversation_id = manager.create_conversation()
        subscription = manager.subscribe(0, manager.manager_epoch)
        self.addCleanup(subscription.close)

        command = manager.submit_message("What is JARVIS?", conversation_id)

        self.assertTrue(command.accepted)
        task_id = command.task_id
        self.assertTrue(wait_for(lambda: manager.get_task(task_id).state == TaskState.COMPLETED))
        task = manager.get_task(task_id)
        self.assertEqual(task.conversation_id, conversation_id)
        self.assertEqual(task.objective, "What is JARVIS?")
        self.assertEqual(task.result, "answer")
        self.assertEqual(task.statistics.steps_completed, 1)
        self.assertEqual(task.plan[0].state.value, "completed")

        events = []
        while True:
            event = subscription.get(timeout=0.05)
            if event is None:
                break
            events.append(event)
            if event.event_type == "task_finished":
                break
        types = [event.event_type for event in events]
        self.assertLess(types.index("task_created"), types.index("task_queued"))
        self.assertEqual(types[-1], "task_finished")

    def test_independent_tasks_run_concurrently_and_report_progress(self):
        first_started = threading.Event()
        release_first = threading.Event()
        second_finished = threading.Event()

        def executor(request, hooks):
            if request.user_message == "background analysis":
                first_started.set()
                hooks.update_progress(42)
                release_first.wait(2)
                return "analysis done"
            second_finished.set()
            return "side answer"

        manager = self.make_manager(executor, max_concurrent_tasks=2)
        first = manager.submit_message("background analysis", manager.create_conversation())
        self.assertTrue(first_started.wait(1))
        second = manager.submit_message("side question", manager.create_conversation())
        self.assertTrue(second_finished.wait(1), "independent task blocked behind background task")
        self.assertEqual(manager.get_task(first.task_id).progress, 42)
        self.assertIn(first.task_id, manager.get_snapshot().active_task_ids)
        self.assertTrue(wait_for(lambda: manager.get_task(second.task_id).state == TaskState.COMPLETED))
        release_first.set()
        self.assertTrue(wait_for(lambda: manager.get_task(first.task_id).state == TaskState.COMPLETED))

    def test_cancelling_one_concurrent_task_does_not_cancel_another(self):
        cancel_entered = threading.Event()
        other_entered = threading.Event()
        release_cancelled = threading.Event()
        release_other = threading.Event()

        def executor(request, hooks):
            if request.user_message == "cancel this":
                cancel_entered.set()
                release_cancelled.wait(2)
                hooks.checkpoint()
            other_entered.set()
            release_other.wait(2)
            hooks.checkpoint()
            return "independent result"

        manager = self.make_manager(executor, max_concurrent_tasks=2)
        cancelled = manager.submit_message("cancel this", manager.create_conversation())
        other = manager.submit_message("keep running", manager.create_conversation())
        self.assertTrue(cancel_entered.wait(1))
        self.assertTrue(other_entered.wait(1))
        self.assertTrue(manager.cancel_task(cancelled.task_id).accepted)
        release_cancelled.set()
        self.assertTrue(wait_for(lambda: manager.get_task(cancelled.task_id).state == TaskState.CANCELLED))
        self.assertEqual(manager.get_task(other.task_id).state, TaskState.RUNNING)
        release_other.set()
        self.assertTrue(wait_for(lambda: manager.get_task(other.task_id).state == TaskState.COMPLETED))

    def test_same_conversation_tasks_remain_ordered_with_multiple_workers(self):
        started = threading.Event()
        release = threading.Event()
        order = []

        def executor(request, hooks):
            order.append(request.user_message)
            if request.user_message == "first turn":
                started.set()
                release.wait(2)
            return request.user_message

        manager = self.make_manager(executor, max_concurrent_tasks=2)
        conversation = manager.create_conversation()
        first = manager.submit_message("first turn", conversation)
        self.assertTrue(started.wait(1))
        second = manager.submit_message("second turn", conversation)
        time.sleep(0.05)
        self.assertEqual(order, ["first turn"])
        release.set()
        self.assertTrue(wait_for(lambda: manager.get_task(second.task_id).state == TaskState.COMPLETED))
        self.assertEqual(order, ["first turn", "second turn"])

    def test_pause_and_resume_at_safe_boundary(self):
        started = threading.Event()
        release = threading.Event()
        calls = []

        def executor(request, hooks):
            calls.append(request.task_id)
            if len(calls) == 1:
                started.set()
                release.wait(2)
                hooks.checkpoint()
            return "resumed result"

        manager = self.make_manager(executor)
        result = manager.submit_message("long work")
        self.assertTrue(started.wait(1))
        self.assertTrue(manager.pause_task(result.task_id).accepted)
        self.assertTrue(manager.get_task(result.task_id).pause_requested)
        release.set()
        self.assertTrue(wait_for(lambda: manager.get_task(result.task_id).state == TaskState.PAUSED))

        self.assertTrue(manager.resume_task(result.task_id).accepted)
        self.assertTrue(wait_for(lambda: manager.get_task(result.task_id).state == TaskState.COMPLETED))
        self.assertEqual(len(calls), 2)

    def test_cancel_waits_for_safe_boundary(self):
        started = threading.Event()
        release = threading.Event()

        def executor(request, hooks):
            started.set()
            release.wait(2)
            hooks.checkpoint()
            return "should not complete"

        manager = self.make_manager(executor)
        result = manager.submit_message("work to cancel")
        self.assertTrue(started.wait(1))
        self.assertTrue(manager.cancel_task(result.task_id).accepted)
        still_running = manager.get_task(result.task_id)
        self.assertEqual(still_running.state, TaskState.RUNNING)
        self.assertTrue(still_running.cancel_requested)

        release.set()
        self.assertTrue(wait_for(lambda: manager.get_task(result.task_id).state == TaskState.CANCELLED))
        self.assertEqual(manager.get_task(result.task_id).result, "Cancelled by user.")

    def test_shutdown_cancels_active_task_suspended_for_user_input(self):
        suspended = threading.Event()
        release_executor = threading.Event()

        def executor(request, hooks):
            try:
                hooks.request_user_input("Need one detail")
            except Exception:
                suspended.set()
                release_executor.wait(2)
                raise

        manager = self.make_manager(executor)
        result = manager.submit_message("work that needs input")
        self.assertTrue(suspended.wait(1))
        self.assertEqual(manager.get_task(result.task_id).state, TaskState.WAITING_FOR_USER_INPUT)
        self.assertEqual(manager.get_snapshot().active_task_id, result.task_id)

        manager.close(wait=False)
        release_executor.set()

        self.assertTrue(wait_for(lambda: manager.get_task(result.task_id).state == TaskState.CANCELLED))

    def test_task_contexts_are_isolated_between_conversations(self):
        first_started = threading.Event()
        release_first = threading.Event()
        second_context = {}

        def executor(request, hooks):
            if request.user_message == "first":
                request.messages.append({"role": "assistant", "content": "private first-task state"})
                first_started.set()
                release_first.wait(2)
                return "first result"
            second_context["contains_first_state"] = any(
                message.get("content") == "private first-task state"
                for message in request.messages
            )
            return "second result"

        manager = self.make_manager(executor)
        first = manager.submit_message("first", manager.create_conversation())
        self.assertTrue(first_started.wait(1))
        second = manager.submit_message("second", manager.create_conversation())
        release_first.set()

        self.assertTrue(wait_for(lambda: manager.get_task(second.task_id).state == TaskState.COMPLETED))
        self.assertFalse(second_context["contains_first_state"])

    def test_queued_same_conversation_task_uses_prior_completed_context(self):
        first_started = threading.Event()
        release_first = threading.Event()
        second_context = {}

        def executor(request, hooks):
            if request.user_message == "first":
                first_started.set()
                release_first.wait(2)
                request.messages.append({"role": "assistant", "content": "first result"})
                return "first result"
            second_context["contains_first_result"] = any(
                message.get("content") == "first result"
                for message in request.messages
            )
            return "second result"

        manager = self.make_manager(executor)
        conversation_id = manager.create_conversation()
        first = manager.submit_message("first", conversation_id)
        self.assertTrue(first_started.wait(1))
        second = manager.submit_message("second", conversation_id)
        release_first.set()

        self.assertTrue(wait_for(lambda: manager.get_task(second.task_id).state == TaskState.COMPLETED))
        self.assertTrue(second_context["contains_first_result"])

    def test_event_sequences_are_monotonic_and_ordered(self):
        manager = self.make_manager(lambda request, hooks: "done")
        subscription = manager.subscribe(0, manager.manager_epoch)
        self.addCleanup(subscription.close)
        result = manager.submit_message("ordered")
        self.assertTrue(wait_for(lambda: manager.get_task(result.task_id).state == TaskState.COMPLETED))

        sequences = []
        event_types = []
        while True:
            event = subscription.get(timeout=0.05)
            if event is None:
                break
            sequences.append(event.sequence)
            event_types.append(event.event_type)
            if event.event_type == "task_finished":
                break
        self.assertEqual(sequences, sorted(set(sequences)))
        self.assertEqual(event_types[0:2], ["task_created", "task_queued"])

    def test_snapshot_cursor_and_subscribe_replay_have_no_gap(self):
        started = threading.Event()
        release = threading.Event()

        def executor(request, hooks):
            started.set()
            release.wait(2)
            return "done"

        manager = self.make_manager(executor)
        task = manager.submit_message("snapshot")
        self.assertTrue(started.wait(1))
        snapshot = manager.get_snapshot()
        self.assertEqual(snapshot.active_task_id, task.task_id)
        self.assertEqual(snapshot.last_sequence, max(entry.sequence for entry in snapshot.tasks[0].timeline))

        subscription = manager.subscribe(snapshot.last_sequence, snapshot.manager_epoch)
        self.addCleanup(subscription.close)
        self.assertTrue(manager.focus_task(task.task_id).accepted)
        event = subscription.get(timeout=1)
        self.assertEqual(event.sequence, snapshot.last_sequence + 1)
        self.assertEqual(event.event_type, "focus_changed")
        release.set()

    def test_subscribe_replays_retained_events(self):
        started = threading.Event()
        release = threading.Event()

        def executor(request, hooks):
            started.set()
            release.wait(2)
            return "done"

        manager = self.make_manager(executor)
        task = manager.submit_message("replay")
        self.assertTrue(started.wait(1))
        subscription = manager.subscribe(0, manager.manager_epoch)
        self.addCleanup(subscription.close)

        first = subscription.get(timeout=1)
        second = subscription.get(timeout=1)
        self.assertEqual(first.event_type, "task_created")
        self.assertEqual(second.event_type, "task_queued")
        self.assertEqual(first.task_id, task.task_id)
        release.set()

    def test_duplicate_command_id_creates_only_one_task(self):
        started = threading.Event()
        release = threading.Event()

        def executor(request, hooks):
            started.set()
            release.wait(2)
            return "done"

        manager = self.make_manager(executor)
        conversation_id = manager.create_conversation()
        command = TaskCommand(
            command_id="same-command",
            command_type="submit_message",
            conversation_id=conversation_id,
            payload={"text": "once"},
        )
        first = manager.send_command(command)
        self.assertTrue(started.wait(1))
        duplicate = manager.send_command(command)

        self.assertTrue(first.accepted)
        self.assertTrue(duplicate.accepted)
        self.assertTrue(duplicate.duplicate)
        self.assertEqual(first.task_id, duplicate.task_id)
        self.assertEqual(len(manager.get_snapshot().tasks), 1)
        release.set()

    def test_invalid_transitions_are_rejected(self):
        started = threading.Event()
        release = threading.Event()

        def executor(request, hooks):
            if request.user_message == "active":
                started.set()
                release.wait(2)
                return "done"
            return "also done"

        manager = self.make_manager(executor)
        active = manager.submit_message("active")
        self.assertTrue(started.wait(1))
        queued = manager.submit_message("queued")
        original_revision = manager.get_task(queued.task_id).revision

        self.assertFalse(manager.resume_task(queued.task_id).accepted)
        self.assertTrue(manager.pause_task(queued.task_id).accepted)
        self.assertEqual(manager.get_task(queued.task_id).state, TaskState.PAUSED)
        self.assertFalse(manager.pause_task(queued.task_id).accepted)
        stale = manager.send_command(
            TaskCommand(
                command_id="stale-revision",
                command_type="resume_task",
                task_id=queued.task_id,
                expected_revision=original_revision,
            )
        )
        self.assertFalse(stale.accepted)
        self.assertTrue(manager.resume_task(queued.task_id).accepted)
        self.assertTrue(manager.cancel_task(queued.task_id).accepted)
        self.assertEqual(manager.get_task(queued.task_id).state, TaskState.CANCELLED)
        self.assertFalse(manager.cancel_task(queued.task_id).accepted)
        release.set()

    def test_answer_question_resumes_the_same_task_context(self):
        calls = []

        def executor(request, hooks):
            calls.append(request.task_id)
            if len(calls) == 1:
                hooks.request_user_input("Which value should I use?")
            return request.messages[-1]["content"]

        manager = self.make_manager(executor)
        result = manager.submit_message("calculate something")
        self.assertTrue(
            wait_for(
                lambda: manager.get_task(result.task_id).state
                == TaskState.WAITING_FOR_USER_INPUT
            )
        )
        question = manager.get_task(result.task_id).pending_question
        answer = manager.answer_question(result.task_id, question.question_id, "42")

        self.assertTrue(answer.accepted)
        self.assertTrue(wait_for(lambda: manager.get_task(result.task_id).state == TaskState.COMPLETED))
        self.assertEqual(manager.get_task(result.task_id).result, "42")
        self.assertEqual(calls, [result.task_id, result.task_id])

    def test_focus_change_does_not_change_execution_or_queue(self):
        first_started = threading.Event()
        release_first = threading.Event()
        execution_order = []

        def executor(request, hooks):
            execution_order.append(request.user_message)
            if request.user_message == "first":
                first_started.set()
                release_first.wait(2)
            return request.user_message

        manager = self.make_manager(executor)
        first = manager.submit_message("first")
        self.assertTrue(first_started.wait(1))
        second = manager.submit_message("second")

        self.assertTrue(manager.focus_task(second.task_id).accepted)
        snapshot = manager.get_snapshot()
        self.assertEqual(snapshot.focused_task_id, second.task_id)
        self.assertEqual(snapshot.active_task_id, first.task_id)
        self.assertEqual(manager.get_task(second.task_id).state, TaskState.QUEUED)

        release_first.set()
        self.assertTrue(wait_for(lambda: manager.get_task(second.task_id).state == TaskState.COMPLETED))
        self.assertEqual(execution_order, ["first", "second"])

    def test_extracted_orchestrator_preserves_tool_call_exchange(self):
        fake_ollama = SimpleNamespace(chat=lambda **kwargs: None)
        with patch.dict("sys.modules", {"ollama": fake_ollama}):
            from brain import orchestrator

        tool_schema = {"type": "function", "function": {"name": "calculate"}}
        responses = [
            SimpleNamespace(
                message={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "function": {
                                "name": "calculate",
                                "arguments": {"expression": "2 + 2"},
                            },
                        }
                    ],
                }
            ),
            SimpleNamespace(message={"role": "assistant", "content": "4"}),
        ]
        chat_inputs = []

        def fake_chat(messages, tools=None):
            chat_inputs.append((copy.deepcopy(messages), tools))
            return responses.pop(0)

        hooks = SimpleNamespace(
            checkpoint=lambda **kwargs: None,
            update_activity=lambda activity: None,
            tool_started=lambda name: None,
            tool_finished=lambda name: None,
            request_user_input=lambda prompt, options=(): None,
            task_context=lambda: TaskContext("task", 0),
            execute_authorized=lambda operation: "4",
            request_approval=lambda name, arguments, call_id: None,
            security_gate=SecurityGate(),
            update_progress=lambda progress: None,
            update_step=lambda step_id, title, state: None,
        )
        request = TaskExecutionRequest(
            task_id="task",
            conversation_id="conversation",
            user_message="calculate 2 + 2",
            messages=[
                {"role": "system", "content": "system"},
                {"role": "user", "content": "calculate 2 + 2"},
            ],
            seen_call_ids=set(),
            base_message_count=1,
        )

        with patch.object(orchestrator, "select_tools", return_value={"calculate": tool_schema}) as select:
            with patch.object(orchestrator, "chat", side_effect=fake_chat):
                result = orchestrator.execute_task_turn(request, hooks)

        self.assertEqual(result, "4")
        select.assert_called_once()
        self.assertEqual(select.call_args.args, ("calculate 2 + 2",))
        self.assertEqual(select.call_args.kwargs["context"], request.messages)
        self.assertEqual(chat_inputs[1][0][-1]["content"], "4")
        from tools.schemas import schemas_for
        expected_tools = schemas_for(["calculate"])
        self.assertEqual(chat_inputs[0][1], expected_tools)
        self.assertEqual(chat_inputs[1][1], expected_tools)
        self.assertEqual(chat_inputs[1][0][-2]["tool_calls"][0]["id"], "call-1")
        self.assertEqual(chat_inputs[1][0][-1]["tool_call_id"], "call-1")
        self.assertEqual(chat_inputs[1][0][-1]["content"], "4")

    def test_expired_event_history_requests_resynchronization(self):
        started = threading.Event()
        release = threading.Event()

        def executor(request, hooks):
            started.set()
            release.wait(2)
            return "done"

        manager = self.make_manager(executor, event_history_limit=2)
        manager.submit_message("history expires")
        self.assertTrue(started.wait(1))
        subscription = manager.subscribe(0, manager.manager_epoch)
        self.addCleanup(subscription.close)

        result = subscription.get(timeout=1)
        self.assertIsInstance(result, ResyncRequired)
        release.set()


    def test_slow_authorized_tool_does_not_block_manager_or_snapshot_events(self):
        tool_entered = threading.Event()
        release_tool = threading.Event()
        snapshot_read = threading.Event()
        event_read = threading.Event()
        snapshot_result = {}
        tool_calls = []

        def slow_calculate(expression):
            tool_entered.set()
            if not release_tool.wait(3):
                raise TimeoutError("test tool was not released")
            tool_calls.append(expression)
            return "2"

        def executor(request, hooks):
            context = hooks.task_context()
            operation, _ = hooks.security_gate.propose(
                context, "calculate", {"expression": "1 + 1"}
            )
            authorized = hooks.security_gate.authorize(context, operation)
            hooks.tool_started("calculate")
            try:
                return hooks.execute_authorized(authorized)
            finally:
                hooks.tool_finished("calculate")

        gate = SecurityGate()
        subscription = None
        manager = None
        consumer = None
        snapshot_reader = None
        try:
            with patch.dict(
                "tools.security.TOOL_SECURITY",
                {"calculate": ToolSecurity("calculate", "low")},
            ), patch.dict("tools.registry.TOOLS", {"calculate": slow_calculate}):
                manager = TaskManager(
                    executor=executor,
                    system_prompt="system",
                    security_gate=gate,
                )
                subscription = manager.subscribe(0, manager.manager_epoch)
                queued = manager.submit_message("calculate 1 + 1")
                self.assertTrue(tool_entered.wait(1), "authorized tool did not start")

                def read_snapshot():
                    snapshot_result["snapshot"] = manager.get_snapshot()
                    snapshot_read.set()

                def read_tool_event():
                    while True:
                        event = subscription.get(timeout=2)
                        if event is None:
                            return
                        if event.event_type == "tool_started":
                            event_read.set()
                            return

                snapshot_reader = threading.Thread(target=read_snapshot, daemon=True)
                consumer = threading.Thread(target=read_tool_event, daemon=True)
                snapshot_reader.start()
                consumer.start()

                self.assertTrue(
                    snapshot_read.wait(1),
                    "snapshot waited for the blocked authorized tool",
                )
                self.assertTrue(
                    event_read.wait(1),
                    "event consumer waited for the blocked authorized tool",
                )
                self.assertEqual(
                    snapshot_result["snapshot"].active_task_id,
                    queued.task_id,
                )
                self.assertTrue(manager.focus_task(queued.task_id).accepted)
        finally:
            release_tool.set()
            if consumer is not None:
                consumer.join(1)
            if snapshot_reader is not None:
                snapshot_reader.join(1)
            if subscription is not None:
                subscription.close()
            if manager is not None:
                manager.close(wait=True)

        self.assertEqual(tool_calls, ["1 + 1"])

    def test_consumed_authorized_operation_cannot_be_reused(self):
        calls = []

        def executor(request, hooks):
            context = hooks.task_context()
            operation, _ = hooks.security_gate.propose(
                context, "calculate", {"expression": "7"}
            )
            authorized = hooks.security_gate.authorize(context, operation)
            first = hooks.execute_authorized(authorized)
            second = hooks.execute_authorized(authorized)
            return f"{first}; {second}"

        with patch.dict(
            "tools.security.TOOL_SECURITY",
            {"calculate": ToolSecurity("calculate", "low")},
        ), patch.dict(
            "tools.registry.TOOLS",
            {"calculate": lambda expression: calls.append(expression) or expression},
        ):
            manager = self.make_manager(executor, security_gate=SecurityGate())
            result = manager.submit_message("calculate 7")
            self.assertTrue(
                wait_for(
                    lambda: manager.get_task(result.task_id).state
                    == TaskState.COMPLETED
                )
            )

        self.assertEqual(calls, ["7"])
        self.assertIn("Execution denied", manager.get_task(result.task_id).result)

    def test_approval_proposal_failure_fails_task_and_invalidates_gate_record(self):
        class FailAfterProposeGate(SecurityGate):
            def __init__(self):
                super().__init__()
                self.fail_once = True
                self.failed_operation = None
                self.failed_context = None

            def propose(self, context, tool_name, arguments):
                operation, decision = super().propose(context, tool_name, arguments)
                if self.fail_once:
                    self.fail_once = False
                    self.failed_operation = operation
                    self.failed_context = context
                    raise RuntimeError("deliberate proposal failure")
                return operation, decision

        gate = FailAfterProposeGate()

        def executor(request, hooks):
            if request.user_message == "cause approval failure":
                hooks.request_approval(
                    "calculate", {"expression": "1 + 1"}, "call-approval-failure"
                )
            return "manager remains usable"

        with patch.dict(
            "tools.security.TOOL_SECURITY",
            {"calculate": ToolSecurity("calculate", "high", human_approval=True)},
        ):
            manager = self.make_manager(executor, security_gate=gate)
            start = manager.get_snapshot()
            subscription = manager.subscribe(start.last_sequence, start.manager_epoch)
            self.addCleanup(subscription.close)

            failed = manager.submit_message("cause approval failure")
            failure_event = None
            while failure_event is None:
                event = subscription.get(timeout=2)
                self.assertIsNotNone(event, "task failure event was not published")
                if (
                    event.event_type == "task_finished"
                    and event.task_id == failed.task_id
                ):
                    failure_event = event

            failed_task = manager.get_task(failed.task_id)
            self.assertEqual(failed_task.state, TaskState.FAILED)
            self.assertIsNone(failed_task.pending_approval)
            self.assertIsNone(manager._contexts[failed.task_id].pending_operation)
            self.assertEqual(
                failed_task.result,
                "Approval could not be prepared. No action was executed.",
            )
            self.assertEqual(failure_event.payload["state"], TaskState.FAILED.value)
            self.assertEqual(failure_event.payload["result"], failed_task.result)
            with self.assertRaises(AuthorizationError):
                gate.approve(gate.failed_context, gate.failed_operation)

            followup = manager.submit_message("second task")
            self.assertTrue(
                wait_for(
                    lambda: manager.get_task(followup.task_id).state
                    == TaskState.COMPLETED
                )
            )
            self.assertEqual(
                manager.get_task(followup.task_id).result,
                "manager remains usable",
            )

    def test_task_view_omits_internal_messages_prompts_and_authorization(self):
        started = threading.Event()
        release_executor = threading.Event()

        def executor(request, hooks):
            started.set()
            release_executor.wait(2)
            return "finished"

        manager = TaskManager(
            executor=executor,
            system_prompt="PRIVATE_SYSTEM_PROMPT_SENTINEL",
            security_gate=SecurityGate(),
        )
        try:
            queued = manager.submit_message("A user-visible objective")
            self.assertTrue(started.wait(1))
            request = manager._contexts[queued.task_id]
            request.messages.append(
                {"role": "assistant", "content": "PRIVATE_HISTORY_SENTINEL"}
            )
            task = manager.get_task(queued.task_id)
            request.approved_operation = AuthorizedOperation(
                "private-operation-id",
                queued.task_id,
                task.revision,
                "calculate",
                {"expression": "PRIVATE_ARGUMENT_SENTINEL"},
                "private-fingerprint",
                "PRIVATE_AUTH_TOKEN_SENTINEL",
            )

            snapshot_view = manager.get_view_snapshot()
            task_view = next(
                item for item in snapshot_view.tasks if item.task_id == queued.task_id
            )
            rendered = repr(snapshot_view)
            self.assertEqual(snapshot_view.active_task_id, queued.task_id)
            self.assertEqual(
                task_view.conversation_id,
                manager.get_task(queued.task_id).conversation_id,
            )
            self.assertEqual(task_view.objective, "A user-visible objective")
            self.assertEqual(task_view.state, TaskState.RUNNING.value)
            self.assertGreater(task_view.revision, 0)
            self.assertTrue(task_view.created_at)
            self.assertEqual(task_view.statistics.tool_calls, 0)
            self.assertFalse(hasattr(task_view, "messages"))
            self.assertFalse(hasattr(task_view, "approved_operation"))
            for private_value in (
                "PRIVATE_SYSTEM_PROMPT_SENTINEL",
                "PRIVATE_HISTORY_SENTINEL",
                "PRIVATE_ARGUMENT_SENTINEL",
                "PRIVATE_AUTH_TOKEN_SENTINEL",
            ):
                self.assertNotIn(private_value, rendered)
        finally:
            release_executor.set()
            manager.close(wait=True)

    def test_task_view_keeps_lifecycle_and_hides_approval_target(self):
        task = Task(
            task_id="task-id",
            conversation_id="conversation-id",
            state=TaskState.WAITING_FOR_APPROVAL,
            revision=12,
            objective="Review requested change",
            current_activity="Checking delete_file",
            plan=[],
            pending_approval=ApprovalRequest(
                "approval-id",
                "delete_file (destructive) — C:\\Users\\private\\secret.txt",
            ),
        )

        view = project_task(task)

        self.assertEqual(view.task_id, "task-id")
        self.assertEqual(view.state, TaskState.WAITING_FOR_APPROVAL.value)
        self.assertEqual(view.revision, 12)
        self.assertEqual(view.current_activity, "Reviewing a requested operation")
        self.assertEqual(view.pending_approval.approval_id, "approval-id")
        self.assertEqual(
            view.pending_approval.summary,
            "Sensitive operation requires approval.",
        )
        self.assertNotIn("secret.txt", repr(view))
        self.assertNotIn("delete_file", repr(view))

    def test_task_event_view_keeps_cursor_and_drops_arguments_and_tokens(self):
        event = TaskEvent(
            manager_epoch="manager-epoch",
            sequence=42,
            event_id="event-id",
            timestamp="2026-09-27T00:00:00+00:00",
            event_type="approval_requested",
            task_id="task-id",
            task_revision=9,
            payload={
                "approval_id": "approval-id",
                "operation_id": "internal-operation-id",
                "tool_name": "install_package",
                "risk": "high",
                "target": "C:\\Users\\private\\package.whl",
                "summary": "raw operation summary",
                "arguments": {"package": "PRIVATE_TOOL_ARGUMENT_SENTINEL"},
                "authorization": {"token": "PRIVATE_TOKEN_SENTINEL"},
            },
        )

        view = project_event(event)
        rendered = repr(view)

        self.assertEqual(view.manager_epoch, "manager-epoch")
        self.assertEqual(view.sequence, 42)
        self.assertEqual(view.event_id, "event-id")
        self.assertEqual(view.task_id, "task-id")
        self.assertEqual(view.task_revision, 9)
        self.assertEqual(
            view.payload["summary"],
            "Sensitive operation requires approval.",
        )
        for private_value in (
            "internal-operation-id",
            "install_package",
            "package.whl",
            "PRIVATE_TOOL_ARGUMENT_SENTINEL",
            "PRIVATE_TOKEN_SENTINEL",
        ):
            self.assertNotIn(private_value, rendered)

    def test_expected_revision_can_reject_stale_state_command(self):
        started = threading.Event()
        release_first = threading.Event()

        def executor(request, hooks):
            if request.user_message == "first":
                started.set()
                release_first.wait(2)
            return "done"

        manager = self.make_manager(executor)
        first = manager.submit_message("first")
        self.assertTrue(started.wait(1))
        queued = manager.submit_message("queued")
        stale_revision = manager.get_task(queued.task_id).revision

        self.assertTrue(manager.pause_task(queued.task_id).accepted)
        rejected = manager.resume_task(
            queued.task_id, expected_revision=stale_revision
        )
        self.assertFalse(rejected.accepted)
        self.assertEqual(manager.get_task(queued.task_id).state, TaskState.PAUSED)

        current_revision = manager.get_task(queued.task_id).revision
        self.assertTrue(
            manager.resume_task(
                queued.task_id, expected_revision=current_revision
            ).accepted
        )
        release_first.set()
        self.assertTrue(
            wait_for(
                lambda: manager.get_task(queued.task_id).state
                == TaskState.COMPLETED
            )
        )



if __name__ == "__main__":
    unittest.main()
