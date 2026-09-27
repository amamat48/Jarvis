import copy
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tasks.manager import TaskExecutionRequest, TaskManager
from tasks.models import ResyncRequired, TaskCommand, TaskState
from tools.security import SecurityGate, TaskContext


def wait_for(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class TaskManagerTests(unittest.TestCase):
    def make_manager(self, executor, **kwargs):
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
        select.assert_called_once_with("calculate 2 + 2")
        self.assertEqual(chat_inputs[1][0][-1]["content"], "4")
        self.assertEqual(chat_inputs[0][1], [tool_schema])
        self.assertEqual(chat_inputs[1][1], [tool_schema])
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


if __name__ == "__main__":
    unittest.main()
