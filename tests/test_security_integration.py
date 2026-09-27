import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tasks.manager import TaskManager
from tasks.models import TaskState
from tools import runner
from tools.security import (
    ApprovalRequired,
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


def tool_call(name, arguments):
    return SimpleNamespace(message={
        "role": "assistant", "content": None,
        "tool_calls": [{"id": "call-1", "function": {"name": name, "arguments": arguments}}],
    })


class SecurityIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.gate = SecurityGate()

    def tearDown(self):
        self.temp.cleanup()

    def propose_authorize(self, name, arguments, task="safe", revision=1, grant=None):
        context = TaskContext(task, revision)
        operation, decision = self.gate.propose(context, name, arguments)
        if decision.requires_approval and grant is None:
            raise ApprovalRequired(decision)
        authorized = self.gate.authorize(context, operation, grant)
        return context, operation, authorized

    def test_calculate_and_repository_search_execute_through_gate(self):
        context, _, authorized = self.propose_authorize("calculate", {"expression": "2 + 2"})
        self.assertEqual(runner.execute(context, authorized, gate=self.gate), "4")

        context, _, authorized = self.propose_authorize("search_files", {"query": "def calculate"}, task="search")
        result = runner.execute(context, authorized, gate=self.gate)
        self.assertRegex(result, r"tools[\\/]calculator\.py")

    def test_secret_like_path_is_denied(self):
        context = TaskContext("secret", 1)
        with self.assertRaises(AuthorizationError):
            self.gate.propose(context, "read_file", {"path": ".env"})
        self.assertIn("Access denied", __import__("tools.file_reader", fromlist=["read_file"]).read_file(".env"))

    def test_approved_operation_executes_exactly_once(self):
        calls = []
        metadata = ToolSecurity("test_approved", "high", "none", human_approval=True)
        context = TaskContext("approve-once", 4)
        with patch.dict("tools.security.TOOL_SECURITY", {"test_approved": metadata}):
            with patch.dict(runner.TOOLS, {"test_approved": lambda value: calls.append(value) or value}):
                operation, decision = self.gate.propose(context, "test_approved", {"value": "approved"})
                self.assertTrue(decision.requires_approval)
                grant = self.gate.approve(context, operation)
                authorized = self.gate.authorize(context, operation, grant)
                self.assertEqual(runner.execute(context, authorized, gate=self.gate), "approved")
                replay = runner.execute(context, authorized, gate=self.gate)
        self.assertIn("denied", replay.lower())
        self.assertEqual(calls, ["approved"])

    def test_changed_arguments_after_approval_are_denied(self):
        context = TaskContext("mutated", 2)
        metadata = ToolSecurity("test_approved", "high", human_approval=True)
        with patch.dict("tools.security.TOOL_SECURITY", {"test_approved": metadata}):
            with patch.dict(runner.TOOLS, {"test_approved": lambda value: value}):
                operation, _ = self.gate.propose(context, "test_approved", {"value": "original"})
                grant = self.gate.approve(context, operation)
                operation.arguments["value"] = "changed"
                with self.assertRaises(AuthorizationError):
                    self.gate.authorize(context, operation, grant)

    def test_changed_revision_invalidates_approval(self):
        context = TaskContext("stale", 2)
        metadata = ToolSecurity("test_approved", "high", human_approval=True)
        with patch.dict("tools.security.TOOL_SECURITY", {"test_approved": metadata}):
            with patch.dict(runner.TOOLS, {"test_approved": lambda value: value}):
                operation, _ = self.gate.propose(context, "test_approved", {"value": "same"})
                grant = self.gate.approve(context, operation)
                with self.assertRaises(AuthorizationError):
                    self.gate.authorize(TaskContext("stale", 3), operation, grant)

    def test_revision_change_immediately_before_dispatch_is_denied(self):
        context, _, authorized = self.propose_authorize(
            "calculate", {"expression": "9"}, task="dispatch-stale", revision=6
        )
        result = runner.execute(
            TaskContext(context.task_id, context.revision + 1), authorized, gate=self.gate
        )
        self.assertIn("denied", result.lower())

    def test_cancelled_task_cannot_execute_pending_authorization(self):
        context, _, authorized = self.propose_authorize("calculate", {"expression": "1+1"}, task="cancel")
        self.gate.cancel_task(context.task_id)
        result = runner.execute(context, authorized, gate=self.gate)
        self.assertIn("denied", result.lower())

    def test_unknown_tool_is_denied(self):
        with self.assertRaises(AuthorizationError):
            self.gate.propose(TaskContext("unknown", 1), "not_registered", {})

    def test_python_is_denied_even_after_approval(self):
        context = TaskContext("python", 1)
        operation, decision = self.gate.propose(context, "run_python_file", {"path": "tools/calculator.py"})
        self.assertTrue(decision.requires_approval)
        grant = self.gate.approve(context, operation)
        authorized = self.gate.authorize(context, operation, grant)
        result = runner.execute(context, authorized, gate=self.gate)
        self.assertIn("sandbox", result.lower())

    def test_executor_checks_gate_before_dispatch(self):
        order = []

        class OrderedGate(SecurityGate):
            def consume(self, context, authorized):
                order.append("authorized")
                return super().consume(context, authorized)

        gate = OrderedGate()
        context = TaskContext("order", 1)
        operation, _ = gate.propose(context, "calculate", {"expression": "1"})
        authorized = gate.authorize(context, operation)
        metadata = ToolSecurity("calculate", "low")
        with patch.dict("tools.security.TOOL_SECURITY", {"calculate": metadata}):
            with patch.dict(runner.TOOLS, {"calculate": lambda expression: order.append("dispatch") or expression}):
                runner.execute(context, authorized, gate=gate)
        self.assertEqual(order, ["authorized", "dispatch"])

    def _manager_for_tool(self, tool_name, tool_arguments, function, gate=None):
        fake_ollama = SimpleNamespace(chat=lambda **kwargs: None)
        with patch.dict("sys.modules", {"ollama": fake_ollama}):
            from brain import orchestrator

        tool_schema = {"type": "function", "function": {"name": tool_name}}
        responses = [tool_call(tool_name, tool_arguments), SimpleNamespace(message={"role": "assistant", "content": "finished"})]
        manager = TaskManager(
            executor=orchestrator.execute_task_turn,
            system_prompt="system",
            security_gate=gate or self.gate,
        )
        self.addCleanup(manager.close)
        patches = [
            patch.object(orchestrator, "select_tools", return_value={tool_name: tool_schema}),
            patch.object(orchestrator, "chat", side_effect=lambda *args, **kwargs: responses.pop(0)),
            patch.dict(runner.TOOLS, {tool_name: function}),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        return manager

    def test_task_manager_approval_flow_executes_exact_call_once(self):
        calls = []
        metadata = ToolSecurity("test_approved", "high", human_approval=True)
        with patch.dict("tools.security.TOOL_SECURITY", {"test_approved": metadata}):
            manager = self._manager_for_tool("test_approved", {"value": "one"}, lambda value: calls.append(value) or "done")
            start_sequence = manager.get_snapshot().last_sequence
            subscription = manager.subscribe(start_sequence, manager.manager_epoch)
            self.addCleanup(subscription.close)
            queued = manager.submit_message("perform approved operation")
            self.assertTrue(wait_for(lambda: manager.get_task(queued.task_id).state == TaskState.WAITING_FOR_APPROVAL))
            task = manager.get_task(queued.task_id)
            approval_id = task.pending_approval.approval_id
            events = []
            while True:
                event = subscription.get(timeout=0.01)
                if event is None:
                    break
                events.append(event)
            approval_event = next(e for e in events if e.event_type == "approval_requested")
            self.assertNotIn("arguments", approval_event.payload)
            self.assertNotIn("value", approval_event.payload)

            approved = manager.approve_action(
                task.task_id, approval_id, expected_revision=task.revision
            )
            self.assertTrue(approved.accepted)
            self.assertTrue(wait_for(lambda: manager.get_task(task.task_id).state == TaskState.COMPLETED), manager.get_task(task.task_id))
            self.assertEqual(calls, ["one"])
            replay = manager.approve_action(
                task.task_id, approval_id, expected_revision=manager.get_task(task.task_id).revision
            )
            self.assertFalse(replay.accepted)

    def test_task_manager_reject_stale_and_rejected_approval(self):
        calls = []
        metadata = ToolSecurity("test_approved", "high", human_approval=True)
        with patch.dict("tools.security.TOOL_SECURITY", {"test_approved": metadata}):
            manager = self._manager_for_tool("test_approved", {"value": "no"}, lambda value: calls.append(value) or "bad")
            queued = manager.submit_message("ask for approval")
            self.assertTrue(wait_for(lambda: manager.get_task(queued.task_id).state == TaskState.WAITING_FOR_APPROVAL))
            task = manager.get_task(queued.task_id)
            approval_id = task.pending_approval.approval_id
            stale = manager.approve_action(task.task_id, approval_id, expected_revision=task.revision - 1)
            self.assertFalse(stale.accepted)
            rejected = manager.reject_action(task.task_id, approval_id, expected_revision=task.revision)
            self.assertTrue(rejected.accepted)
            self.assertTrue(wait_for(lambda: manager.get_task(task.task_id).state == TaskState.COMPLETED), manager.get_task(task.task_id))
            self.assertEqual(calls, [])

    def test_cancelled_pending_approval_cannot_be_approved(self):
        metadata = ToolSecurity("test_approved", "high", human_approval=True)
        with patch.dict("tools.security.TOOL_SECURITY", {"test_approved": metadata}):
            manager = self._manager_for_tool("test_approved", {"value": "cancel"}, lambda value: value)
            queued = manager.submit_message("request then cancel")
            self.assertTrue(wait_for(lambda: manager.get_task(queued.task_id).state == TaskState.WAITING_FOR_APPROVAL))
            task = manager.get_task(queued.task_id)
            approval_id = task.pending_approval.approval_id
            self.assertTrue(manager.cancel_task(task.task_id).accepted)
            self.assertEqual(manager.get_task(task.task_id).state, TaskState.CANCELLED)
            result = manager.approve_action(
                task.task_id, approval_id, expected_revision=manager.get_task(task.task_id).revision
            )
            self.assertFalse(result.accepted)

    def test_task_manager_has_no_name_arguments_dispatch_bypass(self):
        fake_ollama = SimpleNamespace(chat=lambda **kwargs: None)
        with patch.dict("sys.modules", {"ollama": fake_ollama}):
            from brain import orchestrator
        self.assertFalse(hasattr(runner, "execute_tool"))
        self.assertFalse(hasattr(orchestrator, "execute_tool"))


if __name__ == "__main__":
    unittest.main()
