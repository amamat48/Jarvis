import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from brain import orchestrator
from tasks.manager import TaskManager
from tasks.models import TaskState
from tools import runner
from tools.registry import select_tools
from tools.security import SecurityGate


def wait_for_task(manager, task_id, timeout=8.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        task = manager.get_task(task_id)
        if task and task.state in {TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED}:
            return task
        time.sleep(0.02)
    return manager.get_task(task_id)


class AuditedSecurityGate(SecurityGate):
    def __init__(self):
        super().__init__()
        self.authorized_tools = []

    def authorize(self, context, operation, grant=None):
        authorized = super().authorize(context, operation, grant)
        self.authorized_tools.append(authorized.tool_name)
        return authorized


class ToolSelectionTests(unittest.TestCase):
    def test_natural_language_tool_routing(self):
        cases = {
            "What files are in the brain directory?": {"list_files"},
            "List the Python files in brain.": {"list_files"},
            "Show me what's inside the brain folder.": {"list_files"},
            "Find the file that handles routing.": {"list_files", "read_file"},
            "Read brain/router.py.": {"read_file"},
            "What is 15% of 240?": {"calculate"},
            "Calculate 1234 * 567.": {"calculate"},
            "Remember that I prefer concise responses.": {"remember_memory"},
            "What do you remember about my programming language preference?": {"recall_memory"},
            "Debug brain/router.py.": {"debug_python_file"},
            "What could be wrong with this Python file?": set(),
            "Search the web for the latest information about X.": {"web_search"},
            "What is a mutex?": set(),
            "Explain how TCP differs from UDP.": set(),
            "What is a Python class?": set(),
        }
        for request, expected in cases.items():
            with self.subTest(request=request):
                self.assertEqual(set(select_tools(request)), expected)

    def test_debug_request_without_a_path_waits_for_clarification(self):
        self.assertNotIn("debug_python_file", select_tools("Debug this Python file."))

    def test_discovery_requests_expose_read_file_for_the_dependent_step(self):
        self.assertIn("read_file", select_tools("Find the Python file that handles V6 routing."))
        self.assertIn("read_file", select_tools("Find the file that handles the V6 provider."))

    def test_follow_up_can_read_a_path_returned_by_prior_listing(self):
        context = [{
            "role": "tool",
            "tool_name": "list_files",
            "tool_call_id": "listing-call",
            "content": "brain/router.py\nbrain/v6_provider.py",
        }]
        self.assertEqual(set(select_tools("Read the file you found.", context=context)), {"read_file"})

    def test_missing_tool_call_gets_one_retry_then_safe_fallback(self):
        responses = [
            {"role": "assistant", "content": "I listed the files in brain."},
            {"role": "assistant", "content": "I listed the files in brain."},
        ]
        chats = []

        def fake_chat(messages, tools=None):
            chats.append((messages, tools))
            return SimpleNamespace(message=responses.pop(0))

        with patch.object(orchestrator, "chat", side_effect=fake_chat):
            manager = TaskManager(executor=orchestrator.execute_task_turn, security_gate=AuditedSecurityGate())
            try:
                submitted = manager.submit_message("What files are in the brain directory?")
                task = wait_for_task(manager, submitted.task_id)
            finally:
                manager.close()

        self.assertEqual(task.state, TaskState.COMPLETED)
        self.assertEqual(len(chats), 2)
        self.assertIn("no tool was executed", task.result.lower())
        self.assertNotIn("listed the files", task.result.lower())

    def test_unsupported_action_claim_is_removed_even_when_selector_misses(self):
        def fake_chat(messages, tools=None):
            return SimpleNamespace(message={"role": "assistant", "content": "I listed the files in brain."})

        with patch.object(orchestrator, "chat", side_effect=fake_chat):
            manager = TaskManager(executor=orchestrator.execute_task_turn, security_gate=AuditedSecurityGate())
            try:
                submitted = manager.submit_message("Could you tell me what exists under brain?")
                task = wait_for_task(manager, submitted.task_id)
            finally:
                manager.close()

        self.assertEqual(task.state, TaskState.COMPLETED)
        self.assertIn("no matching tool succeeded", task.result.lower())
        self.assertNotIn("listed the files", task.result.lower())

    def test_action_claim_must_match_a_successful_tool(self):
        outcomes = [("list_files", True, "brain/router.py")]
        self.assertTrue(orchestrator._has_unsupported_action_claim("I calculated the result.", outcomes))
        self.assertFalse(orchestrator._has_unsupported_action_claim("I listed the files.", outcomes))

    def test_retry_executes_real_read_only_tool_and_returns_result_to_model(self):
        responses = [
            {"role": "assistant", "content": "I listed the files in brain."},
            {"role": "assistant", "tool_calls": [{
                "id": "list-call",
                "type": "function",
                "function": {"name": "list_files", "arguments": {}},
            }]},
            {"role": "assistant", "content": "The brain directory contains the available Python modules."},
        ]
        chats = []
        execution_count = []
        gate = AuditedSecurityGate()
        actual_list_files = runner.TOOLS["list_files"]

        def tracked_list_files():
            execution_count.append(True)
            return actual_list_files()

        def fake_chat(messages, tools=None):
            chats.append((messages, tools))
            return SimpleNamespace(message=responses.pop(0))

        with patch.object(orchestrator, "chat", side_effect=fake_chat), patch.dict(
            runner.TOOLS, {"list_files": tracked_list_files}
        ):
            manager = TaskManager(executor=orchestrator.execute_task_turn, security_gate=gate)
            try:
                submitted = manager.submit_message("What files are in the brain directory?")
                task = wait_for_task(manager, submitted.task_id)
            finally:
                manager.close()

        self.assertEqual(task.state, TaskState.COMPLETED)
        self.assertEqual(gate.authorized_tools, ["list_files"])
        self.assertEqual(len(execution_count), 1)
        self.assertEqual(len(chats), 3)
        returned_messages = chats[-1][0]
        tool_results = [message for message in returned_messages if message.get("role") == "tool"]
        self.assertEqual(len(tool_results), 1)
        self.assertIn("brain\\router.py", tool_results[0]["content"])
        self.assertIn("available Python modules", task.result)

    def test_failed_tool_result_cannot_be_reported_as_completed_action(self):
        responses = [
            {"role": "assistant", "tool_calls": [{
                "id": "list-call",
                "type": "function",
                "function": {"name": "list_files", "arguments": {}},
            }]},
            {"role": "assistant", "content": "I listed the files in brain."},
        ]
        gate = AuditedSecurityGate()

        def fake_chat(messages, tools=None):
            return SimpleNamespace(message=responses.pop(0))

        with patch.object(orchestrator, "chat", side_effect=fake_chat), patch.dict(
            runner.TOOLS,
            {"list_files": lambda: "Tool 'list_files' failed: simulated failure"},
        ):
            manager = TaskManager(executor=orchestrator.execute_task_turn, security_gate=gate)
            try:
                submitted = manager.submit_message("What files are in the brain directory?")
                task = wait_for_task(manager, submitted.task_id)
            finally:
                manager.close()

        self.assertEqual(task.state, TaskState.COMPLETED)
        self.assertEqual(gate.authorized_tools, ["list_files"])
        self.assertIn("tool did not succeed", task.result.lower())
        self.assertNotIn("i listed the files", task.result.lower())

    def test_unexposed_tool_call_is_blocked_before_security_gate_and_retry_can_succeed(self):
        responses = [
            {"role": "assistant", "tool_calls": [{
                "id": "hidden-search",
                "type": "function",
                "function": {"name": "search_files", "arguments": {"query": "V6 routing"}},
            }]},
            {"role": "assistant", "tool_calls": [{
                "id": "read-router",
                "type": "function",
                "function": {"name": "read_file", "arguments": {"path": "brain/router.py"}},
            }]},
            {"role": "assistant", "content": "router.py dispatches chat to the configured provider."},
        ]
        chats = []
        executed = []
        gate = AuditedSecurityGate()

        def fake_chat(messages, tools=None):
            chats.append((messages, tools))
            return SimpleNamespace(message=responses.pop(0))

        def fake_read(path):
            executed.append(path)
            return "def chat(messages, tools=None): pass"

        with patch.object(orchestrator, "chat", side_effect=fake_chat), patch.dict(
            runner.TOOLS, {"read_file": fake_read, "search_files": lambda query: self.fail("hidden tool ran")}
        ):
            manager = TaskManager(executor=orchestrator.execute_task_turn, security_gate=gate)
            try:
                submitted = manager.submit_message(
                    "Find the Python file that handles V6 routing and tell me what it does."
                )
                task = wait_for_task(manager, submitted.task_id)
            finally:
                manager.close()

        self.assertEqual(task.state, TaskState.COMPLETED)
        self.assertEqual(executed, ["brain/router.py"])
        self.assertEqual(gate.authorized_tools, ["read_file"])
        self.assertNotIn("search_files", {schema["function"]["name"] for schema in chats[0][1]})
        self.assertTrue(any("not available for this request" in message.get("content", "")
                            for message in chats[1][0] if message.get("role") == "tool"))
        self.assertIn("dispatches chat", task.result)


if __name__ == "__main__":
    unittest.main()
