import sys
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tasks.manager import TaskManager
from tasks.models import TaskState
from tools.security import SecurityGate


def wait_for(predicate, timeout=4.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


class RuntimeIntegrationTests(unittest.TestCase):
    def test_multi_step_search_then_read_uses_real_tools_and_paired_ids(self):
        with patch.dict("sys.modules", {"ollama": SimpleNamespace(chat=lambda **kwargs: None)}):
            from brain import orchestrator
        responses = [
            {"role":"assistant","tool_calls":[{"id":"search-call","function":{"name":"search_files","arguments":{"query":"from tools.calculator import calculate"}}}]},
            {"role":"assistant","tool_calls":[{"id":"read-call","function":{"name":"read_file","arguments":{"path":"tools/calculator.py"}}}]},
            {"role":"assistant","content":"The registry imports calculate from tools.calculator, whose source evaluates the supplied expression and returns a string."},
        ]
        observed = []
        def fake_chat(messages, tools=None):
            observed.append((messages, tools))
            return SimpleNamespace(message=responses.pop(0))
        with patch.object(orchestrator, "chat", side_effect=fake_chat):
            manager = TaskManager(executor=orchestrator.execute_task_turn, security_gate=SecurityGate(), max_concurrent_tasks=1)
            self.addCleanup(manager.close)
            result = manager.submit_message("Find where the calculate function is imported, then read tools/calculator.py source code.")
            self.assertTrue(result.accepted)
            self.assertTrue(wait_for(lambda: manager.get_task(result.task_id).state == TaskState.COMPLETED))
        task = manager.get_task(result.task_id)
        self.assertIn("registry imports calculate", task.result)
        self.assertEqual(len(responses), 0)
        exchange = observed[-1][0]
        tool_messages = [item for item in exchange if item.get("role") == "tool"]
        self.assertEqual([item["tool_call_id"] for item in tool_messages], ["search-call", "read-call"])
        self.assertIn("def calculate", tool_messages[-1]["content"])
        self.assertTrue(observed[0][1])
        self.assertIn("search_files", [item["function"]["name"] for item in observed[0][1]])
        self.assertIn("read_file", [item["function"]["name"] for item in observed[0][1]])

    def test_real_task_executor_failure_is_reported_as_failed_not_completed(self):
        with patch.dict("sys.modules", {"ollama": SimpleNamespace(chat=lambda **kwargs: None)}):
            from brain import orchestrator
        with patch.object(orchestrator, "chat", side_effect=RuntimeError("local model unavailable")):
            manager = TaskManager(executor=orchestrator.execute_task_turn, security_gate=SecurityGate(), max_concurrent_tasks=1)
            self.addCleanup(manager.close)
            task = manager.submit_message("Hello Jarvis.")
            self.assertTrue(wait_for(lambda: manager.get_task(task.task_id).state == TaskState.FAILED))
            self.assertIn("local model unavailable", manager.get_task(task.task_id).error)


if __name__ == "__main__":
    unittest.main()
