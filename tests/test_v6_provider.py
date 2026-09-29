import json
from pathlib import Path
import sys
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from brain.v6_format import parse_model_tool_calls
from brain.v6_provider import ADAPTER_CONFIG_PATH, V6Provider
from tools.schemas import schemas_for


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeEvaluation:
    def __init__(self, responses=()):
        self.responses = list(responses)
        self.calls = []

    def generate_response(self, model, backend, messages, tools=None):
        self.calls.append((messages, tools))
        return self.responses.pop(0)


def make_provider(responses=()):
    evaluation = FakeEvaluation(responses)
    model = object()
    backend = object()
    loader = Mock(return_value=(model, backend, evaluation))
    return V6Provider(component_loader=loader), loader, evaluation, model, backend


class V6ProviderTests(unittest.TestCase):
    def test_ollama_provider_remains_selectable_as_fallback(self):
        from brain import router

        expected = SimpleNamespace(message={"role": "assistant", "content": "fallback"})
        fake_ollama = SimpleNamespace(chat=Mock(return_value=expected))
        with patch.dict(sys.modules, {"ollama": fake_ollama}), patch.object(router, "PROVIDER", "local"):
            response = router.chat([{"role": "user", "content": "Hello"}], tools=[])
        self.assertIs(response, expected)
        fake_ollama.chat.assert_called_once()

    def test_provider_uses_existing_adapter_configuration_and_mock_loader(self):
        self.assertTrue(ADAPTER_CONFIG_PATH.is_file())
        config = json.loads(ADAPTER_CONFIG_PATH.read_text(encoding="utf-8"))
        provider, loader, _, model, backend = make_provider()
        self.assertEqual(provider.base_model_name, config["base_model_name_or_path"])
        self.assertEqual(provider.base_model_name, "mistralai/Ministral-3-8B-Reasoning-2512")

        provider._ensure_loaded()

        loader.assert_called_once_with()
        self.assertIs(provider._model, model)
        self.assertIs(provider._backend, backend)
        self.assertTrue(provider.loaded)

    def test_normal_response_becomes_orchestrator_message(self):
        provider, _, _, _, _ = make_provider(["Hello from V6.</s>"])

        response = provider.chat([{"role": "user", "content": "Hello"}], tools=[])

        self.assertEqual(response.message, {"role": "assistant", "content": "Hello from V6."})

    def test_tool_call_preserves_name_arguments_and_supplied_id(self):
        raw = '<tool_call>{"id":"call_123","name":"search_files","arguments":{"query":"needle"}}</tool_call></s>'
        provider, _, _, _, _ = make_provider([raw])

        response = provider.chat([], tools=schemas_for({"search_files"}))

        self.assertEqual(response.message["role"], "assistant")
        call = response.message["tool_calls"][0]
        self.assertEqual(call["id"], "call_123")
        self.assertEqual(call["type"], "function")
        self.assertEqual(call["function"]["name"], "search_files")
        self.assertEqual(call["function"]["arguments"], {"query": "needle"})

    def test_native_tool_call_without_id_gets_unique_internal_id(self):
        provider, _, _, _, _ = make_provider([
            '[TOOL_CALLS]read_file[ARGS]{"path":"tools/file_reader.py"}</s>',
            '[TOOL_CALLS]read_file[ARGS]{"path":"tools/security.py"}</s>',
        ])

        first = provider.chat([], tools=[]).message["tool_calls"][0]
        second = provider.chat([], tools=[]).message["tool_calls"][0]

        self.assertEqual(first["id"], "v6_call_1")
        self.assertEqual(second["id"], "v6_call_2")

    def test_tool_result_round_trip_keeps_call_id_and_uses_given_schemas(self):
        provider, _, evaluation, _, _ = make_provider(["Done."])
        call_id = "call_round_trip"
        messages = [
            {"role": "user", "content": "Search for a marker."},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {"name": "search_files", "arguments": {"query": "marker"}},
            }]},
            {"role": "tool", "tool_name": "search_files", "tool_call_id": call_id, "content": "tools/a.py:1 marker"},
        ]
        canonical_tools = schemas_for({"search_files"})

        response = provider.chat(messages, tools=canonical_tools)

        self.assertEqual(response.message["content"], "Done.")
        sent_messages, sent_tools = evaluation.calls[0]
        self.assertEqual(sent_messages[1]["tool_calls"][0]["id"], call_id)
        self.assertEqual(sent_messages[2]["tool_call_id"], call_id)
        self.assertEqual(sent_tools, canonical_tools)

    def test_evaluator_and_runtime_use_shared_native_call_parser(self):
        parsed = parse_model_tool_calls(
            '<tool_call>{"id":"call_keep","name":"calculate","arguments":{"expression":"2 + 2"}}</tool_call>'
        )
        self.assertEqual(parsed, [{
            "id": "call_keep",
            "name": "calculate",
            "arguments": {"expression": "2 + 2"},
        }])

    def test_model_tool_request_still_passes_security_gate(self):
        try:
            import ollama  # noqa: F401
        except ImportError:
            with patch.dict(sys.modules, {"ollama": SimpleNamespace(chat=lambda **kwargs: None)}):
                from brain import orchestrator, router
        else:
            from brain import orchestrator, router

        protected_path = "model/evaluations/mistral_v6_results.json"
        evaluation = FakeEvaluation([
            f'[TOOL_CALLS]read_file[ARGS]{{"path":"{protected_path}"}}</s>',
            "That model evaluation file is not accessible through project tools.</s>",
        ])
        provider = V6Provider(component_loader=lambda: (object(), object(), evaluation))
        original_read = Mock(side_effect=AssertionError("protected file read reached executor"))

        def completed(manager, task_id, timeout=4):
            end = time.monotonic() + timeout
            while time.monotonic() < end:
                task = manager.get_task(task_id)
                if task and task.state.value in {"completed", "failed", "cancelled"}:
                    return task
                time.sleep(0.02)
            return manager.get_task(task_id)

        from tools import runner
        with patch.object(router, "PROVIDER", "v6"), patch.object(router, "_v6_provider", provider), \
             patch.dict(runner.TOOLS, {"read_file": original_read}):
            from tasks.manager import TaskManager
            from tasks.models import TaskState
            from tools.security import SecurityGate

            manager = TaskManager(
                executor=orchestrator.execute_task_turn,
                security_gate=SecurityGate(PROJECT_ROOT),
                max_concurrent_tasks=1,
            )
            try:
                command = manager.submit_message("Read tools/file_reader.py and explain it.")
                self.assertTrue(command.accepted)
                task = completed(manager, command.task_id)
                self.assertEqual(task.state, TaskState.COMPLETED)
                self.assertIn("not accessible", task.result)
                original_read.assert_not_called()
            finally:
                manager.close(wait=True)


if __name__ == "__main__":
    unittest.main()
