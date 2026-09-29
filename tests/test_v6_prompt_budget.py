import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from brain.prompt import SYSTEM_PROMPT
from model.evaluations import mistral_v4_evaluation as evaluation
from tools.schemas import schemas_for


class FakeBackend:
    def __init__(self):
        self.calls = []

    @staticmethod
    def render(messages, tools):
        parts = []
        for index, message in enumerate(messages):
            if index == 0 and tools:
                parts.append("[TOOLS]" + json.dumps(tools, sort_keys=True))
            role = message.get("role")
            content = message.get("content") or ""
            if role == "system":
                parts.append("[SYSTEM]" + content + "[/SYSTEM]")
            elif role == "user":
                parts.append("[INST]" + content + "[/INST]")
            else:
                parts.append("[" + str(role) + "]" + str(content))
        return "".join(parts)

    def apply_chat_template(
        self,
        messages,
        *,
        tools,
        add_generation_prompt,
        tokenize,
        truncation,
        return_tensors,
        return_dict,
    ):
        self.calls.append({
            "messages": messages,
            "tools": tools,
            "add_generation_prompt": add_generation_prompt,
            "truncation": truncation,
        })
        rendered = self.render(messages, tools)
        ids = torch.tensor([[ord(char) for char in rendered]], dtype=torch.long)
        return {"input_ids": ids}

    @staticmethod
    def decode(input_ids):
        return "".join(chr(token) for token in input_ids[0].tolist())


class V6PromptBudgetTests(unittest.TestCase):
    def setUp(self):
        self.backend = FakeBackend()
        self.system = {"role": "system", "content": "Keep the security policy."}

    def test_short_user_message_and_native_boundary_survive(self):
        messages = [self.system, {"role": "user", "content": "Hello JARVIS."}]

        encoded = evaluation._encode_prompt_with_budget(
            self.backend, messages, [], max_input_tokens=1024
        )

        rendered = self.backend.decode(encoded["input_ids"])
        self.assertIn("Hello JARVIS.", rendered)
        self.assertTrue(rendered.endswith("[/INST]"))
        self.assertFalse(self.backend.calls[-1]["truncation"])

    def test_long_history_is_trimmed_before_newest_user_turn(self):
        latest = {"role": "user", "content": "Newest request survives."}
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "old request " * 80},
            {"role": "assistant", "content": "old response " * 80},
            latest,
        ]
        protected_size = len(self.backend.render([messages[0], latest], []))

        encoded = evaluation._encode_prompt_with_budget(
            self.backend, messages, [], max_input_tokens=protected_size
        )

        rendered = self.backend.decode(encoded["input_ids"])
        self.assertIn(SYSTEM_PROMPT, rendered)
        self.assertIn(latest["content"], rendered)
        self.assertNotIn("old request", rendered)
        self.assertTrue(rendered.endswith("[/INST]"))
        self.assertLessEqual(encoded["input_ids"].shape[1], protected_size)

    def test_tool_schema_causes_old_turn_to_be_trimmed_not_latest_user(self):
        latest = {"role": "user", "content": "Find the requested source symbol."}
        tools = schemas_for({"search_files"})
        messages = [
            self.system,
            {"role": "user", "content": "old request " * 80},
            {"role": "assistant", "content": "old answer " * 80},
            latest,
        ]
        protected_size = len(self.backend.render([self.system, latest], tools))

        encoded = evaluation._encode_prompt_with_budget(
            self.backend, messages, tools, max_input_tokens=protected_size
        )

        rendered = self.backend.decode(encoded["input_ids"])
        self.assertIn(latest["content"], rendered)
        self.assertIn('"name": "search_files"', rendered)
        self.assertNotIn("old request", rendered)
        self.assertTrue(rendered.endswith("[/INST]"))
        self.assertEqual(self.backend.calls[-1]["tools"], tools)

    def test_protected_prompt_overflow_fails_instead_of_silently_truncating(self):
        messages = [self.system, {"role": "user", "content": "Keep this request."}]
        required_size = len(self.backend.render(messages, []))

        with self.assertRaisesRegex(ValueError, "newest user turn exceed"):
            evaluation._encode_prompt_with_budget(
                self.backend, messages, [], max_input_tokens=required_size - 1
            )

    def test_input_budget_comes_from_model_context_and_reserves_generation(self):
        model = SimpleNamespace(
            config=SimpleNamespace(
                text_config=SimpleNamespace(max_position_embeddings=262144)
            ),
            generation_config=SimpleNamespace(max_length=262144),
        )

        self.assertEqual(
            evaluation._get_input_token_limit(model),
            262144 - evaluation.MAX_NEW_TOKENS,
        )


class EvaluatorToolSchemaTests(unittest.TestCase):
    def test_main_passes_each_examples_tools_to_generation(self):
        tools = schemas_for({"search_files"})
        example = {
            "messages": [
                {"role": "user", "content": "Find a definition."},
                {"role": "assistant", "content": "Found it."},
            ],
            "tools": tools,
        }
        model = object()
        backend = object()
        with tempfile.TemporaryDirectory() as temp_dir:
            results_path = Path(temp_dir) / "evaluation.json"
            with (
                patch.object(evaluation, "load_evaluation_data", return_value=[example]),
                patch.object(evaluation, "load_model", return_value=model),
                patch.object(evaluation.MistralCommonBackend, "from_pretrained", return_value=backend),
                patch.object(evaluation, "generate_response", return_value="Found it.") as generate,
                patch.object(evaluation, "RESULTS_FILE", results_path),
                redirect_stdout(io.StringIO()),
            ):
                evaluation.main()

        generate.assert_called_once_with(
            model,
            backend,
            [example["messages"][0]],
            tools=tools,
        )


if __name__ == "__main__":
    unittest.main()
