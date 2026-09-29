import json
import unittest
from collections import Counter
from pathlib import Path

from tools.schemas import TOOLS_SCHEMA

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "model" / "training_data" / "v5" / "train.jsonl"


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class V5DatasetTests(unittest.TestCase):
    def test_v5_records_have_valid_schemas_and_tool_result_associations(self):
        records = load_jsonl(DATASET)
        canonical = {item["function"]["name"]: item["function"]["parameters"] for item in TOOLS_SCHEMA}
        for record_number, record in enumerate(records, 1):
            self.assertIsInstance(record.get("messages"), list, record_number)
            advertised = {
                item["function"]["name"]: item["function"]["parameters"]
                for item in record.get("tools", [])
            }
            for name in advertised:
                self.assertIn(name, canonical, (record_number, name))
                self.assertEqual(advertised[name], canonical[name], (record_number, name))
            pending = {}
            seen = set()
            for index, message in enumerate(record["messages"]):
                role = message.get("role")
                if role == "assistant":
                    for call in message.get("tool_calls", []):
                        call_id = call.get("id")
                        self.assertIsInstance(call_id, str, (record_number, index))
                        self.assertTrue(call_id, (record_number, index))
                        self.assertNotIn(call_id, seen, (record_number, call_id))
                        seen.add(call_id)
                        function = call.get("function", {})
                        name = function.get("name")
                        self.assertIn(name, advertised, (record_number, name))
                        raw = function.get("arguments", {})
                        arguments = json.loads(raw) if isinstance(raw, str) else raw
                        self.assertIsInstance(arguments, dict, (record_number, name))
                        schema = advertised[name]
                        self.assertFalse(set(arguments) - set(schema.get("properties", {})), (record_number, name, arguments))
                        self.assertFalse(set(schema.get("required", ())) - set(arguments), (record_number, name, arguments))
                        for key, value in arguments.items():
                            expected = schema["properties"][key].get("type")
                            self.assertTrue(
                                expected == "string" and isinstance(value, str)
                                or expected == "object" and isinstance(value, dict),
                                (record_number, name, key),
                            )
                        pending[call_id] = name
                elif role == "tool":
                    call_id = message.get("tool_call_id")
                    self.assertIn(call_id, pending, (record_number, index, call_id))
                    expected_name = pending.pop(call_id)
                    if message.get("tool_name") is not None:
                        self.assertEqual(message.get("tool_name"), expected_name, (record_number, index))
                    if expected_name == "search_files":
                        for result_line in message.get("content", "").splitlines():
                            folded = result_line.casefold().replace("/", "\\")
                            self.assertFalse(
                                folded.startswith(("model\\evaluations\\", "model\\training_data\\")),
                                (record_number, result_line),
                            )
                elif role not in {"user", "system"}:
                    self.fail(f"record {record_number}: unsupported role {role}")
                if role in {"user", "system"} and pending:
                    self.fail(f"record {record_number}: tool result ordering is invalid")
            if pending:
                self.assertEqual(record["messages"][-1].get("role"), "assistant", record_number)
                self.assertTrue(record["messages"][-1].get("tool_calls"), record_number)

    def test_primary_dataset_has_no_exact_duplicate_records_or_benchmark_prompts(self):
        records = load_jsonl(DATASET)
        canonical_records = [json.dumps(item, sort_keys=True, ensure_ascii=False, separators=(",", ":")) for item in records]
        self.assertEqual(len(canonical_records), len(set(canonical_records)))
        benchmark = load_jsonl(ROOT / "model" / "evaluations" / "jarvis_eval_32.jsonl")
        benchmark_prompts = {m.get("content", "").strip() for r in benchmark for m in r.get("messages", []) if m.get("role") == "user"}
        dataset_prompts = {m.get("content", "").strip() for r in records for m in r.get("messages", []) if m.get("role") == "user"}
        self.assertFalse(benchmark_prompts & dataset_prompts)
        self.assertGreaterEqual(len(records), 210)

    def test_runtime_training_and_evaluator_share_the_same_schema(self):
        from model.training_data.mlx_tools import TOOLS_SCHEMA as evaluator_schema
        from tools.registry import TOOLS
        from tools.schemas import schemas_for
        self.assertEqual(evaluator_schema, TOOLS_SCHEMA)
        self.assertEqual(set(TOOLS), {item["function"]["name"] for item in TOOLS_SCHEMA})
        by_name = {item["function"]["name"]: item["function"]["parameters"] for item in TOOLS_SCHEMA}
        self.assertEqual(by_name["list_files"]["properties"], {})
        self.assertEqual(set(by_name["search_files"]["properties"]), {"query"})
        self.assertEqual(by_name["recall_memory"]["properties"]["key"]["default"], "")
        self.assertEqual(schemas_for(["list_files"]), [next(item for item in TOOLS_SCHEMA if item["function"]["name"] == "list_files")])

    def test_key_search_examples_are_literal_and_results_grounded(self):
        records = load_jsonl(DATASET)
        self.assertEqual(json.loads(records[41]["messages"][1]["tool_calls"][0]["function"]["arguments"])["key"], "explanation_preference")
        literal_query = "def calculate(expression: str) " + "-> str:"
        self.assertEqual(json.loads(records[47]["messages"][1]["tool_calls"][0]["function"]["arguments"])["query"], literal_query)
        self.assertNotIn("create_eval32.py", records[47]["messages"][2]["content"])
        self.assertIn("registry.py:3", records[37]["messages"][-1]["content"])
        self.assertNotIn("calculator implementation is in tools/calculator.py", records[37]["messages"][-1]["content"])


if __name__ == "__main__":
    unittest.main()
