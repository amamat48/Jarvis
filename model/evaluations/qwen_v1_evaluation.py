"""
JARVIS v1 Behavioral Evaluation
================================

This script evaluates the fine-tuned JARVIS LoRA adapter against the
same held-out 16-example test set used for the untouched Qwen3-4B
baseline.

The experiment is intentionally controlled:

    BASELINE
        Qwen3-4B-Instruct-2507-4bit
        no adapter

    JARVIS v1
        Qwen3-4B-Instruct-2507-4bit
        + jarvis_lora_v1 adapter

Both models are evaluated on exactly the same test examples.

The primary measurement is TOOL BEHAVIOR:

    - Did the model choose the expected tool?
    - Did it avoid using a tool when none was expected?
    - Did it follow the expected tool sequence?

We also save the model's raw responses and arguments so that we can
manually inspect failures later.

The test set is not used for training.
"""

import json
import re
import sys
from pathlib import Path


# ============================================================================
# PROJECT PATH SETUP
# ============================================================================
#
# This script lives at:
#
#     jarvis/model/evaluations/qwen_v1_evaluation.py
#
# parents[0] = model/evaluations
# parents[1] = model
# parents[2] = jarvis
#
# Adding the project root to sys.path allows Python to import modules from
# the rest of the JARVIS project regardless of where the script is launched.
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# MLX IMPORTS
# ============================================================================

from mlx_lm import load, generate

# MLX-LM provides this utility for reconstructing the LoRA layers and
# loading the adapter weights saved during training.
from mlx_lm.tuner.utils import load_adapters


# ============================================================================
# JARVIS TRAINING DATA / TOOL SCHEMA
# ============================================================================

from model.training_data.mlx_tools import TOOLS_SCHEMA


# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"

ADAPTER_PATH = (
    PROJECT_ROOT
    / "model"
    / "evaluations"
    / "jarvis_lora_v1"
)

TEST_FILE = (
    PROJECT_ROOT
    / "model"
    / "training_data"
    / "mlx"
    / "test.jsonl"
)

BASELINE_RESULTS_FILE = (
    PROJECT_ROOT
    / "model"
    / "evaluations"
    / "qwen_baseline_results.json"
)

RESULTS_FILE = (
    PROJECT_ROOT
    / "model"
    / "evaluations"
    / "jarvis_v1_results.json"
)

# Tool calls are normally very short. 256 generated tokens is enough to
# capture a normal tool decision plus a concise answer without making
# evaluation unnecessarily slow.
MAX_TOKENS = 256


# ============================================================================
# TOOL-CALL PARSER
# ============================================================================

TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.DOTALL,
)


def parse_tool_calls(response_text: str) -> list[dict]:
    """
    Parse native Qwen tool calls from generated text.

    Qwen3 emits tool calls in the general form:

        <tool_call>
        {"name": "...", "arguments": {...}}
        </tool_call>

    We parse the JSON ourselves so that the evaluation script can record
    malformed tool calls instead of silently ignoring them.
    """

    calls = []

    for match in TOOL_CALL_PATTERN.findall(response_text):
        raw_tool_call = match.strip()

        try:
            parsed = json.loads(raw_tool_call)
        except json.JSONDecodeError:
            calls.append({
                "name": None,
                "arguments": {},
                "parse_error": True,
                "raw": raw_tool_call,
            })
            continue

        if not isinstance(parsed, dict):
            calls.append({
                "name": None,
                "arguments": {},
                "parse_error": True,
                "raw": raw_tool_call,
            })
            continue

        name = parsed.get("name")
        arguments = parsed.get("arguments", {})

        # Some implementations may serialize the arguments as a JSON
        # string rather than leaving them as an object. Accept either form
        # so the evaluator remains tolerant of representation differences.
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {
                    "_raw_arguments": arguments
                }

        calls.append({
            "name": name,
            "arguments": arguments,
            "parse_error": False,
            "raw": raw_tool_call,
        })

    return calls


# ============================================================================
# EXPECTED TOOL EXTRACTION
# ============================================================================

def get_expected_tool_calls(message: dict) -> list[dict]:
    """
    Extract the reference tool calls from one assistant message.

    The converted MLX dataset represents arguments as a JSON string.
    This function converts them back into Python dictionaries so that the
    saved evaluation results are easier to read.
    """

    expected = []

    for call in message.get("tool_calls", []):
        function = call.get("function", {})

        name = function.get("name")
        arguments = function.get("arguments", {})

        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {
                    "_raw_arguments": arguments
                }

        expected.append({
            "name": name,
            "arguments": arguments,
        })

    return expected


# ============================================================================
# TOOL-SEQUENCE COMPARISON
# ============================================================================

def compare_tool_sequences(
    expected_calls: list[dict],
    actual_calls: list[dict],
) -> bool:
    """
    Compare the ordered tool names selected by the model.

    We intentionally compare tool names rather than exact arguments for the
    primary behavioral metric.

    Why?

    A task such as:

        Calculate 3180 / 15, then multiply the result by 7.

    can validly be performed as either:

        calculate("(3180 / 15) * 7")

    or:

        calculate("3180 / 15")
        calculate("212 * 7")

    Those are different argument sequences but represent the same overall
    operation.

    Argument details are still recorded for manual analysis.
    """

    expected_names = [
        call.get("name")
        for call in expected_calls
    ]

    actual_names = [
        call.get("name")
        for call in actual_calls
    ]

    return expected_names == actual_names


# ============================================================================
# SINGLE ASSISTANT-STEP EVALUATION
# ============================================================================

def evaluate_assistant_step(
    model,
    tokenizer,
    context_messages: list[dict],
    reference_message: dict,
    example_index: int,
    step_index: int,
) -> dict:
    """
    Evaluate one assistant decision.

    The model receives every message BEFORE the reference assistant
    message. The reference assistant response itself is hidden.

    This is teacher-forced evaluation.

    It isolates individual decisions so that one early mistake doesn't
    alter the context used to evaluate later assistant decisions.
    """

    prompt = tokenizer.apply_chat_template(
        context_messages,
        tools=TOOLS_SCHEMA,
        add_generation_prompt=True,
        tokenize=False,
    )

    try:
        response_text = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=MAX_TOKENS,
            verbose=False,
        )

    except Exception as error:
        expected_calls = get_expected_tool_calls(
            reference_message
        )

        return {
            "example_index": example_index,
            "step_index": step_index,
            "passed": False,
            "error": str(error),
            "expected_tools": [
                call.get("name")
                for call in expected_calls
            ],
            "actual_tools": [],
            "expected_arguments": [
                call.get("arguments")
                for call in expected_calls
            ],
            "actual_arguments": [],
            "response": "",
        }

    actual_calls = parse_tool_calls(response_text)
    expected_calls = get_expected_tool_calls(
        reference_message
    )

    passed = compare_tool_sequences(
        expected_calls,
        actual_calls,
    )

    return {
        "example_index": example_index,
        "step_index": step_index,
        "passed": passed,
        "expected_tools": [
            call.get("name")
            for call in expected_calls
        ],
        "actual_tools": [
            call.get("name")
            for call in actual_calls
        ],
        "expected_arguments": [
            call.get("arguments")
            for call in expected_calls
        ],
        "actual_arguments": [
            call.get("arguments")
            for call in actual_calls
        ],
        "response": response_text,
    }


# ============================================================================
# EXAMPLE EVALUATION
# ============================================================================

def evaluate_example(
    model,
    tokenizer,
    example: dict,
    example_index: int,
) -> dict:
    """
    Evaluate every assistant decision contained in one test example.

    A multi-step conversation may contain several assistant decisions:

        user
        assistant -> search_files

        tool result
        assistant -> read_file

        tool result
        assistant -> final answer

    Each assistant decision is measured separately.
    """

    messages = example.get("messages", [])

    step_results = []

    for message_index, message in enumerate(messages):

        # Only assistant messages represent model decisions that we need
        # to evaluate.
        if message.get("role") != "assistant":
            continue

        context = messages[:message_index]

        result = evaluate_assistant_step(
            model=model,
            tokenizer=tokenizer,
            context_messages=context,
            reference_message=message,
            example_index=example_index,
            step_index=len(step_results) + 1,
        )

        step_results.append(result)

    passed_steps = sum(
        1
        for result in step_results
        if result.get("passed")
    )

    return {
        "example_index": example_index,
        "steps": step_results,
        "passed_steps": passed_steps,
        "total_steps": len(step_results),
        "passed": (
            len(step_results) > 0
            and passed_steps == len(step_results)
        ),
    }


# ============================================================================
# TEST DATA LOADING
# ============================================================================

def load_test_examples() -> list[dict]:
    """
    Load the held-out test set.

    This is the exact same test file used by the Qwen baseline evaluator,
    which makes the comparison meaningful.
    """

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {TEST_FILE}"
        )

    examples = []

    for line_number, line in enumerate(
        TEST_FILE.read_text().splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            examples.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON in {TEST_FILE} at line "
                f"{line_number}: {error}"
            ) from error

    return examples


# ============================================================================
# BASELINE LOADING
# ============================================================================

def load_baseline_results() -> dict | None:
    """
    Load the previously recorded Qwen3 baseline.

    The baseline isn't required for the JARVIS v1 evaluation itself, but
    loading it lets us automatically calculate the before/after change
    at the end.
    """

    if not BASELINE_RESULTS_FILE.exists():
        return None

    try:
        return json.loads(
            BASELINE_RESULTS_FILE.read_text()
        )
    except json.JSONDecodeError:
        return None


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=== JARVIS v1 EVALUATION ===")
    print()
    print(f"Base model: {MODEL}")
    print(f"Adapter:    {ADAPTER_PATH}")
    print(f"Test set:   {TEST_FILE}")
    print()

    if not ADAPTER_PATH.exists():
        raise FileNotFoundError(
            f"JARVIS v1 adapter was not found at: "
            f"{ADAPTER_PATH}"
        )

    examples = load_test_examples()

    print(f"Test examples: {len(examples)}")
    print()
    print("Loading base model...")

    model, tokenizer = load(MODEL)

    print("Base model loaded.")
    print("Loading JARVIS v1 adapter...")

    # This reconstructs the LoRA layers using the adapter's saved
    # configuration and loads adapters.safetensors into those layers.
    load_adapters(
        model,
        str(ADAPTER_PATH),
    )

    print("JARVIS v1 adapter loaded successfully.")
    print()

    all_results = []

    total_steps = 0
    passed_steps = 0
    passed_examples = 0

    for example_index, example in enumerate(
        examples,
        start=1,
    ):
        print(
            f"Running JARVIS v1 test "
            f"{example_index}/{len(examples)}..."
        )

        result = evaluate_example(
            model=model,
            tokenizer=tokenizer,
            example=example,
            example_index=example_index,
        )

        all_results.append(result)

        total_steps += result["total_steps"]
        passed_steps += result["passed_steps"]

        if result["passed"]:
            passed_examples += 1

        print(
            f"  Example: "
            f"{'PASS' if result['passed'] else 'FAIL'}"
        )

        for step in result["steps"]:
            print(
                f"    Step {step['step_index']}: "
                f"{'PASS' if step['passed'] else 'FAIL'}"
            )
            print(
                f"      Expected: "
                f"{step['expected_tools']}"
            )
            print(
                f"      Actual:   "
                f"{step['actual_tools']}"
            )

        print()

    if total_steps:
        step_accuracy = (
            passed_steps / total_steps
        )
    else:
        step_accuracy = 0.0

    if examples:
        example_accuracy = (
            passed_examples / len(examples)
        )
    else:
        example_accuracy = 0.0

    baseline = load_baseline_results()

    comparison = None

    if baseline:
        baseline_example_accuracy = baseline.get(
            "example_accuracy",
            0.0,
        )

        baseline_step_accuracy = baseline.get(
            "step_accuracy",
            0.0,
        )

        comparison = {
            "baseline_example_accuracy": baseline_example_accuracy,
            "jarvis_v1_example_accuracy": example_accuracy,
            "example_accuracy_delta": (
                example_accuracy
                - baseline_example_accuracy
            ),
            "baseline_step_accuracy": baseline_step_accuracy,
            "jarvis_v1_step_accuracy": step_accuracy,
            "step_accuracy_delta": (
                step_accuracy
                - baseline_step_accuracy
            ),
        }

    output = {
        "model": MODEL,
        "adapter": str(ADAPTER_PATH),
        "test_file": str(TEST_FILE),
        "test_examples": len(examples),
        "total_assistant_steps": total_steps,
        "passed_assistant_steps": passed_steps,
        "step_accuracy": step_accuracy,
        "passed_examples": passed_examples,
        "example_accuracy": example_accuracy,
        "comparison_to_baseline": comparison,
        "results": all_results,
    }

    RESULTS_FILE.write_text(
        json.dumps(
            output,
            indent=4,
            ensure_ascii=False,
        )
    )

    print("=== JARVIS v1 SUMMARY ===")
    print(
        f"Examples passed: "
        f"{passed_examples}/{len(examples)}"
    )
    print(
        f"Example accuracy: "
        f"{example_accuracy * 100:.1f}%"
    )
    print(
        f"Assistant steps passed: "
        f"{passed_steps}/{total_steps}"
    )
    print(
        f"Step accuracy: "
        f"{step_accuracy * 100:.1f}%"
    )

    if comparison:
        print()
        print("=== BASELINE COMPARISON ===")
        print(
            f"Baseline example accuracy: "
            f"{comparison['baseline_example_accuracy'] * 100:.1f}%"
        )
        print(
            f"JARVIS v1 example accuracy: "
            f"{comparison['jarvis_v1_example_accuracy'] * 100:.1f}%"
        )
        print(
            f"Example accuracy change: "
            f"{comparison['example_accuracy_delta'] * 100:+.1f} "
            f"percentage points"
        )
        print()
        print(
            f"Baseline step accuracy: "
            f"{comparison['baseline_step_accuracy'] * 100:.1f}%"
        )
        print(
            f"JARVIS v1 step accuracy: "
            f"{comparison['jarvis_v1_step_accuracy'] * 100:.1f}%"
        )
        print(
            f"Step accuracy change: "
            f"{comparison['step_accuracy_delta'] * 100:+.1f} "
            f"percentage points"
        )

    print()
    print("Detailed results saved to:")
    print(f"  {RESULTS_FILE}")


if __name__ == "__main__":
    main()