import sys
import json
import copy
from pathlib import Path

import torch
from transformers import (
    Mistral3ForConditionalGeneration,
    MistralCommonBackend,
    BitsAndBytesConfig,
)
from peft import PeftModel


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_NAME = "mistralai/Ministral-3-8B-Reasoning-2512"

# ============================================================
# V6 ADAPTER
# ============================================================

ADAPTER_PATH = (
    PROJECT_ROOT
    / "model"
    / "evaluations"
    / "jarvis_lora_v6"
    / "final"
)

# ============================================================
# FIXED EVALUATION SET
#
# DO NOT MODIFY THIS FILE.
# ============================================================

EVAL_FILE = (
    PROJECT_ROOT
    / "model"
    / "evaluations"
    / "jarvis_eval_32.jsonl"
)

# ============================================================
# V6 RESULTS
# ============================================================

RESULTS_FILE = (
    PROJECT_ROOT
    / "model"
    / "evaluations"
    / "mistral_v6_results.json"
)


# ============================================================
# GENERATION SETTINGS
# ============================================================

MAX_NEW_TOKENS = 512


# ============================================================
# JARVIS TOOL SCHEMA
# ============================================================

from model.training_data.mlx_tools import TOOLS_SCHEMA
from brain.v6_format import parse_arguments, parse_model_tool_calls


# ============================================================
# LOAD EVALUATION DATA
# ============================================================

def load_evaluation_data():
    if not EVAL_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation file not found:\n{EVAL_FILE}"
        )

    examples = []

    with EVAL_FILE.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                example = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc

            examples.append(example)

    return examples


# ============================================================
# MESSAGE NORMALIZATION
#
# IMPORTANT:
#
# Tool-call IDs are normalized consistently across the entire
# conversation.
#
# This prevents evaluator failures where an assistant tool call
# has one ID while its corresponding tool result references a
# different ID.
#
# These are evaluator-internal IDs only. They do not change the
# fixed evaluation cases.
# ============================================================

def normalize_messages(messages):
    normalized = []

    # Maps original tool-call IDs to queues of normalized IDs.
    #
    # A queue is used because multiple calls can theoretically
    # reuse the same source ID in malformed/historical data.
    pending_by_original_id = {}

    # Ordered list of currently unmatched normalized calls.
    pending_normalized_ids = []

    for message_index, original_message in enumerate(messages):
        message = copy.deepcopy(original_message)

        role = message.get("role")

        # --------------------------------------------------------
        # ASSISTANT TOOL CALL
        # --------------------------------------------------------

        if role == "assistant":
            tool_calls = message.get("tool_calls") or []

            if tool_calls:
                normalized_tool_calls = []

                for call_index, original_tool_call in enumerate(
                    tool_calls
                ):
                    tool_call = dict(original_tool_call)

                    original_id = tool_call.get("id")

                    # Every tool call gets a deterministic internal ID.
                    normalized_id = (
                        f"eval_call_{message_index}_{call_index}"
                    )

                    # Record mapping from original ID to normalized ID.
                    if original_id:
                        pending_by_original_id.setdefault(
                            original_id,
                            [],
                        ).append(normalized_id)

                    pending_normalized_ids.append(
                        normalized_id
                    )

                    tool_call["id"] = normalized_id

                    function = tool_call.get("function")

                    if function:
                        function = dict(function)

                        arguments = function.get(
                            "arguments",
                            {},
                        )

                        if not isinstance(arguments, str):
                            arguments = json.dumps(
                                arguments,
                                separators=(",", ":"),
                            )

                        function["arguments"] = arguments
                        tool_call["function"] = function

                    normalized_tool_calls.append(
                        tool_call
                    )

                message["tool_calls"] = normalized_tool_calls

        # --------------------------------------------------------
        # TOOL RESULT
        # --------------------------------------------------------

        elif role == "tool":
            original_tool_call_id = message.get(
                "tool_call_id"
            )

            normalized_tool_call_id = None

            # If the result provides an original ID, map it to the
            # corresponding normalized call ID.
            if original_tool_call_id:
                mapped_ids = pending_by_original_id.get(
                    original_tool_call_id
                )

                if mapped_ids:
                    normalized_tool_call_id = mapped_ids.pop(0)

                    if not mapped_ids:
                        del pending_by_original_id[
                            original_tool_call_id
                        ]

            # If the original ID was absent or could not be mapped,
            # use the oldest unmatched call.
            if normalized_tool_call_id is None:
                if not pending_normalized_ids:
                    raise ValueError(
                        "Tool result has no preceding unmatched "
                        "tool call."
                    )

                normalized_tool_call_id = (
                    pending_normalized_ids[0]
                )

            # Remove the normalized ID from the unmatched queue.
            if normalized_tool_call_id in pending_normalized_ids:
                pending_normalized_ids.remove(
                    normalized_tool_call_id
                )

            message["tool_call_id"] = (
                normalized_tool_call_id
            )

            if message.get("content") is None:
                message["content"] = ""

            if not isinstance(message["content"], str):
                message["content"] = json.dumps(
                    message["content"],
                    separators=(",", ":"),
                )

        normalized.append(message)

    # --------------------------------------------------------
    # Check for unmatched tool calls.
    # --------------------------------------------------------

    if pending_normalized_ids:
        raise ValueError(
            "Assistant tool calls have no matching tool results: "
            + ", ".join(pending_normalized_ids)
        )

    return normalized


# ============================================================
# EXPECTED TOOL-CALL EXTRACTION
# ============================================================

def extract_expected_tool_calls(messages):
    expected_calls = []

    for message in messages:
        if message.get("role") != "assistant":
            continue

        tool_calls = message.get("tool_calls")

        if not tool_calls:
            continue

        for tool_call in tool_calls:
            function = tool_call.get(
                "function",
                {},
            )

            name = function.get("name")

            if not name:
                continue

            arguments = parse_arguments(
                function.get(
                    "arguments",
                    {},
                )
            )

            if arguments is None:
                arguments = {}

            expected_calls.append(
                {
                    "name": name,
                    "arguments": arguments,
                }
            )

    return expected_calls


# ============================================================
# TOOL DECISION METRIC
#
# Did the model correctly decide whether a tool was needed?
# ============================================================

def tool_decision_matches(
    expected_calls,
    predicted_calls,
):
    expected_has_tools = len(expected_calls) > 0
    predicted_has_tools = len(predicted_calls) > 0

    return expected_has_tools == predicted_has_tools


# ============================================================
# TOOL SELECTION METRIC
#
# Exact ordered tool-name match.
# Arguments are intentionally ignored here.
# ============================================================

def tool_selection_matches(
    expected_calls,
    predicted_calls,
):
    expected_names = [
        call["name"]
        for call in expected_calls
    ]

    predicted_names = [
        call["name"]
        for call in predicted_calls
    ]

    return expected_names == predicted_names


# ============================================================
# ARGUMENT METRIC
#
# Exact tool names + exact argument dictionaries.
# ============================================================

def arguments_match(
    expected_calls,
    predicted_calls,
):
    if len(expected_calls) != len(predicted_calls):
        return False

    if not expected_calls:
        return True

    for expected, predicted in zip(
        expected_calls,
        predicted_calls,
    ):
        if expected["name"] != predicted["name"]:
            return False

        if expected["arguments"] != predicted["arguments"]:
            return False

    return True


# ============================================================
# LOAD BASE MODEL + V6 LORA
# ============================================================

def load_model():
    print()
    print("Loading Mistral 3 8B base model...")

    if not ADAPTER_PATH.exists():
        raise FileNotFoundError(
            f"V6 adapter not found:\n{ADAPTER_PATH}"
        )

    print()
    print("V6 adapter verified:")
    print(ADAPTER_PATH.resolve())

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = Mistral3ForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        tie_word_embeddings=False,
    )

    print("Loading V6 LoRA adapter...")

    model = PeftModel.from_pretrained(
        model,
        ADAPTER_PATH,
    )

    model.eval()

    return model


# ============================================================
# GENERATION
# ============================================================

def _config_value(config, name):
    if isinstance(config, dict):
        return config.get(name)
    return getattr(config, name, None)


def _get_input_token_limit(model):
    """Reserve the output budget within the model's configured context window."""
    config = getattr(model, "config", None)
    text_config = _config_value(config, "text_config")
    generation_config = getattr(model, "generation_config", None)
    configured_limits = (
        _config_value(text_config, "max_position_embeddings"),
        _config_value(config, "max_position_embeddings"),
        _config_value(generation_config, "max_length"),
    )
    context_limits = [
        int(limit)
        for limit in configured_limits
        if isinstance(limit, int) and limit > 0
    ]
    if not context_limits:
        raise ValueError("Could not determine the V6 model's supported context length.")

    input_limit = min(context_limits) - MAX_NEW_TOKENS
    if input_limit <= 0:
        raise ValueError("The V6 model context is smaller than its generation budget.")
    return input_limit


def _drop_oldest_user_turn(messages):
    """Drop one complete old user turn while preserving system and newest turns."""
    user_positions = [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "user"
    ]
    if len(user_positions) < 2:
        return None

    start, next_start = user_positions[:2]
    return [
        message
        for index, message in enumerate(messages)
        if not (
            start <= index < next_start
            and message.get("role") != "system"
        )
    ]


def _encode_prompt_with_budget(backend, messages, tools, max_input_tokens):
    """Encode without tokenizer truncation, pruning only complete oldest turns."""
    current_messages = copy.deepcopy(messages)

    while True:
        encoded = backend.apply_chat_template(
            current_messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=True,
            truncation=False,
            return_tensors="pt",
            return_dict=True,
        )
        input_ids = encoded.get("input_ids")
        if input_ids is None or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("V6 prompt encoding must contain one input_ids sequence.")

        if input_ids.shape[1] <= max_input_tokens:
            return encoded

        trimmed_messages = _drop_oldest_user_turn(current_messages)
        if trimmed_messages is None:
            raise ValueError(
                "The V6 system prompt, available tools, and newest user turn exceed "
                f"the supported input budget of {max_input_tokens} tokens."
            )
        current_messages = trimmed_messages

def generate_response(
    model,
    backend,
    messages,
    tools=None,
):
    normalized_messages = normalize_messages(
        messages
    )

    if not normalized_messages:
        raise ValueError(
            "Evaluation context contains an empty "
            "message list."
        )

    active_tools = TOOLS_SCHEMA if tools is None else tools
    encoded = _encode_prompt_with_budget(
        backend,
        normalized_messages,
        tools=active_tools,
        max_input_tokens=_get_input_token_limit(model),
    )

    encoded = {
        key: value.to(model.device)
        for key, value in encoded.items()
        if hasattr(value, "to")
    }

    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
        )

    input_length = encoded["input_ids"].shape[1]

    generated_tokens = generated[
        0,
        input_length:,
    ]

    raw_response = backend.decode(
        generated_tokens,
        skip_special_tokens=False,
    )

    return raw_response


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():
    print("=" * 70)
    print("JARVIS MISTRAL V6 EVALUATION")
    print("=" * 70)

    print()
    print(f"Model:       {MODEL_NAME}")
    print(f"Adapter:     {ADAPTER_PATH.resolve()}")
    print(f"Evaluation:  {EVAL_FILE.resolve()}")
    print(f"Results:     {RESULTS_FILE.resolve()}")

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    if not ADAPTER_PATH.exists():
        raise FileNotFoundError(
            "V6 adapter does not exist:\n"
            f"{ADAPTER_PATH.resolve()}"
        )

    if not EVAL_FILE.exists():
        raise FileNotFoundError(
            "Fixed evaluation file does not exist:\n"
            f"{EVAL_FILE.resolve()}"
        )

    # --------------------------------------------------------
    # Load evaluation data
    # --------------------------------------------------------

    print()
    print("Loading fixed evaluation data...")

    evaluation_data = load_evaluation_data()

    print(
        f"Loaded {len(evaluation_data)} "
        "evaluation examples."
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Load tokenizer/backend
    # --------------------------------------------------------

    print()
    print("Loading Mistral tokenizer backend...")

    backend = MistralCommonBackend.from_pretrained(
        MODEL_NAME
    )

    print("Tokenizer backend loaded.")

    # --------------------------------------------------------
    # Evaluation counters
    #
    # total_assistant_steps:
    #     Every assistant decision encountered.
    #
    # successful_steps:
    #     Steps where the evaluator successfully generated
    #     and parsed a model response.
    #
    # evaluator_errors:
    #     Runtime/protocol/template errors in the evaluator.
    #
    # Metrics are calculated ONLY from successful steps.
    # --------------------------------------------------------

    total_assistant_steps = 0
    successful_steps = 0
    evaluator_errors = 0

    tool_decision_correct = 0
    tool_selection_correct = 0
    argument_correct = 0

    detailed_results = []

    # --------------------------------------------------------
    # Evaluate every example
    # --------------------------------------------------------

    for example_index, example in enumerate(
        evaluation_data,
        start=1,
    ):
        print()
        print(
            f"[{example_index}/{len(evaluation_data)}] "
            "Evaluating example..."
        )

        messages = example.get("messages")

        if not isinstance(messages, list):
            print(
                "  WARNING: example has no valid "
                "messages list."
            )

            detailed_results.append(
                {
                    "example_index": example_index,
                    "error_type": "evaluator_error",
                    "error": (
                        "Missing or invalid messages list."
                    ),
                }
            )

            evaluator_errors += 1

            continue

        example_result = {
            "example_index": example_index,
            "assistant_steps": [],
        }

        # ----------------------------------------------------
        # Evaluate each expected assistant decision step.
        #
        # The context is everything before the assistant
        # message currently being evaluated.
        # ----------------------------------------------------

        for message_index, message in enumerate(messages):
            if message.get("role") != "assistant":
                continue

            total_assistant_steps += 1

            context = messages[:message_index]

            expected_calls = extract_expected_tool_calls(
                [message]
            )

            print(
                f"  Assistant step "
                f"{total_assistant_steps}:",
                end=" ",
                flush=True,
            )

            try:
                raw_response = generate_response(
                    model,
                    backend,
                    context,
                    tools=example.get("tools", []),
                )

                predicted_calls = parse_model_tool_calls(
                    raw_response
                )

                decision_match = tool_decision_matches(
                    expected_calls,
                    predicted_calls,
                )

                selection_match = tool_selection_matches(
                    expected_calls,
                    predicted_calls,
                )

                argument_match = arguments_match(
                    expected_calls,
                    predicted_calls,
                )

                successful_steps += 1

                if decision_match:
                    tool_decision_correct += 1

                if selection_match:
                    tool_selection_correct += 1

                if argument_match:
                    argument_correct += 1

                print(
                    f"decision="
                    f"{'PASS' if decision_match else 'FAIL'} "
                    f"selection="
                    f"{'PASS' if selection_match else 'FAIL'} "
                    f"arguments="
                    f"{'PASS' if argument_match else 'FAIL'}"
                )

                step_result = {
                    "message_index": message_index,
                    "status": "evaluated",
                    "expected_tool_calls": expected_calls,
                    "predicted_tool_calls": predicted_calls,
                    "raw_response": raw_response,
                    "tool_decision_match": decision_match,
                    "tool_selection_match": selection_match,
                    "argument_match": argument_match,
                }

            except Exception as exc:
                evaluator_errors += 1

                print("EVALUATOR ERROR")

                step_result = {
                    "message_index": message_index,
                    "status": "evaluator_error",
                    "expected_tool_calls": expected_calls,
                    "predicted_tool_calls": [],
                    "raw_response": None,
                    "tool_decision_match": None,
                    "tool_selection_match": None,
                    "argument_match": None,
                    "error": str(exc),
                }

            example_result["assistant_steps"].append(
                step_result
            )

        detailed_results.append(
            example_result
        )

    # --------------------------------------------------------
    # Calculate metrics
    #
    # IMPORTANT:
    #
    # Evaluator errors are excluded from the behavioral
    # denominator.
    # --------------------------------------------------------

    if successful_steps > 0:
        tool_decision_accuracy = (
            tool_decision_correct
            / successful_steps
        )

        tool_selection_accuracy = (
            tool_selection_correct
            / successful_steps
        )

        argument_accuracy = (
            argument_correct
            / successful_steps
        )
    else:
        tool_decision_accuracy = 0.0
        tool_selection_accuracy = 0.0
        argument_accuracy = 0.0

    # --------------------------------------------------------
    # Final result object
    # --------------------------------------------------------

    results = {
        "model": MODEL_NAME,
        "adapter": str(
            ADAPTER_PATH.resolve()
        ),
        "evaluation_file": str(
            EVAL_FILE.resolve()
        ),
        "total_examples": len(
            evaluation_data
        ),
        "total_assistant_steps": (
            total_assistant_steps
        ),
        "successful_evaluation_steps": (
            successful_steps
        ),
        "evaluator_errors": (
            evaluator_errors
        ),
        "metrics": {
            "tool_decision_accuracy": (
                tool_decision_accuracy
            ),
            "tool_selection_accuracy": (
                tool_selection_accuracy
            ),
            "argument_accuracy": (
                argument_accuracy
            ),
        },
        "counts": {
            "tool_decision_correct": (
                tool_decision_correct
            ),
            "tool_selection_correct": (
                tool_selection_correct
            ),
            "argument_correct": (
                argument_correct
            ),
        },
        "examples": detailed_results,
    }

    # --------------------------------------------------------
    # Write results
    # --------------------------------------------------------

    RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RESULTS_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("V6 EVALUATION COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Examples:                  "
        f"{len(evaluation_data)}"
    )

    print(
        f"Assistant steps:           "
        f"{total_assistant_steps}"
    )

    print(
        f"Successfully evaluated:    "
        f"{successful_steps}"
    )

    print(
        f"Evaluator errors:          "
        f"{evaluator_errors}"
    )

    print()

    print(
        f"Tool decision accuracy:    "
        f"{tool_decision_accuracy:.2%}"
    )

    print(
        f"Tool selection accuracy:   "
        f"{tool_selection_accuracy:.2%}"
    )

    print(
        f"Argument accuracy:         "
        f"{argument_accuracy:.2%}"
    )

    print()

    print(
        "V6 adapter evaluated:"
    )

    print(
        ADAPTER_PATH.resolve()
    )

    print()

    print(
        "Fixed evaluation set:"
    )

    print(
        EVAL_FILE.resolve()
    )

    print()

    print(
        f"Results written to:\n"
        f"{RESULTS_FILE.resolve()}"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print()
        print("Evaluation interrupted by user.")
        sys.exit(1)

    except Exception as exc:
        print()
        print("=" * 70)
        print("EVALUATION FAILED")
        print("=" * 70)
        print()
        print(str(exc))
        sys.exit(1)
