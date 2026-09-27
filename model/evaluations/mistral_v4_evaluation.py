import sys
import json
import re
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

MODEL_NAME = "mistralai/Ministral-3-8B-Reasoning-2512"

ADAPTER_PATH = (
    PROJECT_ROOT
    / "model"
    / "evaluations"
    / "jarvis_lora_v4"
    / "checkpoint-160"
)

EVAL_FILE = (
    PROJECT_ROOT
    / "model"
    / "evaluations"
    / "jarvis_eval_32.jsonl"
)

RESULTS_FILE = (
    PROJECT_ROOT
    / "model"
    / "evaluations"
    / "mistral_v4_results.json"
)


# ============================================================
# GENERATION SETTINGS
# ============================================================

MAX_INPUT_TOKENS = 1024
MAX_NEW_TOKENS = 512


# ============================================================
# JARVIS TOOL SCHEMA
# ============================================================

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to write.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and directories at a path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to list.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search files for a text query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Text to search for.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory or file path to search.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python_file",
            "description": "Run a Python file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the Python file.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "debug_python_file",
            "description": "Debug a Python file and report errors.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the Python file.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Calculate a mathematical expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate.",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember_memory",
            "description": "Store information in JARVIS memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Memory key.",
                    },
                    "value": {
                        "type": "string",
                        "description": "Value to remember.",
                    },
                },
                "required": ["key", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_memory",
            "description": "Recall stored JARVIS memory.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


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
# ============================================================

def normalize_messages(messages):
    normalized = []

    for message in messages:
        message = dict(message)

        role = message.get("role")

        if role == "assistant":
            tool_calls = message.get("tool_calls")

            if tool_calls:
                normalized_tool_calls = []

                for index, tool_call in enumerate(tool_calls):
                    tool_call = dict(tool_call)

                    if not tool_call.get("id"):
                        tool_call["id"] = f"call_{index}"

                    function = tool_call.get("function")

                    if function:
                        function = dict(function)

                        arguments = function.get("arguments", {})

                        if not isinstance(arguments, str):
                            arguments = json.dumps(
                                arguments,
                                separators=(",", ":"),
                            )

                        function["arguments"] = arguments
                        tool_call["function"] = function

                    normalized_tool_calls.append(tool_call)

                message["tool_calls"] = normalized_tool_calls

        elif role == "tool":
            if not message.get("tool_call_id"):
                message["tool_call_id"] = "call_0"

            if message.get("content") is None:
                message["content"] = ""

            if not isinstance(message["content"], str):
                message["content"] = json.dumps(
                    message["content"],
                    separators=(",", ":"),
                )

        normalized.append(message)

    return normalized


# ============================================================
# ARGUMENT PARSING
# ============================================================

def parse_arguments(raw_arguments):
    if raw_arguments is None:
        return {}

    if isinstance(raw_arguments, dict):
        return raw_arguments

    if not isinstance(raw_arguments, str):
        return None

    raw_arguments = raw_arguments.strip()

    if not raw_arguments:
        return {}

    try:
        parsed = json.loads(raw_arguments)
    except json.JSONDecodeError:
        return None

    if isinstance(parsed, dict):
        return parsed

    return None


# ============================================================
# MINISTRAL NATIVE TOOL-CALL PARSER
#
# Expected format:
#
# [TOOL_CALLS]read_file[ARGS]{"path":"foo.py"}
#
# Multiple calls:
#
# [TOOL_CALLS]read_file[ARGS]{"path":"a.py"}
# [TOOL_CALLS]read_file[ARGS]{"path":"b.py"}
# ============================================================

def parse_tool_calls_bracket_format(text):
    if not text:
        return []

    pattern = re.compile(
        r"\[TOOL_CALLS\]\s*"
        r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
        r"\s*\[ARGS\]\s*"
        r"(?P<arguments>\{.*?\})",
        re.DOTALL,
    )

    matches = pattern.finditer(text)

    tool_calls = []

    for match in matches:
        name = match.group("name")
        raw_arguments = match.group("arguments")

        arguments = parse_arguments(raw_arguments)

        if arguments is None:
            continue

        tool_calls.append(
            {
                "name": name,
                "arguments": arguments,
            }
        )

    return tool_calls


# ============================================================
# XML TOOL-CALL FALLBACK
#
# Supports:
#
# <tool_call>
# {"name":"read_file","arguments":{"path":"foo.py"}}
# </tool_call>
# ============================================================

def parse_xml_tool_calls(text):
    if not text:
        return []

    pattern = re.compile(
        r"<tool_call>\s*(.*?)\s*</tool_call>",
        re.DOTALL | re.IGNORECASE,
    )

    tool_calls = []

    for match in pattern.finditer(text):
        raw_call = match.group(1).strip()

        try:
            parsed = json.loads(raw_call)
        except json.JSONDecodeError:
            continue

        if not isinstance(parsed, dict):
            continue

        name = parsed.get("name")

        if not name:
            continue

        arguments = parsed.get("arguments", {})

        if isinstance(arguments, str):
            arguments = parse_arguments(arguments)

        if arguments is None:
            continue

        tool_calls.append(
            {
                "name": name,
                "arguments": arguments,
            }
        )

    return tool_calls


# ============================================================
# MODEL TOOL-CALL PARSER
# ============================================================

def parse_model_tool_calls(text):
    bracket_calls = parse_tool_calls_bracket_format(text)

    if bracket_calls:
        return bracket_calls

    return parse_xml_tool_calls(text)


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
            function = tool_call.get("function", {})

            name = function.get("name")

            if not name:
                continue

            arguments = parse_arguments(
                function.get("arguments", {})
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

def tool_decision_matches(expected_calls, predicted_calls):
    expected_has_tools = len(expected_calls) > 0
    predicted_has_tools = len(predicted_calls) > 0

    return expected_has_tools == predicted_has_tools


# ============================================================
# TOOL SELECTION METRIC
#
# Exact ordered tool-name match.
# Arguments are intentionally ignored here.
# ============================================================

def tool_selection_matches(expected_calls, predicted_calls):
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

def arguments_match(expected_calls, predicted_calls):
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
# LOAD BASE MODEL + V4 LORA
# ============================================================

def load_model():
    print()
    print("Loading Mistral 3 8B base model...")

    if not ADAPTER_PATH.exists():
        raise FileNotFoundError(
            f"V4 adapter checkpoint not found:\n{ADAPTER_PATH}"
        )

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

    print("Loading V4 LoRA adapter...")

    model = PeftModel.from_pretrained(
        model,
        ADAPTER_PATH,
    )

    model.eval()

    return model


# ============================================================
# GENERATION
# ============================================================

def generate_response(model, backend, messages):
    normalized_messages = normalize_messages(messages)

    if not normalized_messages:
        raise ValueError(
            "Evaluation example contains an empty message list."
        )

    encoded = backend.apply_chat_template(
        normalized_messages,
        tools=TOOLS_SCHEMA,
        add_generation_prompt=True,
        tokenize=True,
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
        return_tensors="pt",
        return_dict=True,
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
    print("JARVIS MISTRAL V4 EVALUATION")
    print("=" * 70)

    print()
    print(f"Model:       {MODEL_NAME}")
    print(f"Adapter:     {ADAPTER_PATH}")
    print(f"Evaluation:  {EVAL_FILE}")
    print(f"Results:     {RESULTS_FILE}")

    # --------------------------------------------------------
    # Load evaluation data
    # --------------------------------------------------------

    print()
    print("Loading evaluation data...")

    evaluation_data = load_evaluation_data()

    print(
        f"Loaded {len(evaluation_data)} evaluation examples."
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # The tokenizer is NOT stored on the PEFT model.
    #
    # MistralCommonBackend is itself the tokenizer/backend and
    # must be loaded independently from the model.
    # --------------------------------------------------------

    print()
    print("Loading Mistral tokenizer backend...")

    backend = MistralCommonBackend.from_pretrained(
        MODEL_NAME
    )

    print("Tokenizer backend loaded.")

    # --------------------------------------------------------
    # Evaluation counters
    # --------------------------------------------------------

    total_assistant_steps = 0

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
                "  WARNING: example has no valid messages list."
            )

            detailed_results.append(
                {
                    "example_index": example_index,
                    "error": "Missing or invalid messages list.",
                }
            )

            continue

        example_result = {
            "example_index": example_index,
            "assistant_steps": [],
        }

        # ----------------------------------------------------
        # Evaluate each expected assistant tool-decision step.
        #
        # The evaluation context is everything before the
        # assistant message being evaluated.
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

                if decision_match:
                    tool_decision_correct += 1

                if selection_match:
                    tool_selection_correct += 1

                if argument_match:
                    argument_correct += 1

                print(
                    f"decision={'PASS' if decision_match else 'FAIL'} "
                    f"selection={'PASS' if selection_match else 'FAIL'} "
                    f"arguments={'PASS' if argument_match else 'FAIL'}"
                )

                step_result = {
                    "message_index": message_index,
                    "expected_tool_calls": expected_calls,
                    "predicted_tool_calls": predicted_calls,
                    "raw_response": raw_response,
                    "tool_decision_match": decision_match,
                    "tool_selection_match": selection_match,
                    "argument_match": argument_match,
                }

            except Exception as exc:
                print("ERROR")

                step_result = {
                    "message_index": message_index,
                    "expected_tool_calls": expected_calls,
                    "predicted_tool_calls": [],
                    "raw_response": None,
                    "tool_decision_match": False,
                    "tool_selection_match": False,
                    "argument_match": False,
                    "error": str(exc),
                }

            example_result["assistant_steps"].append(
                step_result
            )

        detailed_results.append(example_result)

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    if total_assistant_steps > 0:
        tool_decision_accuracy = (
            tool_decision_correct
            / total_assistant_steps
        )

        tool_selection_accuracy = (
            tool_selection_correct
            / total_assistant_steps
        )

        argument_accuracy = (
            argument_correct
            / total_assistant_steps
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
        "adapter": str(ADAPTER_PATH),
        "evaluation_file": str(EVAL_FILE),
        "total_examples": len(evaluation_data),
        "total_assistant_steps": total_assistant_steps,
        "metrics": {
            "tool_decision_accuracy": tool_decision_accuracy,
            "tool_selection_accuracy": tool_selection_accuracy,
            "argument_accuracy": argument_accuracy,
        },
        "counts": {
            "tool_decision_correct": tool_decision_correct,
            "tool_selection_correct": tool_selection_correct,
            "argument_correct": argument_correct,
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
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Examples:               {len(evaluation_data)}"
    )

    print(
        f"Assistant steps:        {total_assistant_steps}"
    )

    print(
        f"Tool decision accuracy: "
        f"{tool_decision_accuracy:.2%}"
    )

    print(
        f"Tool selection accuracy:"
        f" {tool_selection_accuracy:.2%}"
    )

    print(
        f"Argument accuracy:      "
        f"{argument_accuracy:.2%}"
    )

    print()
    print(
        f"Results written to:\n{RESULTS_FILE}"
    )

    print("=" * 70)


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
