"""QLoRA training entry point for the existing JARVIS V6 dataset."""

# Imports
import json
import math
import random
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    BitsAndBytesConfig,
    Mistral3ForConditionalGeneration,
    MistralCommonBackend,
)


# Configuration
MODEL_NAME = "mistralai/Ministral-3-8B-Reasoning-2512"

BASE_DIR = Path(__file__).resolve().parent
TRAIN_FILE = BASE_DIR / "training_data" / "v6" / "train.jsonl"
OUTPUT_DIR = BASE_DIR / "evaluations" / "jarvis_lora_v6"

MAX_LENGTH = 1024
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 1e-5
TOTAL_ITERATIONS = 240
VALIDATION_INTERVAL = 40
VALIDATION_FRACTION = 0.10
SEED = 42


# Reproducibility
def set_reproducibility(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Utility functions
def print_memory(label):
    if not torch.cuda.is_available():
        return

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3

    print(
        f"[MEMORY] {label}: "
        f"allocated={allocated:.2f} GB | "
        f"reserved={reserved:.2f} GB | "
        f"peak={peak:.2f} GB"
    )


def load_examples(path):
    examples = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}: {error}"
                ) from error

            if not isinstance(item, dict):
                raise ValueError(
                    f"Expected an object on line {line_number} of {path}"
                )

            messages = item.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(
                    f"Missing non-empty messages on line {line_number} of {path}"
                )

            tools = item.get("tools")
            if tools is not None and not isinstance(tools, list):
                raise ValueError(
                    f"Expected tools to be a list on line {line_number} of {path}"
                )

            examples.append(
                {
                    "messages": normalize_tool_messages(messages),
                    "tools": tools,
                }
            )

    if len(examples) < 2:
        raise ValueError(
            f"At least two examples are required for a train/validation split: {path}"
        )

    print(f"Loaded {len(examples)} examples from: {path}")
    return examples


def split_examples(examples, validation_fraction=VALIDATION_FRACTION, seed=SEED):
    """Create a deterministic in-memory holdout; no validation file is written."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    indices = list(range(len(examples)))
    random.Random(seed).shuffle(indices)

    validation_count = max(
        1,
        math.ceil(len(examples) * validation_fraction),
    )
    validation_count = min(validation_count, len(examples) - 1)

    validation_indices = set(indices[:validation_count])
    train_examples = [
        example
        for index, example in enumerate(examples)
        if index not in validation_indices
    ]
    validation_examples = [
        example
        for index, example in enumerate(examples)
        if index in validation_indices
    ]

    return train_examples, validation_examples


# Tool/message normalization
def normalize_tool_messages(messages):
    normalized = []
    pending_tool_ids = []
    seen_tool_ids = set()

    for message_index, original_message in enumerate(messages):
        if not isinstance(original_message, dict):
            raise ValueError(
                f"Message {message_index} must be an object"
            )

        message = dict(original_message)
        role = message.get("role")

        if role == "assistant" and message.get("tool_calls"):
            original_calls = message["tool_calls"]
            if not isinstance(original_calls, list):
                raise ValueError(
                    f"tool_calls in message {message_index} must be a list"
                )

            tool_calls = []
            for call_index, original_call in enumerate(original_calls):
                if not isinstance(original_call, dict):
                    raise ValueError(
                        f"Tool call {call_index} in message {message_index} must be an object"
                    )

                call = dict(original_call)
                call_id = call.get("id")
                if not isinstance(call_id, str) or not call_id:
                    call_id = f"jarvis_v5_tool_{message_index}_{call_index}"

                if call_id in seen_tool_ids:
                    raise ValueError(f"Duplicate tool-call ID: {call_id}")

                function = call.get("function")
                if not isinstance(function, dict):
                    raise ValueError(
                        f"Tool call {call_id} has no function object"
                    )

                function = dict(function)
                if not isinstance(function.get("name"), str):
                    raise ValueError(
                        f"Tool call {call_id} has no function name"
                    )

                arguments = function.get("arguments", {})
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)

                function["arguments"] = arguments
                call["id"] = call_id
                call.setdefault("type", "function")
                call["function"] = function

                seen_tool_ids.add(call_id)
                pending_tool_ids.append(call_id)
                tool_calls.append(call)

            message["tool_calls"] = tool_calls

        elif role == "tool":
            tool_call_id = message.get("tool_call_id") or message.get("id")

            if tool_call_id is None:
                if not pending_tool_ids:
                    raise ValueError(
                        f"Orphan tool result in message {message_index}"
                    )
                tool_call_id = pending_tool_ids[0]

            if not isinstance(tool_call_id, str) or tool_call_id not in pending_tool_ids:
                raise ValueError(
                    f"Tool result in message {message_index} does not match a pending call"
                )

            pending_tool_ids.remove(tool_call_id)
            message["tool_call_id"] = tool_call_id
            message.pop("id", None)

        normalized.append(message)

    # A final assistant tool-call target may intentionally have no result.
    return normalized


# Dataset
class JarvisDataset(Dataset):
    def __init__(self, examples, backend):
        self.tokens = []

        print("Tokenizing dataset...")
        for index, example in enumerate(examples):
            messages = example["messages"]
            continue_final_message = messages[-1].get("role") == "assistant"

            try:
                result = backend.apply_chat_template(
                    messages,
                    tools=example["tools"],
                    add_generation_prompt=False,
                    continue_final_message=continue_final_message,
                    tokenize=True,
                    truncation=True,
                    max_length=MAX_LENGTH,
                    return_tensors="pt",
                    return_dict=True,
                )

                input_ids = result["input_ids"]
                if input_ids.dim() == 2:
                    input_ids = input_ids[0]

                self.tokens.append(input_ids.to(torch.long))

            except Exception as error:
                final_role = messages[-1].get("role")
                raise RuntimeError(
                    f"Tokenization failed for example {index} "
                    f"(final role: {final_role}): "
                    f"{type(error).__name__}: {error}"
                ) from error

        print(f"Tokenization complete: {len(self.tokens)} examples")

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        input_ids = self.tokens[index]
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
        }


# Batch collation
def collate_batch(batch, pad_token_id):
    if not batch:
        raise ValueError("Cannot collate an empty batch")

    max_length = max(item["input_ids"].shape[0] for item in batch)
    input_ids = []
    labels = []

    for item in batch:
        ids = item["input_ids"]
        padding = max_length - ids.shape[0]

        if padding:
            ids = torch.cat(
                [
                    ids,
                    torch.full(
                        (padding,),
                        pad_token_id,
                        dtype=torch.long,
                    ),
                ]
            )

        input_ids.append(ids)

        label = ids.clone()
        if padding:
            label[-padding:] = -100
        labels.append(label)

    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels)

    return {
        "input_ids": input_ids,
        "attention_mask": input_ids.ne(pad_token_id),
        "labels": labels,
    }


# Model setup
def setup_model(device):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print("Loading 4-bit model...")
    model = Mistral3ForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        tie_word_embeddings=False,
    )
    print("4-bit model loaded successfully")
    print_memory("After model load")

    print("Loading Mistral Common tokenizer backend...")
    backend = MistralCommonBackend.from_pretrained(MODEL_NAME)
    print("Tokenizer backend loaded")

    print("Preparing model for QLoRA...")
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print_memory("After LoRA setup")

    return model, backend


# Validation
def evaluate(model, validation_dataset, backend, device):
    if len(validation_dataset) == 0:
        raise ValueError("Validation dataset is empty")

    was_training = model.training
    model.eval()

    total_loss = 0.0
    try:
        with torch.no_grad():
            for index in range(len(validation_dataset)):
                batch = collate_batch(
                    [validation_dataset[index]],
                    backend.pad_token_id,
                )

                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device),
                )
                total_loss += outputs.loss.item()
    finally:
        if was_training:
            model.train()

    return total_loss / len(validation_dataset)


# Final save
def ensure_final_output_is_empty():
    final_dir = OUTPUT_DIR / "final"
    if final_dir.exists() and (
        not final_dir.is_dir() or any(final_dir.iterdir())
    ):
        raise FileExistsError(
            f"Refusing to overwrite existing V6 adapter output: {final_dir}"
        )


def save_final_adapter(model, backend):
    ensure_final_output_is_empty()
    final_dir = OUTPUT_DIR / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    backend.save_pretrained(final_dir)
    print(f"[V6] Final adapter saved to: {final_dir}")
    print_memory("Final")


# Training
def train_model(model, backend, train_dataset, validation_dataset, device):
    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
    )

    print("Starting JARVIS V6 training")
    model.train()
    start_time = time.time()
    iteration = 0

    while iteration < TOTAL_ITERATIONS:
        order = list(range(len(train_dataset)))
        random.shuffle(order)

        for dataset_index in order:
            if iteration >= TOTAL_ITERATIONS:
                break

            batch = collate_batch(
                [train_dataset[dataset_index]],
                backend.pad_token_id,
            )
            optimizer.zero_grad(set_to_none=True)

            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            iteration += 1

            if iteration == 1 or iteration % 10 == 0:
                elapsed_minutes = (time.time() - start_time) / 60
                print(
                    f"[V6] Iteration {iteration:3d}/{TOTAL_ITERATIONS} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Time: {elapsed_minutes:.1f} min"
                )

            if iteration % VALIDATION_INTERVAL == 0:
                validation_loss = evaluate(
                    model,
                    validation_dataset,
                    backend,
                    device,
                )
                print(
                    f"[V6 VALIDATION] Iteration {iteration} | "
                    f"Loss: {validation_loss:.4f}"
                )



# Main entry point
def main():
    print("JARVIS V6 QLoRA Training")
    print(f"Model: {MODEL_NAME}")
    print(f"Training data: {TRAIN_FILE}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Max sequence length: {MAX_LENGTH}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Iterations: {TOTAL_ITERATIONS}")
    print(f"Validation holdout: {VALIDATION_FRACTION:.0%}")
    print()

    ensure_final_output_is_empty()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device = torch.device("cuda:0")
    print(f"Using device: {device}")

    set_reproducibility()
    examples = load_examples(TRAIN_FILE)
    train_examples, validation_examples = split_examples(examples)

    print(f"Training examples: {len(train_examples)}")
    print(f"Validation examples: {len(validation_examples)}")

    model, backend = setup_model(device)
    train_dataset = JarvisDataset(train_examples, backend)
    validation_dataset = JarvisDataset(validation_examples, backend)

    train_model(
        model,
        backend,
        train_dataset,
        validation_dataset,
        device,
    )
    save_final_adapter(model, backend)
    print("JARVIS V6 training finished successfully.")


if __name__ == "__main__":
    main()
