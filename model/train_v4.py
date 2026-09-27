import gc
import json
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


# ============================================================
# JARVIS V4 QLoRA Training
# Base model: Ministral 3 8B Reasoning 2512
# ============================================================

MODEL_NAME = "mistralai/Ministral-3-8B-Reasoning-2512"

BASE_DIR = Path(__file__).resolve().parent

TRAIN_FILE = BASE_DIR / "training_data" / "v3" / "mlx" / "train.jsonl"
VAL_FILE = BASE_DIR / "training_data" / "v3" / "mlx" / "valid.jsonl"

OUTPUT_DIR = BASE_DIR / "evaluations" / "jarvis_lora_v4"

MAX_LENGTH = 1024
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 1e-5

TOTAL_ITERATIONS = 240
VALIDATION_INTERVAL = 20
CHECKPOINT_INTERVAL = 40

SEED = 42


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Utility
# ============================================================

def print_memory(label):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        peak = torch.cuda.max_memory_allocated() / 1024**3

        print(
            f"[MEMORY] {label}: "
            f"allocated={allocated:.2f} GB | "
            f"reserved={reserved:.2f} GB | "
            f"peak={peak:.2f} GB"
        )


# ============================================================
# Tool-message normalization
#
# The original V3 JSONL is NEVER modified.
# Everything here happens in memory.
# ============================================================

def normalize_tool_messages(messages):
    normalized = []
    pending_tool_ids = []

    for message_index, original_message in enumerate(messages):
        message = dict(original_message)

        # ----------------------------------------------------
        # Assistant tool calls
        # ----------------------------------------------------
        if message.get("role") == "assistant" and message.get("tool_calls"):
            tool_calls = []

            for call_index, original_call in enumerate(
                message["tool_calls"]
            ):
                call = dict(original_call)

                call_id = call.get("id")

                if not call_id:
                    call_id = (
                        f"jarvis_tool_{message_index}_{call_index}"
                    )

                call["id"] = call_id

                if "type" not in call:
                    call["type"] = "function"

                function = dict(
                    call.get("function", {})
                )

                if "name" not in function:
                    function["name"] = "unknown"

                arguments = function.get(
                    "arguments",
                    {},
                )

                if not isinstance(arguments, str):
                    arguments = json.dumps(
                        arguments,
                        ensure_ascii=False,
                    )

                function["arguments"] = arguments
                call["function"] = function

                tool_calls.append(call)
                pending_tool_ids.append(call_id)

            message["tool_calls"] = tool_calls

        # ----------------------------------------------------
        # Tool results
        # ----------------------------------------------------
        elif message.get("role") == "tool":
            tool_call_id = (
                message.get("tool_call_id")
                or message.get("id")
            )

            if not tool_call_id:
                if pending_tool_ids:
                    tool_call_id = pending_tool_ids.pop(0)
                else:
                    tool_call_id = (
                        f"jarvis_tool_result_{message_index}"
                    )

            message["tool_call_id"] = tool_call_id

            # The tool_call_id is what Mistral expects for
            # the relationship between the tool result and
            # the assistant's tool call.
            message.pop("id", None)

        normalized.append(message)

    return normalized


# ============================================================
# Dataset
# ============================================================

class JarvisDataset(Dataset):

    def __init__(self, path, backend):
        self.backend = backend
        self.examples = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                item = json.loads(line)

                messages = item.get("messages")

                if messages is None:
                    continue

                messages = normalize_tool_messages(messages)

                self.examples.append(
                    {
                        "messages": messages,
                        "tools": item.get("tools"),
                    }
                )

        print(
            f"Loaded {len(self.examples)} examples from: {path}"
        )

        print("Tokenizing dataset...")

        self.tokens = []

        for index, example in enumerate(self.examples):

            try:
                messages = example["messages"]

                # ------------------------------------------------
                # Mistral Common requires different handling
                # depending on the final role.
                #
                # Assistant final message:
                #   Treat it as an assistant prefix/prefill.
                #
                # User/tool final message:
                #   Treat it as a completed conversation that
                #   will normally receive an assistant response.
                # ------------------------------------------------
                final_role = messages[-1].get("role")

                continue_final_message = (
                    final_role == "assistant"
                )

                result = self.backend.apply_chat_template(
                    messages,
                    tools=example["tools"],
                    add_generation_prompt=False,
                    continue_final_message=(
                        continue_final_message
                    ),
                    tokenize=True,
                    truncation=True,
                    max_length=MAX_LENGTH,
                    return_tensors="pt",
                    return_dict=True,
                )

                input_ids = result["input_ids"]

                if input_ids.dim() == 2:
                    input_ids = input_ids[0]

                input_ids = input_ids.to(torch.long)

                self.tokens.append(input_ids)

            except Exception as exc:
                print()
                print("TOKENIZATION ERROR")
                print(f"Example: {index}")
                print(
                    f"Final role: "
                    f"{example['messages'][-1].get('role')}"
                )
                print(
                    f"Error: "
                    f"{type(exc).__name__}: {exc}"
                )
                print()
                raise

        print(
            f"Tokenization complete: "
            f"{len(self.tokens)} examples"
        )

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        input_ids = self.tokens[index]

        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
        }


# ============================================================
# Padding
# ============================================================

def collate_batch(batch, pad_token_id):
    max_length = max(
        item["input_ids"].shape[0]
        for item in batch
    )

    input_ids = []
    labels = []

    for item in batch:
        ids = item["input_ids"]

        padding = max_length - ids.shape[0]

        if padding > 0:
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

        if padding > 0:
            label[-padding:] = -100

        labels.append(label)

    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels)

    attention_mask = input_ids.ne(
        pad_token_id
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ============================================================
# Model
# ============================================================

print()
print("=" * 60)
print("JARVIS V4 TRAINING")
print("=" * 60)
print()

print(f"Model: {MODEL_NAME}")
print(f"Max length: {MAX_LENGTH}")
print(f"Iterations: {TOTAL_ITERATIONS}")
print(f"Learning rate: {LEARNING_RATE}")
print()

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available."
    )

device = torch.device("cuda:0")

print(f"Using device: {device}")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

print()
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


# ============================================================
# Mistral Common tokenizer backend
# ============================================================

print()
print("Loading Mistral Common tokenizer backend...")

backend = MistralCommonBackend.from_pretrained(
    MODEL_NAME
)

print("Tokenizer backend loaded")


# ============================================================
# QLoRA preparation
# ============================================================

print()
print("Preparing model for QLoRA...")

model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
    gradient_checkpointing_kwargs={
        "use_reentrant": False,
    },
)

model.config.use_cache = False

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "v_proj",
    ],
)

model = get_peft_model(
    model,
    lora_config,
)

model.print_trainable_parameters()

print_memory("After LoRA setup")


# ============================================================
# Dataset
# ============================================================

train_dataset = JarvisDataset(
    TRAIN_FILE,
    backend,
)

val_dataset = JarvisDataset(
    VAL_FILE,
    backend,
)

print()
print(
    f"Training examples: "
    f"{len(train_dataset)}"
)

print(
    f"Validation examples: "
    f"{len(val_dataset)}"
)


# ============================================================
# Optimizer
# ============================================================

trainable_parameters = [
    parameter
    for parameter in model.parameters()
    if parameter.requires_grad
]

optimizer = torch.optim.AdamW(
    trainable_parameters,
    lr=LEARNING_RATE,
)


# ============================================================
# Validation
# ============================================================

def evaluate():
    model.eval()

    total_loss = 0.0
    count = 0

    with torch.no_grad():

        for index in range(len(val_dataset)):

            batch = collate_batch(
                [val_dataset[index]],
                backend.pad_token_id,
            )

            input_ids = batch[
                "input_ids"
            ].to(device)

            attention_mask = batch[
                "attention_mask"
            ].to(device)

            labels = batch[
                "labels"
            ].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            total_loss += outputs.loss.item()
            count += 1

    model.train()

    if count == 0:
        return float("nan")

    return total_loss / count


# ============================================================
# Training
# ============================================================

print()
print("=" * 60)
print("STARTING TRAINING")
print("=" * 60)
print()

model.train()

start_time = time.time()
iteration = 0

while iteration < TOTAL_ITERATIONS:

    order = list(
        range(len(train_dataset))
    )

    random.shuffle(order)

    for dataset_index in order:

        if iteration >= TOTAL_ITERATIONS:
            break

        batch = collate_batch(
            [train_dataset[dataset_index]],
            backend.pad_token_id,
        )

        input_ids = batch[
            "input_ids"
        ].to(device)

        attention_mask = batch[
            "attention_mask"
        ].to(device)

        labels = batch[
            "labels"
        ].to(device)

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss

        loss.backward()

        optimizer.step()

        iteration += 1

        if iteration == 1 or iteration % 10 == 0:

            elapsed = (
                time.time() - start_time
            )

            print(
                f"Iteration "
                f"{iteration:3d}/"
                f"{TOTAL_ITERATIONS} | "
                f"Loss: "
                f"{loss.item():.4f} | "
                f"Time: "
                f"{elapsed / 60:.1f} min"
            )

        if (
            iteration
            % VALIDATION_INTERVAL
            == 0
        ):

            validation_loss = evaluate()

            print(
                f"[VALIDATION] "
                f"Iteration "
                f"{iteration} | "
                f"Loss: "
                f"{validation_loss:.4f}"
            )

        if (
            iteration
            % CHECKPOINT_INTERVAL
            == 0
        ):

            checkpoint_dir = (
                OUTPUT_DIR
                / f"checkpoint-{iteration}"
            )

            checkpoint_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            model.save_pretrained(
                checkpoint_dir
            )

            backend.save_pretrained(
                checkpoint_dir
            )

            print(
                f"[CHECKPOINT] Saved: "
                f"{checkpoint_dir}"
            )

            gc.collect()
            torch.cuda.empty_cache()


# ============================================================
# Final save
# ============================================================

print()
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

final_dir = (
    OUTPUT_DIR / "final"
)

final_dir.mkdir(
    parents=True,
    exist_ok=True,
)

model.save_pretrained(
    final_dir
)

backend.save_pretrained(
    final_dir
)

print()
print("Final adapter saved to:")
print(final_dir)

print_memory("Final")

print()
print(
    "JARVIS V4 training finished successfully."
)