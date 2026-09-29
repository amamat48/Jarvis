"""Runtime adapter for the trained JARVIS V6 Mistral/PEFT model."""
from __future__ import annotations

import json
from pathlib import Path
import threading
from types import SimpleNamespace
from typing import Any, Callable

from brain.v6_format import parse_model_tool_calls


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = PROJECT_ROOT / "model" / "evaluations" / "jarvis_lora_v6" / "final"
ADAPTER_CONFIG_PATH = ADAPTER_PATH / "adapter_config.json"


def _load_v6_components():
    """Load through the V6 evaluation implementation to keep parity."""
    from model.evaluations import mistral_v4_evaluation as evaluation
    from transformers import MistralCommonBackend

    configured_base = json.loads(ADAPTER_CONFIG_PATH.read_text(encoding="utf-8"))[
        "base_model_name_or_path"
    ]
    if configured_base != evaluation.MODEL_NAME:
        raise ValueError(
            "V6 adapter base model does not match the evaluation loader: "
            f"{configured_base!r} != {evaluation.MODEL_NAME!r}"
        )

    model = evaluation.load_model()
    backend = MistralCommonBackend.from_pretrained(evaluation.MODEL_NAME)
    return model, backend, evaluation


class V6Provider:
    """Translate Mistral V6 generations into the existing runtime message shape."""

    def __init__(self, component_loader: Callable[[], tuple[Any, Any, Any]] | None = None):
        if not ADAPTER_CONFIG_PATH.is_file():
            raise FileNotFoundError(f"V6 adapter configuration not found: {ADAPTER_CONFIG_PATH}")
        self.adapter_config = json.loads(ADAPTER_CONFIG_PATH.read_text(encoding="utf-8"))
        self.base_model_name = self.adapter_config.get("base_model_name_or_path")
        if not self.base_model_name:
            raise ValueError(f"V6 adapter configuration has no base model: {ADAPTER_CONFIG_PATH}")

        self._component_loader = component_loader or _load_v6_components
        self._model = None
        self._backend = None
        self._evaluation = None
        self._generation_lock = threading.Lock()
        self._call_sequence = 0
        self._used_call_ids: set[str] = set()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self._model, self._backend, self._evaluation = self._component_loader()

    def chat(self, messages: list[dict[str, Any]], tools=None):
        # The shared Mistral model is serialized across concurrent tasks.
        with self._generation_lock:
            self._ensure_loaded()
            raw_response = self._evaluation.generate_response(
                self._model,
                self._backend,
                messages,
                tools=tools,
            )
            assistant_message = self._to_assistant_message(raw_response)
        return SimpleNamespace(message=assistant_message)

    def _to_assistant_message(self, raw_response: str) -> dict[str, Any]:
        if not isinstance(raw_response, str):
            raise TypeError("V6 generation must return decoded text.")

        parsed_calls = parse_model_tool_calls(raw_response)
        if parsed_calls:
            tool_calls = []
            for call in parsed_calls:
                explicit_id = call.get("id")
                if explicit_id is not None and (not isinstance(explicit_id, str) or not explicit_id):
                    raise ValueError("V6 tool-call IDs must be non-empty strings.")
                if explicit_id is None:
                    while True:
                        self._call_sequence += 1
                        explicit_id = f"v6_call_{self._call_sequence}"
                        if explicit_id not in self._used_call_ids:
                            break
                self._used_call_ids.add(explicit_id)

                arguments = call.get("arguments", {})
                if not isinstance(arguments, dict):
                    raise ValueError("V6 tool-call arguments must be an object.")
                tool_calls.append({
                    "id": explicit_id,
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": arguments,
                    },
                })
            return {"role": "assistant", "content": None, "tool_calls": tool_calls}

        content = raw_response.replace("</s>", "").replace("<|im_end|>", "").strip()
        return {"role": "assistant", "content": content}
