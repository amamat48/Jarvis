from pathlib import Path
import json


MEMORY_FILE = Path(__file__).resolve().parent / "memory.json"


def load_memory() -> dict:
    if not MEMORY_FILE.exists():
        return {}

    try:
        return json.loads(MEMORY_FILE.read_text())

    except Exception:
        return {}


def save_memory(memory: dict) -> None:
    MEMORY_FILE.write_text(
        json.dumps(memory, indent=4)
    )


def remember(key: str, value: str) -> str:
    memory = load_memory()

    memory[key] = value

    save_memory(memory)

    return f"Stored memory: {key} = {value}"


def recall() -> str:
    memory = load_memory()

    if not memory:
        return "No persistent memories stored."

    return json.dumps(memory, indent=4)