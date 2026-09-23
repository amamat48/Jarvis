from memory.store import remember, recall

from memory.store import remember as store_memory
from memory.store import recall as load_memories


def remember_memory(key: str, value: str) -> str:
    """
    Permanently remember information about the user or their projects.

    Only use this when the user explicitly asks JARVIS to remember something.
    Never store passwords, API keys, tokens, or other secrets.
    """
    return store_memory(key, value)


def recall_memory() -> str:
    """
    Retrieve JARVIS's persistent memories.
    """
    return load_memories()