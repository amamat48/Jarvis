from memory.store import remember, recall

from memory.store import remember as store_memory
from memory.store import recall as load_memories


def remember_memory(key: str, value: str) -> str:
    """
    Store information in persistent memory.

    IMPORTANT:
    Only call this tool when the user's current request explicitly
    asks JARVIS to remember, save, store, or retain information.

    Do NOT call this tool for:
    - greetings
    - casual conversation
    - JARVIS's own statements
    - normal questions
    - information that the user did not explicitly ask to save

    Never store passwords, API keys, tokens, or other secrets.
    """
    return store_memory(key, value)


def recall_memory(key: str = "") -> str:
    """
    Retrieve persistent memories.

    
    return all persistent memories.

    Only use this tool when the user explicitly asks JARVIS
    what it remembers or asks to recall stored information.
    """
    memories = load_memories()

    if key:
        try:
            import json

            data = json.loads(memories)

            if key not in data:
                return f"No memory found for key: {key}"

            return json.dumps({key: data[key]}, indent=4)

        except Exception as error:
            return f"Error reading memory: {error}"

    return memories