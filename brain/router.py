import os
import threading

PROVIDER = os.getenv("JARVIS_PROVIDER", "v6").strip().lower()

_v6_provider = None
_v6_provider_lock = threading.Lock()


def _get_v6_provider():
    global _v6_provider
    if _v6_provider is None:
        with _v6_provider_lock:
            if _v6_provider is None:
                from brain.v6_provider import V6Provider

                _v6_provider = V6Provider()
    return _v6_provider

def chat(messages, tools=None):
    if PROVIDER == "local":
        from brain.local import chat as local_chat

        return local_chat(messages, tools)
    if PROVIDER == "v6":
        return _get_v6_provider().chat(messages, tools)
    raise ValueError(f"Unknown JARVIS Provider: {PROVIDER}")

