import os 

from brain.local import chat as local_chat

PROVIDER = os.getenv("JARVIS_PROVIDER", "local")

def chat(messages, tools=None):
    if PROVIDER == "local":
        return local_chat(messages, tools)
    else:
        raise ValueError(f"Unknown JARVIS Provider: {PROVIDER}")   

