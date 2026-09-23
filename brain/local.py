import os
import ollama


MODEL = os.getenv("JARVIS_LOCAL_MODEL", "llama3.2:3b")


def chat(messages, tools=None):
    response = ollama.chat(
        model=MODEL,
        messages=messages,
        tools=tools or []
    )

    return response