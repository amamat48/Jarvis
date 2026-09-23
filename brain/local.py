import ollama


MODEL = "llama3.2:3b"


def chat(messages, tools=None):
    response = ollama.chat(
        model=MODEL,
        messages=messages,
        tools=tools or []
    )

    return response