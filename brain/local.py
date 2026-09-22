import ollama


MODEL = "llama3.2:3b"


def chat(messages):
    response = ollama.chat(
        model=MODEL,
        messages=messages
    )

    return response["message"]["content"]