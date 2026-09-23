import ollama


def calculate(expression: str) -> str:
    """
    Calculate a mathematical expression.

    Args:
        expression: The mathematical expression to calculate.

    Returns:
        The result of the calculation.
    """
    return str(eval(expression, {"__builtins__": {}}, {}))


messages = [
    {
        "role": "user",
        "content": "What is 25 * 18?"
    }
]


response = ollama.chat(
    model="llama3.2:3b",
    messages=messages,
    tools=[calculate]
)


print("RESPONSE:")
print(response)

print("\nTOOL CALLS:")
print(response.message.tool_calls)