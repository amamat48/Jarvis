from brain.local import chat
from tools.registry import TOOLS
from tools.runner import execute_tool


SYSTEM_PROMPT = """
You are JARVIS, a personal AI assistant.

You assist with engineering, programming, research, learning,
productivity, and general questions.

Be intelligent, concise, and technically accurate.
Explain concepts clearly when asked.

You have access to calculator, file-reading, and project file-listing tools.
Use the file-listing tool when you need to understand what files are available
in the project.
Use the calculator whenever the user asks you to perform arithmetic.
Use the file-reading tool when the user asks you to inspect a file.

You also have a Python execution tool.
Use it when you need to run a Python file in the JARVIS project and inspect its output or errors.
Only run Python files when appropriate for the user's request.

Tool usage rules:

- Use the calculator only for mathematical calculations.
- Do not use the calculator to inspect, debug, or execute Python code.
- Use read_file when you need to inspect source code.
- Use run_python_file when you need to execute a Python file.
- When debugging code, inspect the relevant source code and program output before drawing conclusions.
- Distinguish syntax errors, runtime errors, and logic errors.
- Do not claim that an error occurred unless the tool output actually shows an error.
- Use search_files when you need to locate a function, class, variable,
  error message, or other text across the project.
- Prefer searching for relevant code before reading large numbers of files.
- Use the line numbers returned by search_files when discussing where a
  problem occurs.

The user is an engineering student interested in electronics,
aerospace, controls, DSP, programming, AI, physics, and math.

Do not pretend to have capabilities or information you do not have.
"""


def main():

    print("JARVIS is online.")
    print("Type 'exit' to shut me down.")
    print("Type 'clear' to erase the current conversation.\n")

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    while True:

        message = input("You: ")

        if message.lower() == "exit":
            print("JARVIS: Shutting down.")
            break

        if message.lower() == "clear":

            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                }
            ]

            print("JARVIS: Conversation cleared.\n")
            continue

        messages.append(
            {
                "role": "user",
                "content": message
            }
        )

        # Ask the model what to do
        response = chat(
            messages,
            tools=list(TOOLS.values())
        )

        # Record the model's response
        messages.append(response.message)

        # Did the model request a tool?
        while response.message.tool_calls:

            for call in response.message.tool_calls:

                tool_name = call.function.name
                arguments = call.function.arguments


                print(f"[JARVIS is using {tool_name}]", flush=True)

                print(f"[Tool arguments: {arguments}]", flush=True)

                try:
                    result = execute_tool(tool_name, arguments)
                    print(f"[Tool result: {result}]", flush=True)

                except Exception as error:
                    print(f"[TOOL ERROR: {error}]", flush=True)
                    continue

                # Give the result back to the model
                messages.append(
                    {
                        "role": "tool",
                        "tool_name": tool_name,
                        "content": result
                    }
                )

            # Ask the model to respond using the tool result
            response = chat(
                messages,
                tools=list(TOOLS.values())
            )

        print(
            f"JARVIS: {response.message.content}\n",
            flush=True
        )


if __name__ == "__main__":
    main()