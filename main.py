from brain.router import chat
from tools.registry import TOOLS, select_tools
from tools.runner import execute_tool
from voice_input import listen
from voice_output import speak



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

        using_voice = False

        if message.lower() == "voice":
            using_voice = True
            message = listen()
            print(f"You (voice): {message}")

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

        available_tools = select_tools(message)

        # Ask the model what to do
        response = chat(
            messages,
            tools=list(available_tools.values())
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
                tools=list(available_tools.values())
            )

        final_response = response.message.content

        print(f"JARVIS: {final_response}")

        if using_voice:
            speak(final_response)

if __name__ == "__main__":
    main()