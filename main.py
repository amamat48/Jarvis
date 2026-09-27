from brain.router import chat
from brain.prompt import SYSTEM_PROMPT
from tools.registry import TOOLS, select_tools
from tools.runner import execute_tool
from tools.message_serialization import serialize_assistant_message
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
    seen_call_ids = set()

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
            seen_call_ids.clear()

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

        # Preserve the assistant response with unique tool-call IDs.
        assistant_message = serialize_assistant_message(
            response.message,
            message_index=len(messages),
            seen_call_ids=seen_call_ids,
        )
        messages.append(assistant_message)

        # Did the model request a tool?
        while assistant_message.get("tool_calls"):

            for call in assistant_message["tool_calls"]:

                function = call.get("function", {})
                tool_name = function.get("name")
                arguments = function.get("arguments", {})
                call_id = call["id"]


                print(f"[JARVIS is using {tool_name}]", flush=True)

                print(f"[Tool arguments: {arguments}]", flush=True)

                try:
                    result = execute_tool(tool_name, arguments)

                except Exception as error:
                    result = f"Tool '{tool_name}' failed: {error}"
                    print(f"[TOOL ERROR: {error}]", flush=True)

                # Give the result back to the model
                messages.append(
                    {
                        "role": "tool",
                        "tool_name": tool_name,
                        "tool_call_id": call_id,
                        "content": result
                    }
                )

            # Ask the model to respond using the tool result
            response = chat(
                messages,
                tools=list(available_tools.values())
            )

            assistant_message = serialize_assistant_message(
                response.message,
                message_index=len(messages),
                seen_call_ids=seen_call_ids,
            )
            messages.append(assistant_message)

        final_response = assistant_message.get("content") or ""

        print(f"JARVIS: {final_response}")

        if using_voice:
            speak(final_response)

if __name__ == "__main__":
    main()
