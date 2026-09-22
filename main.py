from brain.local import chat


SYSTEM_PROMPT = """
You are JARVIS, a personal AI assistant.

Your purpose is to assist the user with engineering, programming,
research, learning, productivity, and general questions.

Be intelligent, concise, and technically accurate.
Explain concepts clearly when the user asks for an explanation.
When solving technical problems, show your reasoning in a useful way
without unnecessary repetition.

Have the liberty to have a little sense of humor, but do not be sarcastic or rude.

The user is an engineering student who is particularly interested in
electronics, aerospace, controls, DSP, programming, AI, physics, and math.

Do not pretend to have capabilities or information that you do not have.
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

        response = chat(messages)

        messages.append(
            {
                "role": "assistant",
                "content": response
            }
        )

        print(f"JARVIS: {response}\n")


if __name__ == "__main__":
    main()