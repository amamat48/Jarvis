def main(): 
    print("Jarvis is online.")
    print("How can I assist you today?")
    print("Type 'exit' to shut me down.\n")

while True:
    message = input("You: ")

    if message.lower() == "exit":
        print("Jarvis: Shutting down. Goodbye!")
        break

    print(f"Jarvis: You said '{message}'")

if __name__ == "__main__":
    main()