import subprocess


def speak(text: str) -> None:
    """
    Speak text using the Mac's built-in text-to-speech system.
    """

    if not text.strip():
        return

    try:
        subprocess.run(
            ["say", text],
            check=True
        )

    except Exception as error:
        print(f"Text-to-speech error: {error}")


if __name__ == "__main__":
    speak("Hello. I am JARVIS.")