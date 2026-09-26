import speech_recognition as sr


def listen():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)

    print("Processing...")

    try:
        text = recognizer.recognize_google(audio)
        return text

    except sr.UnknownValueError:
        return "I couldn't understand what you said."

    except sr.RequestError as error:
        return f"Speech recognition error: {error}"


if __name__ == "__main__":
    result = listen()
    print(f"You said: {result}")