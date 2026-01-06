import pyttsx3

engine = pyttsx3.init()
voices = engine.getProperty('voices')

for i, voice in enumerate(voices):
    print(f"{i}: {voice.name} | {voice.id} | {voice.languages}")




engine = pyttsx3.init()
voices = engine.getProperty('voices')

# Попробуем выбрать голос с русским языком
for voice in voices:
    if "ru" in voice.languages or "russian" in voice.name.lower():
        print(f"✅ Используем голос: {voice.name}")
        engine.setProperty('voice', voice.id)
        break

engine.setProperty('rate', 160)  # Скорость речи
engine.say("Привет, как твои дела?")
engine.runAndWait()
