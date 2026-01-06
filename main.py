# voice_assistant.py
import json
import requests
import pyttsx3
import pyaudio
from vosk import Model, KaldiRecognizer
from core.plugin_loader import load_plugins, run_plugin
import threading
import queue

# 🌐 Настройки Ollama
LLM_URL = "http://127.0.0.1:11434/api/generate"
LLM_MODEL = "gemma3:4b"

# 🎤 Инициализация распознавания речи
model = Model("E:\\python\\PYCHARM\\UZISpeach\\vosk-model-small-ru-0.22")
recognizer = KaldiRecognizer(model, 16000)

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=4096)
stream.start_stream()

# ✅ РЕШЕНИЕ: Очередь для озвучки + выделенный поток
speech_queue = queue.Queue()


def speech_worker():
    """Рабочий поток для озвучки - инициализация TTS внутри потока"""
    # ✅ Инициализируем pyttsx3 ВНУТРИ рабочего потока
    tts = pyttsx3.init()

    # Настройка голоса
    voices = tts.getProperty('voices')
    for voice in voices:
        if "irina" in voice.name.lower():
            tts.setProperty('voice', voice.id)
            print(f"✅ Используется голос: {voice.name}")
            break

    tts.setProperty('rate', 160)

    # Обработка очереди
    while True:
        text = speech_queue.get()
        if text is None:  # Сигнал завершения
            break
        tts.say(text)
        tts.runAndWait()
        speech_queue.task_done()


# Запускаем поток озвучки
speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()


def speak(text):
    """Добавление текста в очередь озвучки (неблокирующее)"""
    speech_queue.put(text)


# ✅ РЕШЕНИЕ 2: Очистка буфера перед прослушиванием
def listen_command():
    """Прослушивание команды с очисткой буфера"""
    print("🎙️ Говори (на русском)...")

    # Очищаем накопленный буфер перед началом прослушивания
    try:
        stream.read(stream.get_read_available(), exception_on_overflow=False)
    except:
        pass

    while True:
        data = stream.read(4096, exception_on_overflow=False)
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            text = result.get("text", "")
            if text:
                print(f"📝 Распознано: {text}")
                return text


chat_history = []  # 💾 История диалога
use_stream = True
MAX_HISTORY = 20  # количество последних сообщений для контекста


def query_llm_stream(user_input):
    chat_history.append(f"User: {user_input}")
    # берем последние MAX_HISTORY сообщений
    context = chat_history[-MAX_HISTORY:]
    full_prompt = "\n".join(context) + "\nAssistant:"

    payload = {
        "model": LLM_MODEL,
        "prompt": full_prompt
    }

    try:
        response = requests.post(LLM_URL, json=payload, stream=True)

        if response.status_code != 200:
            return f"Ошибка: {response.status_code} {response.text}"

        reply = ""
        print("💬 Ответ ИИ:", end=' ', flush=True)

        for line in response.iter_lines():
            if line:
                part = json.loads(line.decode('utf-8')).get("response", "")
                print(part, end='', flush=True)
                reply += part

        print()
        chat_history.append(f"Assistant: {reply}")
        return reply

    except Exception as e:
        return f"Ошибка соединения: {str(e)}"


def main():
    # Загрузка плагинов
    plugin_handlers = load_plugins()

    # Главная функция ассистента
    print("🧠 Локальный голосовой ассистент (на русском)")
    speak("Привет, хозяин!")

    while True:
        user_input = listen_command()
        if not user_input:
            continue

        if any(word in user_input for word in ["выход", "стоп", "выключись", "закройся"]):
            speak("Пока, хозяин!")
            print("👋 Завершение работы.")
            speech_queue.put(None)  # Сигнал завершения потока озвучки
            speech_thread.join(timeout=2)  # Ждём завершения озвучки
            break

        handled = False
        for handler in plugin_handlers:
            result = handler(user_input)
            if result:
                chat_history.append(f"Assistant: {result}")
                print("🧩 Плагин:", result)
                speak(result)
                handled = True
                break

        if not handled:
            reply = query_llm_stream(user_input)
            speak(reply)


if __name__ == "__main__":
    main()
