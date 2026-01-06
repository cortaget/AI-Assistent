# voice_assistant.py - ЧИСТАЯ ВЕРСИЯ
import json
import requests
import pyttsx3
import pyaudio
from vosk import Model, KaldiRecognizer
from core.plugin_loader import load_plugins
import threading
import queue
from memory_manager import MemoryManager

LLM_URL = "http://127.0.0.1:11434/api/generate"
LLM_MODEL = "gemma3:4b"

memory = MemoryManager()
memory.debug_mode = False  # Отключаем логи

model = Model("E:\\python\\PYCHARM\\UZISpeach\\vosk-model-small-ru-0.22")
recognizer = KaldiRecognizer(model, 16000)

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=4096)
stream.start_stream()

speech_queue = queue.Queue()


def speech_worker():
    tts = pyttsx3.init()
    voices = tts.getProperty('voices')
    for voice in voices:
        if "irina" in voice.name.lower():
            tts.setProperty('voice', voice.id)
            break
    tts.setProperty('rate', 160)

    while True:
        text = speech_queue.get()
        if text is None:
            break
        tts.say(text)
        tts.runAndWait()
        speech_queue.task_done()


speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()


def speak(text):
    speech_queue.put(text)


def listen_command():
    print("🎙️ Говори...")
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
                print(f"📝 {text}")
                return text


def auto_save_memory(user_input: str, assistant_reply: str):
    """ИИ само решает сохранять или нет"""

    # Быстрая проверка: есть ли важная информация?
    check_prompt = f"""Диалог:
Пользователь: {user_input}
Ассистент: {assistant_reply}

Содержит ли сообщение пользователя важную личную информацию для долгосрочного запоминания (имя, работа, предпочтения, увлечения, правила)?

Ответь ТОЛЬКО "ДА" или "НЕТ"."""

    payload = {
        "model": LLM_MODEL,
        "prompt": check_prompt,
        "stream": False,
        "options": {"temperature": 0.1}
    }

    try:
        response = requests.post(LLM_URL, json=payload, timeout=10)
        if response.status_code == 200:
            decision = response.json().get("response", "").strip().upper()

            if "ДА" in decision:
                # Извлекаем факт
                extract_prompt = f"""Из этого сообщения извлеки ОДИН короткий факт о пользователе:

Пользователь: {user_input}

Формат ответа: "Пользователь [глагол] [объект]"
Пример: "Пользователь любит лошадей"

Ответь ТОЛЬКО фактом без пояснений."""

                payload2 = {
                    "model": LLM_MODEL,
                    "prompt": extract_prompt,
                    "stream": False,
                    "options": {"temperature": 0.1}
                }

                response2 = requests.post(LLM_URL, json=payload2, timeout=10)
                if response2.status_code == 200:
                    fact = response2.json().get("response", "").strip()

                    # Убираем кавычки и лишнее
                    fact = fact.strip('"\'').strip()

                    if len(fact) > 10 and "пользовател" in fact.lower():
                        # Сохраняем напрямую
                        memory.add_memory(fact, memory_type="user_info")
                        print(f"💾 {fact}")

    except Exception as e:
        pass  # Молча игнорируем ошибки


def query_llm_stream(user_input):
    """Запрос с автосохранением"""

    # Поиск в памяти
    relevant_memories = memory.search_memory(user_input, top_k=2)

    # Контекст памяти
    memory_context = ""
    if relevant_memories:
        filtered = [m for m in relevant_memories if m['relevance'] > 0.4]
        if filtered:
            memory_context = "[Важная информация из памяти]:\n"
            for mem in filtered:
                memory_context += f"- {mem['content']}\n"
            memory_context += "\n"

    # Промпт
    full_prompt = memory_context + f"User: {user_input}\nAssistant:"

    payload = {
        "model": LLM_MODEL,
        "prompt": full_prompt,
        "stream": True
    }

    try:
        response = requests.post(LLM_URL, json=payload, stream=True)

        if response.status_code != 200:
            return f"Ошибка: {response.status_code}"

        reply = ""
        print("💬 ", end='', flush=True)

        for line in response.iter_lines():
            if line:
                part = json.loads(line.decode('utf-8')).get("response", "")
                print(part, end='', flush=True)
                reply += part

        print()

        # ✅ ИИ само решает сохранять или нет
        auto_save_memory(user_input, reply)

        return reply

    except Exception as e:
        return f"Ошибка: {str(e)}"


def handle_memory_commands(user_input: str) -> bool:
    """Команды памяти"""
    lower_input = user_input.lower()

    if "запомни" in lower_input:
        content = user_input.split("запомни", 1)[-1].strip()
        if content:
            memory.add_memory(content, memory_type="user_info")
            speak("Запомнил")
            return True

    if "что ты помнишь" in lower_input or "покажи память" in lower_input:
        memories = memory.list_all_memories()
        if memories:
            print(f"\n💾 Память ({len(memories)} записей):")
            for i, mem in enumerate(memories, 1):
                print(f"{i}. {mem['content']}")
            speak(f"Я помню {len(memories)} записей")
        else:
            speak("Память пуста")
        return True

    if "очисти память" in lower_input:
        memory.clear_all_memories()
        speak("Память очищена")
        return True

    return False


def main():
    plugin_handlers = load_plugins()

    speak("Привет")

    while True:
        user_input = listen_command()
        if not user_input:
            continue

        if any(word in user_input for word in ["выход", "стоп", "выключись"]):
            speak("Пока")
            speech_queue.put(None)
            speech_thread.join(timeout=2)
            break

        #пока не надо это трогать
        #if handle_memory_commands(user_input):
        #    continue

        handled = False
        for handler in plugin_handlers:
            result = handler(user_input)
            if result:
                speak(result)
                handled = True
                break

        if not handled:
            reply = query_llm_stream(user_input)
            speak(reply)


if __name__ == "__main__":
    main()