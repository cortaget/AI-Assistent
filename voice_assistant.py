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
from tools.tool_manager import ToolManager
from typing import List, Dict
LLM_URL = "http://127.0.0.1:11434/api/generate"
LLM_MODEL = "gemma3:4b"

memory = MemoryManager()
memory.debug_mode = False  # Отключаем логи

tool_manager = ToolManager(
    tools_dir="tools",
    llm_url=LLM_URL,
    llm_model=LLM_MODEL
)

model = Model("models\\vosk-model-small-ru-0.22")
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


import re
import threading

_memory_processing = {}
_memory_lock = threading.Lock()

_memory_processing = {}
_memory_lock = threading.Lock()


def auto_memory_manager(user_input: str, assistant_reply: str):
    """Управление памятью через векторный поиск"""

    # Защита от дублирования
    request_id = f"{user_input[:50]}_{assistant_reply[:50]}"
    with _memory_lock:
        if request_id in _memory_processing:
            return
        _memory_processing[request_id] = True

    try:
        _process_memory_vector(user_input, assistant_reply)
    finally:
        with _memory_lock:
            _memory_processing.pop(request_id, None)


def _process_memory_vector(user_input: str, assistant_reply: str):
    """Обработка памяти через векторный поиск"""

    # ШАГ 1: LLM определяет намерение
    intent_prompt = f"""Диалог:
Пользователь: {user_input}
Ассистент: {assistant_reply}

Что хочет пользователь?
A) УДАЛИТЬ информацию из памяти (забудь, удали, сотри)
B) СОХРАНИТЬ информацию (даёт факты о себе)
C) ОБЫЧНЫЙ ВОПРОС

Ответь ТОЛЬКО: A, B или C"""

    intent = _quick_llm_call(intent_prompt, max_tokens=5)

    # ВЕТКА УДАЛЕНИЯ
    if "A" in intent.upper():
        print("🗑️ Обработка удаления через векторный поиск...")

        query_prompt = f"""Из фразы '{user_input}' извлеки ключевые слова для поиска в базе памяти.
Включи контекст (например: "забудь что меня зовут Максим" → "Имя пользователя Максим" или "зовут Максим").

Ответь 2-5 словами:"""
        search_query = _quick_llm_call(query_prompt, max_tokens=20).strip('"\'')

        if len(search_query) < 2:
            print("⚠️ Не удалось понять что удалять")
            return

        print(f"🔍 Векторный поиск: '{search_query}'")

        results = memory.search_memory(search_query, top_k=3)

        if results:
            print(f"📋 Найдено через векторный поиск:")
            for i, r in enumerate(results[:3], 1):
                print(f"  {i}. [{r['relevance']:.2f}] {r['content']}")

            if results[0]['relevance'] > 0.65:
                memory.delete_memory(results[0]['id'])
                print(f"🗑️ Удалено: {results[0]['content']}")
            else:
                print(f"⚠️ Релевантность слишком низкая: {results[0]['relevance']:.2f}")
        else:
            print("❌ Ничего не найдено")
        return

    # ВЕТКА СОХРАНЕНИЯ
    if "B" in intent.upper():
        print("💾 Обработка сохранения...")

        extract_prompt = f"""Диалог:
Пользователь: {user_input}
Ассистент: {assistant_reply}

Извлеки ОДИН факт о пользователе. Если это вопрос - ответь "НЕТ".

Примеры:
- "Привет, я Максим" → "Имя пользователя: Максим"
- "Я люблю кошек" → "Пользователь любит кошек"
- "Как дела?" → "НЕТ"

Ответь ТОЛЬКО фактом или "НЕТ"."""

        fact = _quick_llm_call(extract_prompt, max_tokens=30).strip('"\'')

        if not fact or "НЕТ" in fact.upper() or len(fact) < 7:
            print("⏭️ Нет фактов для сохранения")
            return

        print(f"📝 Извлечённый факт: {fact}")

        similar = memory.search_memory(fact, top_k=1)

        if similar and len(similar) > 0:
            relevance = similar[0]['relevance']
            print(f"🔍 Похожая запись: {relevance:.2f}")

            if relevance > 0.92:
                print(f"⚠️ Точный дубликат, пропускаем")
                return
            elif relevance > 0.80:
                print(f"⚠️ Очень похожая запись, пропускаем")
                return
            elif relevance > 0.70:
                memory.update_memory(similar[0]['id'], fact)
                print(f"🔄 Обновлено: {fact}")
                return

        memory.add_memory(fact, memory_type="user_info")
        print(f"💾 Сохранено: {fact}")

def _quick_llm_call(prompt: str, max_tokens: int = 50) -> str:
    """Быстрый вызов LLM"""
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": max_tokens,
            "top_k": 10,
            "top_p": 0.5
        }
    }

    try:
        response = requests.post(LLM_URL, json=payload, timeout=100)
        if response.status_code == 200:
            return response.json().get("response", "").strip()
    except:
        pass
    return ""


def _extract_fact_parallel(user_input: str, assistant_reply: str) -> str:
    """Параллельное извлечение факта (объединённый запрос)"""
    combined_prompt = f"""Диалог:
Пользователь: {user_input}
Ассистент: {assistant_reply}

Задача: Если пользователь ДАЁТ информацию о себе (не спрашивает), извлеки ОДИН короткий факт.
Если это вопрос - ответь "НЕТ".

Формат:
- Имя: "Имя пользователя: [имя]"
- Предпочтение: "Пользователь любит [что-то]"
- Работа: "Пользователь работает [где]"

Ответь ТОЛЬКО фактом или "НЕТ"."""

    fact = _quick_llm_call(combined_prompt, max_tokens=256)

    if fact and "НЕТ" not in fact.upper() and len(fact) > 7:
        return fact
    return ""


def query_llm_stream(user_input):
    """Запрос с автоматическим определением необходимости инструментов"""

    # === ШАГ 1: LLM решает, нужны ли инструменты ===
    print("🔍 Анализ запроса...")
    decision = tool_manager.decide_tool_usage(user_input)

    print(f"💭 Решение: {'Использовать инструменты' if decision['needs_tools'] else 'Обычный диалог'}")
    print(f"   Причина: {decision['reasoning']}")

    if "ошибка" in decision['reasoning'].lower():
        print("⚠️ Роутер недоступен, обрабатываю память в фоне...")

    if decision['needs_tools'] and decision['suggested_tools']:
        # === ШАГ 2: Получаем полную информацию о выбранных инструментах ===
        relevant_tools = []
        for tool_name in decision['suggested_tools']:
            if tool_name in tool_manager.tools:
                relevant_tools.append({
                    "name": tool_name,
                    "tool": tool_manager.tools[tool_name],
                    "score": 1.0  # Инструмент выбран LLM напрямую
                })

        if relevant_tools:
            print(f"🔧 Выбрано инструментов: {[t['name'] for t in relevant_tools]}\n")
            reply = query_llm_with_tools(user_input, relevant_tools)
            auto_memory_manager(user_input, reply)
            return reply

    # === ШАГ 3: Обычный диалог без инструментов ===
    print("💬 Обработка без инструментов...\n")

    relevant_memories = memory.search_memory(user_input, top_k=10)
    system_prompt = """Ты - голосовой ИИ-ассистент, который ведёт диалог с пользователем.
ВАЖНО понимать:
- ТЫ - это ассистент (искусственный интеллект)
- ПОЛЬЗОВАТЕЛЬ - это человек, который с тобой общается
"""

    memory_context = ""
    if relevant_memories:
        filtered = [m for m in relevant_memories if m['relevance'] > 0.3]
        if filtered:
            memory_context = "[Информация из долговременной памяти]:\n"
            for mem in filtered:
                memory_context += f"- {mem['content']}\n"
            memory_context += "\n"

    full_prompt = system_prompt + memory_context + f"Пользователь: {user_input}\nАссистент:"

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

        auto_memory_manager(user_input, reply)


        return reply
    except Exception as e:
        return f"Ошибка: {str(e)}"


def query_llm_with_tools(user_input: str, relevant_tools: List[Dict], max_iterations: int = 5):
    """
    Многошаговое выполнение с инструментами (ReAct pattern)
    """
    # Формируем описание инструментов
    tools_desc = tool_manager.get_tools_description(relevant_tools)

    system_prompt = f"""Ты - ИИ-ассистент с доступом к инструментам.

{tools_desc}

ПРАВИЛА:
1. Анализируй задачу пошагово
2. Используй инструменты через формат: TOOL[название_инструмента](параметры)
3. ВАЖНО: После TOOL[...] НЕ ПИШИ результат! Жди, система сама вернёт его
4. После получения реального результата - дай финальный ответ с ANSWER[...]

Примеры:
User: Сколько будет 5+3?
Assistant: TOOL[calculator](5+3)
System: [calculator]: 8
Assistant: ANSWER[5 плюс 3 равно 8]

User: Сколько сейчас времени?
Assistant: TOOL[get_time]()
System: [get_time]: 15:30:14, 14.11.2025
Assistant: ANSWER[Сейчас 15 часов 30 минут 14 секунд, 14 ноября 2025 года]

Запрос пользователя: {user_input}
"""

    # Итеративное выполнение
    for iteration in range(max_iterations):
        print(f"\n🔄 Итерация {iteration + 1}/{max_iterations}")

        payload = {
            "model": LLM_MODEL,
            "prompt": system_prompt,
            "stream": False,
            "options": {"temperature": 0.3}
        }

        try:
            response = requests.post(LLM_URL, json=payload, timeout=100)
            if response.status_code != 200:
                return "Ошибка связи с LLM"

            llm_response = response.json().get("response", "").strip()
            print(f"🤖 LLM: {llm_response}")

            # Проверяем финальный ответ
            if "ANSWER[" in llm_response:
                answer = llm_response.split("ANSWER[")[1].split("]")[0]
                print(f"✅ Финальный ответ: {answer}")
                return answer

            # Извлекаем вызовы инструментов
            tool_calls = extract_tool_calls(llm_response)

            if not tool_calls:
                # Проверка на фейковые результаты
                if "[Получен результат:" in llm_response or "[Результат:" in llm_response:
                    print("⚠️ LLM пытается выдумать результат!")
                    return "Ошибка: попытка генерации фейкового результата"
                # LLM не может продолжить
                return llm_response

            # Выполняем все вызовы
            results = []
            for tool_call in tool_calls:
                tool_name = tool_call['tool']
                params = tool_call['params']

                if tool_name not in tool_manager.tools:
                    result = f"❌ Инструмент '{tool_name}' не найден"
                else:
                    result = tool_manager.execute_tool(
                        tool_name,
                        params=params,
                        user_input=user_input
                    )

                result_str = f"[{tool_name}]: {result}"
                results.append(result_str)
                print(f"  🔧 {tool_name}({params}) → {result}")

            # Обновляем промпт с РЕАЛЬНЫМИ результатами
            system_prompt += f"\nAssistant: {llm_response}\n"
            system_prompt += f"System: {chr(10).join(results)}\n"
            system_prompt += "Assistant: "

        except Exception as e:
            print(f"⚠️ Ошибка: {e}")
            return f"Ошибка выполнения: {e}"

    return "Превышено максимальное количество шагов"


def extract_tool_calls(text: str) -> List[Dict]:
    """
    Извлекает вызовы инструментов из текста LLM
    Формат: TOOL[tool_name](params)
    """
    import re
    pattern = r'TOOL\[(\w+)\]\(([^)]*)\)'
    matches = re.findall(pattern, text)

    calls = []
    for tool_name, params_str in matches:
        # Если параметры пустые - возвращаем пустой словарь
        if not params_str or params_str.strip() == "":
            params = {}
        else:
            # Для калькулятора передаём expression
            params = {"expression": params_str.strip()}

        calls.append({"tool": tool_name, "params": params})

    return calls

"""=== Команды памяти ==="""

#не используется
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


def next_prompt(user_input: str, use_voice: bool = False) -> str:
    """
    Простая функция для тестирования через код

    Args:
        user_input: Текст запроса
        use_voice: Озвучивать или нет (по умолчанию False)

    Returns:
        Ответ ассистента
    """
    print(f"\n👤 {user_input}")

    # Проверка команд памяти
    #if handle_memory_commands(user_input):
    #    return "[Команда выполнена]"

    # Проверка плагинов
    plugin_handlers = load_plugins()
    for handler in plugin_handlers:
        result = handler(user_input)
        if result:
            if use_voice:
                speak(result)
            return result

    # Основной запрос
    reply = query_llm_stream(user_input)
    if use_voice:
        speak(reply)

    return reply




def main():



    """
    тестирование запоминалки
    next_prompt("запомни пожалуйста что я люблю лошадей", use_voice=False)
    #next_prompt("какое животное мне нравятся?(ответь коротко)", use_voice=False)
    #next_prompt("сколько сейчас времени?")

    #next_prompt("как меня зовут", use_voice=False)
    """

    """
    проверка инструмента времени
    
    # Тест инструмента времени
    # Тест 1: Время (ключевое слово + контекст)
    next_prompt("сколько сейчас времени", use_voice=False)

    print("\n" + "=" * 50 + "\n")

    # Тест 2: Ложное срабатывание (должно НЕ сработать)
    next_prompt("у меня на сегодня запланирована дата", use_voice=False)

    print("\n" + "=" * 50 + "\n")

    # Тест 3: Семантика без ключевых слов
    next_prompt("какое сегодня число", use_voice=False)
    print("\n" + "=" * 50 + "\n")

    next_prompt("подскажика пожалуйста сколько сейчас времечка", use_voice=False)
    print("\n" + "=" * 50 + "\n")

    # Тест обычного вопроса
    next_prompt("как дела", use_voice=False)
    print("\n" + "=" * 50 + "\n")
    """

    """
    next_prompt("сколько будет два плюс 2", use_voice=False)
    print("\n" + "=" * 50 + "\n")

    next_prompt("сколько будет два плюс четыре", use_voice=False)
    print("\n" + "=" * 50 + "\n")

    """
    """
    next_prompt("сколько сейчас времени", use_voice=False)
    print("\n" + "=" * 50 + "\n")

    #next_prompt("Сколько будет 15*3, потом результат раздели на 5", use_voice=False)
    print("\n" + "=" * 50 + "\n")

    next_prompt("Сколько будет четыреста пятьдесят семь умножить на пятнадцать, разделить на три, разделить на тридцать", use_voice=False)
    print("\n" + "=" * 50 + "\n")

    #next_prompt("Тело массой 5 кг тянут силой 18 Н по горизонтали, сила трения равна 3 Н — найди его ускорение.", use_voice=False)
    print("\n" + "=" * 50 + "\n")

    #next_prompt("привет как дела", use_voice=False)
    print("\n" + "=" * 50 + "\n")




    next_prompt("Привет меня зовут Максим", use_voice=False)
    print("\n" + "=" * 50 + "\n")

    next_prompt("забудь что меня зовут максим", use_voice=False)
    print("\n" + "=" * 50 + "\n")

    
    next_prompt("сколько сейчас времени", use_voice=False)

    print("\n" + "=" * 50 + "\n")
    next_prompt("подскажика пожалуйста сколько сейчас времечка", use_voice=False)

    print("\n" + "=" * 50 + "\n")

    next_prompt("Сколько будет 15*3, потом результат раздели на 5", use_voice=False)

    print("\n" + "=" * 50 + "\n")
"""
    print("\n" + "=" * 50 + "\n")
    # Работает на ЛЮБОМ языке из 50+
    next_prompt("Hello, how are you?", use_voice=False)
    print("\n" + "=" * 50 + "\n")
    next_prompt("Привет, как дела?", use_voice=False)
    print("\n" + "=" * 50 + "\n")
    next_prompt("Bonjour, comment ça va?", use_voice=False)
    print("\n" + "=" * 50 + "\n")
    next_prompt("你好，你好吗？", use_voice=False)
    print("\n" + "=" * 50 + "\n")

    #plugin_handlers = load_plugins()
    print("🧠 проверка завершена")
    #speak("Привет")



    test = False

    while test:
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
        """
        for handler in plugin_handlers:
            result = handler(user_input)
            if result:
                speak(result)
                handled = True
                break
        """
        if not handled:
            reply = query_llm_stream(user_input)
            speak(reply)


if __name__ == "__main__":
    main()