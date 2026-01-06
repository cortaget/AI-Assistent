import os
import requests
from collections import deque




# Папка для работы с файлами
WORKSPACE_DIR = r"E:\Ai_Trash\TestAI"
ALLOWED_EXTENSIONS = [".py", ".txt", ".json", ".md"]

# Настройки LLM
LLM_URL = "http://127.0.0.1:11434/api/generate"
LLM_MODEL = "gemma3:4b"

# Скользящая память для последних 20 инструкций + ответов
plugin_memory = deque(maxlen=20)  # хранит кортежи (instruction, file_content, llm_reply)я

def query_llm(user_instruction: str, file_content: str) -> str:
    """
    Отправляем содержимое файла + инструкцию в LLM и получаем ответ
    """
    # добавляем последние 20 инструкций в контекст
    context_text = ""
    for past_instruction, past_content, past_reply in plugin_memory:
        context_text += f"Инструкция: {past_instruction}\nСодержимое файла:\n{past_content}\nОтвет: {past_reply}\n\n"

    prompt = f"""
Ты локальный помощник. Учитывай предыдущий контекст:

{context_text}

Новая инструкция: "{user_instruction}"
Содержимое файла:

{file_content}

Проанализируй содержимое файла и дай ответ пользователю.
Не редактируй файл, только анализируй и отвечай.
"""
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(LLM_URL, json=payload)
        if response.status_code != 200:
            return f"Ошибка LLM: {response.status_code} - {response.text}"

        data = response.json()
        reply = data.get("response", "⚠ Нет ответа от модели")

        # сохраняем в память
        plugin_memory.append((user_instruction, file_content, reply))
        return reply

    except Exception as e:
        return f"Ошибка соединения с LLM: {str(e)}"


def handle(command: str) -> str | None:
    cmd = command.lower()

    # ----------------- Просмотр файла -----------------
    if any(kw in cmd for kw in ["открой файл", "покажи файл", "прочитай файл", "посмотри файл"]):
        try:
            words = cmd.split()
            filename = words[-1].strip()

            found_file = None
            for ext in ALLOWED_EXTENSIONS:
                candidate = os.path.join(WORKSPACE_DIR, filename + ext)
                if os.path.isfile(candidate):
                    found_file = candidate
                    break

            if not found_file:
                return f"Файл {filename} не найден среди {', '.join(ALLOWED_EXTENSIONS)}."

            with open(found_file, "r", encoding="utf-8") as f:
                content = f.read()

            short_name = os.path.basename(found_file)
            if len(content) > 500:
                preview = content[:500] + "\n...\n[файл слишком длинный, показана только часть]"
                return f"Содержимое файла {short_name}:\n{preview}"
            else:
                return f"Содержимое файла {short_name}:\n{content}"

        except Exception as e:
            return f"Ошибка при чтении файла: {str(e)}"

    # ----------------- Анализ файла -----------------
    if any(kw in cmd for kw in ["проанализируй файл", "посчитай в файле", "скажи что в файле"]):
        try:
            words = cmd.split()
            filename = words[-1].strip()

            found_file = None
            for ext in ALLOWED_EXTENSIONS:
                candidate = os.path.join(WORKSPACE_DIR, filename + ext)
                if os.path.isfile(candidate):
                    found_file = candidate
                    break

            if not found_file:
                return f"Файл {filename} не найден среди {', '.join(ALLOWED_EXTENSIONS)}."

            with open(found_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Отправляем в LLM для анализа с контекстом
            return query_llm(command, content)

        except Exception as e:
            return f"Ошибка при анализе файла: {str(e)}"

    return None
