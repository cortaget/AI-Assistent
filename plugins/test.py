import requests
import json

LLM_URL = "http://127.0.0.1:11434/api/generate"

LLM_MODEL = "gemma3:4b"

def query_llm(prompt: str):
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False   # отключаем стрим, чтобы вернуть весь ответ разом
    }

    response = requests.post(LLM_URL, json=payload)

    if response.status_code != 200:
        print("Ошибка:", response.status_code, response.text)
        return

    data = response.json()
    print("💬 Ответ:", data.get("response", ""))


if __name__ == "__main__":
    query_llm("Привет, как дела?")
