# core/llm_client.py
import json
import requests
from typing import Dict, Any, Optional
from config import Config


class LLMClient:
    """Клиент для взаимодействия с LLM (Ollama)"""

    def __init__(self, config: Config = None):
        """
        Инициализация LLM клиента

        Args:
            config: Объект конфигурации
        """
        self.config = config or Config()
        self.url = self.config.LLM_URL
        self.model = self.config.LLM_MODEL

    def quick_call(self, prompt: str, max_tokens: int = 50, temperature: float = None) -> str:
        """
        Быстрый вызов LLM без стриминга

        Args:
            prompt: Промпт для LLM
            max_tokens: Максимальное количество токенов
            temperature: Температура генерации

        Returns:
            Ответ LLM
        """
        if temperature is None:
            temperature = self.config.LLM_QUICK_CALL_TEMPERATURE

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_k": self.config.LLM_QUICK_CALL_TOP_K,
                "top_p": self.config.LLM_QUICK_CALL_TOP_P
            }
        }

        try:
            response = requests.post(self.url, json=payload, timeout=100)
            if response.status_code == 200:
                return response.json().get("response", "").strip()
        except:
            pass

        return ""

    def stream_call(self, prompt: str, temperature: float = None) -> str:
        """
        Вызов LLM со стримингом

        Args:
            prompt: Промпт для LLM
            temperature: Температура генерации

        Returns:
            Полный ответ LLM
        """
        if temperature is None:
            temperature = self.config.LLM_TEMPERATURE

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature
            }
        }

        try:
            response = requests.post(self.url, json=payload, stream=True)
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
            return reply

        except Exception as e:
            return f"Ошибка: {str(e)}"

    def non_stream_call(self, prompt: str, temperature: float = None, timeout: int = 100) -> str:
        """
        Вызов LLM без стриминга

        Args:
            prompt: Промпт для LLM
            temperature: Температура генерации
            timeout: Таймаут запроса

        Returns:
            Ответ LLM
        """
        if temperature is None:
            temperature = self.config.LLM_TEMPERATURE

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature}
        }

        try:
            response = requests.post(self.url, json=payload, timeout=timeout)
            if response.status_code == 200:
                return response.json().get("response", "").strip()
            return ""
        except Exception as e:
            return f"Ошибка: {str(e)}"
