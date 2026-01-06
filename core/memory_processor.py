# core/memory_processor.py
import threading
import re
from typing import Optional
from memory.memory_manager import MemoryManager
from core.llm_client import LLMClient
from config import Config


class MemoryProcessor:
    """Обработчик автоматического управления памятью"""

    def __init__(self, memory_manager: MemoryManager, llm_client: LLMClient, config: Config = None):
        """
        Инициализация процессора памяти

        Args:
            memory_manager: Менеджер памяти
            llm_client: LLM клиент
            config: Объект конфигурации
        """
        self.memory = memory_manager
        self.llm = llm_client
        self.config = config or Config()

        # Защита от дублирования обработки
        self._processing = {}
        self._lock = threading.Lock()

    def auto_manage(self, user_input: str, assistant_reply: str):
        """
        Автоматическое управление памятью

        Args:
            user_input: Ввод пользователя
            assistant_reply: Ответ ассистента
        """
        # Защита от дублирования
        request_id = f"{user_input[:50]}_{assistant_reply[:50]}"

        with self._lock:
            if request_id in self._processing:
                return
            self._processing[request_id] = True

        try:
            self._process_memory(user_input, assistant_reply)
        finally:
            with self._lock:
                self._processing.pop(request_id, None)

    def _process_memory(self, user_input: str, assistant_reply: str):
        """
        Внутренняя обработка памяти

        Args:
            user_input: Ввод пользователя
            assistant_reply: Ответ ассистента
        """
        # Определение намерения через LLM
        intent = self._detect_intent(user_input, assistant_reply)

        if "A" in intent.upper():
            self._handle_deletion(user_input)
        elif "B" in intent.upper():
            self._handle_saving(user_input, assistant_reply)

    def _detect_intent(self, user_input: str, assistant_reply: str) -> str:
        """
        Определение намерения пользователя

        Args:
            user_input: Ввод пользователя
            assistant_reply: Ответ ассистента

        Returns:
            Код намерения (A/B/C)
        """
        intent_prompt = f"""Диалог:
Пользователь: {user_input}
Ассистент: {assistant_reply}

Что хочет пользователь?
A) УДАЛИТЬ информацию из памяти (забудь, удали, сотри)
B) СОХРАНИТЬ информацию (даёт факты о себе)
C) ОБЫЧНЫЙ ВОПРОС

Ответь ТОЛЬКО: A, B или C"""

        return self.llm.quick_call(intent_prompt, max_tokens=self.config.MEMORY_INTENT_DETECTION_TOKENS)

    def _handle_deletion(self, user_input: str):
        """
        Обработка удаления из памяти

        Args:
            user_input: Ввод пользователя
        """
        print("🗑️ Обработка удаления через векторный поиск...")

        # Извлечение ключевых слов для поиска
        query_prompt = f"""Из фразы '{user_input}' извлеки ключевые слова для поиска в базе памяти.
Включи контекст (например: "забудь что меня зовут Максим" → "Имя пользователя Максим" или "зовут Максим").
Ответь 2-5 словами:"""

        search_query = self.llm.quick_call(query_prompt, max_tokens=self.config.MEMORY_SEARCH_QUERY_TOKENS).strip('"\'')

        if len(search_query) < 2:
            print("⚠️ Не удалось понять что удалять")
            return

        print(f"🔍 Векторный поиск: '{search_query}'")
        results = self.memory.search_memory(search_query, top_k=3)

        if results:
            print(f"📋 Найдено через векторный поиск:")
            for i, r in enumerate(results[:3], 1):
                print(f"  {i}. [{r['relevance']:.2f}] {r['content']}")

            if results[0]['relevance'] > self.config.DELETE_THRESHOLD:
                self.memory.delete_memory(results[0]['id'])
                print(f"🗑️ Удалено: {results[0]['content']}")
            else:
                print(f"⚠️ Релевантность слишком низкая: {results[0]['relevance']:.2f}")
        else:
            print("❌ Ничего не найдено")

    def _handle_saving(self, user_input: str, assistant_reply: str):
        """
        Обработка сохранения в память

        Args:
            user_input: Ввод пользователя
            assistant_reply: Ответ ассистента
        """
        print("💾 Обработка сохранения...")

        # Извлечение факта
        extract_prompt = f"""Диалог:
Пользователь: {user_input}
Ассистент: {assistant_reply}

Извлеки ОДИН факт о пользователе. Если это вопрос - ответь "НЕТ".
Примеры:
- "Привет, я Максим" → "Имя пользователя: Максим"
- "Я люблю кошек" → "Пользователь любит кошек"
- "Как дела?" → "НЕТ"

Ответь ТОЛЬКО фактом или "НЕТ"."""

        fact = self.llm.quick_call(extract_prompt, max_tokens=self.config.MEMORY_FACT_EXTRACTION_TOKENS).strip('"\'')

        if not fact or "НЕТ" in fact.upper() or len(fact) < 7:
            print("⏭️ Нет фактов для сохранения")
            return

        print(f"📝 Извлечённый факт: {fact}")

        # Проверка на дубликаты
        similar = self.memory.search_memory(fact, top_k=1)

        if similar and len(similar) > 0:
            relevance = similar[0]['relevance']
            print(f"🔍 Похожая запись: {relevance:.2f}")

            if relevance > self.config.DUPLICATE_THRESHOLD:
                print(f"⚠️ Точный дубликат, пропускаем")
                return
            elif relevance > self.config.SIMILAR_THRESHOLD:
                print(f"⚠️ Очень похожая запись, пропускаем")
                return
            elif relevance > self.config.UPDATE_THRESHOLD:
                self.memory.update_memory(similar[0]['id'], fact)
                print(f"🔄 Обновлено: {fact}")
                return

        # Добавление нового факта
        self.memory.add_memory(fact, memory_type="user_info")
        print(f"💾 Сохранено: {fact}")
