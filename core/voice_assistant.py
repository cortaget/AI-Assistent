# core/voice_assistant.py
import re
from typing import List, Dict, Optional
from config import Config
from core.speech_manager import SpeechManager
from core.llm_client import LLMClient
from core.memory_processor import MemoryProcessor
from memory.memory_manager import MemoryManager
from tools.tool_manager import ToolManager
from core.plugin_loader import load_plugins
from core.language_utils import LanguageDetector

class VoiceAssistant:
    """Главный класс голосового ассистента"""

    def __init__(self, config: Config = None):
        """Инициализация ассистента"""
        self.config = config or Config()

        # Инициализация компонентов
        self.speech_manager = SpeechManager(self.config)
        self.llm_client = LLMClient(self.config)
        self.memory_manager = MemoryManager(
            persist_dir=self.config.MEMORY_DB_PATH,
            collection_name=self.config.MEMORY_COLLECTION_NAME
        )
        self.memory_manager.debug_mode = self.config.MEMORY_DEBUG_MODE

        self.memory_processor = MemoryProcessor(
            self.memory_manager,
            self.llm_client,
            self.config
        )

        self.tool_manager = ToolManager(
            tools_dir=self.config.TOOLS_DIR,
            llm_url=self.config.LLM_URL,
            llm_model=self.config.LLM_MODEL
        )

        if self.config.ENABLE_MULTILINGUAL:
            self.lang_detector = LanguageDetector(default_lang=self.config.DEFAULT_LANGUAGE)
        else:
            self.lang_detector = None

        # Загрузка плагинов
        self.plugin_handlers = load_plugins()

    def _query_without_tools(self, user_input: str) -> str:
        """Обработка запроса без инструментов"""
        relevant_memories = self.memory_manager.search_memory(
            user_input,
            top_k=self.config.MEMORY_TOP_K
        )

        # Определяем язык через библиотеку
        detected_lang = "ru"
        if self.lang_detector:
            detected_lang = self.lang_detector.detect(user_input)

            # Fallback для кириллицы: mk, bg, sr → ru
            if detected_lang in ['mk', 'bg', 'sr', 'uk', 'be']:
                detected_lang = 'ru'

            print(f"🌍 Язык: {detected_lang}")

        # Простой системный промпт
        system_prompt = """Ты - голосовой ИИ-ассистент.
    ВАЖНО понимать:
    - ТЫ - это ассистент (искусственный интеллект)
    - ПОЛЬЗОВАТЕЛЬ - это человек, который с тобой общается
    """

        # Контекст из памяти
        memory_context = ""
        if relevant_memories:
            filtered = [m for m in relevant_memories if m['relevance'] > self.config.MEMORY_RELEVANCE_THRESHOLD]
            if filtered:
                memory_context = "[Информация из долговременной памяти]:\n"
                for mem in filtered:
                    memory_context += f"- {mem['content']}\n"
                memory_context += "\n"

        # Карта инструкций по языкам
        lang_instructions = {
            'ru': '(ВАЖНО: Отвечай ТОЛЬКО на русском языке)',
            'en': '(IMPORTANT: Reply ONLY in English)',
            'de': '(WICHTIG: Antworte NUR auf Deutsch)',
            'fr': '(IMPORTANT: Réponds UNIQUEMENT en français)',
            'es': '(IMPORTANTE: Responde SOLO en español)',
            'it': '(IMPORTANTE: Rispondi SOLO in italiano)',
            'pt': '(IMPORTANTE: Responda APENAS em português)',
            'cs': '(DŮLEŽITÉ: Odpovídej POUZE česky)',  # ← ЧЕШСКИЙ ДОБАВЛЕН
            'zh': '(重要: 仅用中文回答)',
            'ja': '(重要: 日本語のみで回答)',
            'ko': '(중요: 한국어로만 답변)'
        }

        lang_hint = lang_instructions.get(detected_lang, lang_instructions['ru'])

        # Собираем промпт
        full_prompt = (
                system_prompt +
                memory_context +
                f"Пользователь: {user_input}\n" +
                f"{lang_hint}\n" +
                "Ассистент:"
        )

        return self.llm_client.stream_call(full_prompt)

    def process_query(self, user_input: str) -> str:
        """
        Обработка запроса пользователя

        Args:
            user_input: Текст запроса

        Returns:
            Ответ ассистента
        """
        print(f"\n👤 {user_input}")

        """
        # Проверка плагинов
        for handler in self.plugin_handlers:
            result = handler(user_input)
            if result:
                return result
        """
        # Основная обработка через LLM
        reply = self._query_llm_stream(user_input)

        # Автоматическое управление памятью
        self.memory_processor.auto_manage(user_input, reply)

        return reply

    def _query_llm_stream(self, user_input: str) -> str:
        """
        Запрос с автоматическим определением необходимости инструментов

        Args:
            user_input: Ввод пользователя

        Returns:
            Ответ ассистента
        """
        # Шаг 1: LLM решает, нужны ли инструменты
        print("🔍 Анализ запроса...")
        decision = self.tool_manager.decide_tool_usage(user_input)

        print(f"💭 Решение: {'Использовать инструменты' if decision['needs_tools'] else 'Обычный диалог'}")
        print(f"  Причина: {decision['reasoning']}")

        if "ошибка" in decision['reasoning'].lower():
            print("⚠️ Роутер недоступен, обрабатываю память в фоне...")

        # Шаг 2: Если нужны инструменты
        if decision['needs_tools'] and decision['suggested_tools']:
            relevant_tools = []
            for tool_name in decision['suggested_tools']:
                if tool_name in self.tool_manager.tools:
                    relevant_tools.append({
                        "name": tool_name,
                        "tool": self.tool_manager.tools[tool_name],
                        "score": 1.0
                    })

            if relevant_tools:
                print(f"🔧 Выбрано инструментов: {[t['name'] for t in relevant_tools]}\n")
                return self._query_with_tools(user_input, relevant_tools)

        # Шаг 3: Обычный диалог без инструментов
        print("💬 Обработка без инструментов...\n")
        return self._query_without_tools(user_input)


    def _query_with_tools(self, user_input: str, relevant_tools: List[Dict]) -> str:
        """
        Многошаговое выполнение с инструментами (ReAct pattern)

        Args:
            user_input: Ввод пользователя
            relevant_tools: Список релевантных инструментов

        Returns:
            Финальный ответ
        """
        # Формируем описание инструментов
        tools_desc = self.tool_manager.get_tools_description(relevant_tools)

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
        for iteration in range(self.config.MAX_TOOL_ITERATIONS):
            print(f"\n🔄 Итерация {iteration + 1}/{self.config.MAX_TOOL_ITERATIONS}")

            llm_response = self.llm_client.non_stream_call(
                system_prompt,
                timeout=self.config.TOOL_TIMEOUT
            )
            print(f"🤖 LLM: {llm_response}")

            # Проверка финального ответа
            if "ANSWER[" in llm_response:
                answer = llm_response.split("ANSWER[")[1].split("]")[0]
                print(f"✅ Финальный ответ: {answer}")
                return answer

            # Извлечение вызовов инструментов
            tool_calls = self._extract_tool_calls(llm_response)

            if not tool_calls:
                # Проверка на фейковые результаты
                if "[Получен результат:" in llm_response or "[Результат:" in llm_response:
                    print("⚠️ LLM пытается выдумать результат!")
                    return "Ошибка: попытка генерации фейкового результата"
                return llm_response

            # Выполнение инструментов
            results = []
            for tool_call in tool_calls:
                tool_name = tool_call['tool']
                params = tool_call['params']

                if tool_name not in self.tool_manager.tools:
                    result = f"❌ Инструмент '{tool_name}' не найден"
                else:
                    result = self.tool_manager.execute_tool(
                        tool_name,
                        params=params,
                        user_input=user_input
                    )

                result_str = f"[{tool_name}]: {result}"
                results.append(result_str)
                print(f"  🔧 {tool_name}({params}) → {result}")

            # Обновление промпта
            system_prompt += f"\nAssistant: {llm_response}\n"
            system_prompt += f"System: {chr(10).join(results)}\n"
            system_prompt += "Assistant: "

        return "Превышено максимальное количество шагов"

    def _extract_tool_calls(self, text: str) -> List[Dict]:
        """
        Извлечение вызовов инструментов из текста LLM

        Args:
            text: Текст ответа LLM

        Returns:
            Список вызовов инструментов
        """
        pattern = r'TOOL\[(\w+)\]\(([^)]*)\)'
        matches = re.findall(pattern, text)

        calls = []
        for tool_name, params_str in matches:
            if not params_str or params_str.strip() == "":
                params = {}
            else:
                params = {"expression": params_str.strip()}

            calls.append({"tool": tool_name, "params": params})

        return calls

    def next_prompt(self, user_input: str, use_voice: bool = False) -> str:
        """
        Простая функция для тестирования через код

        Args:
            user_input: Текст запроса
            use_voice: Озвучивать или нет

        Returns:
            Ответ ассистента
        """
        reply = self.process_query(user_input)

        if use_voice:
            self.speech_manager.speak(reply)

        return reply

    def run_voice_loop(self):
        """Запуск основного голосового цикла"""
        while True:
            user_input = self.speech_manager.listen_command()

            if not user_input:
                continue

            # Проверка на выход
            if any(word in user_input for word in ["выход", "стоп", "выключись"]):
                self.speech_manager.speak("Пока")
                self.speech_manager.stop()
                break

            # Обработка запроса
            reply = self.process_query(user_input)
            self.speech_manager.speak(reply)
