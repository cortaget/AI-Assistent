# tools/tool_manager.py
import os
import importlib.util
from typing import Dict, List, Optional, Callable
from sentence_transformers import SentenceTransformer
import numpy as np
import requests

class Tool:
    """Класс инструмента с метаданными"""
    def __init__(self, name: str, description: str, usage_context: str,
                 function: Callable, parameters: Dict = None,
                 keywords: List[str] = None,
                 examples: List[str] = None,
                 parameter_extractor: Callable = None,
                 categories: List[str] = None):  # ← НОВАЯ СТРОКА
        self.name = name
        self.description = description
        self.usage_context = usage_context
        self.function = function
        self.parameters = parameters or {}
        self.keywords = keywords or []
        self.examples = examples or []
        self.parameter_extractor = parameter_extractor
        self.categories = categories or ["general"]  # ← НОВАЯ СТРОКА

    def execute(self, **kwargs):
        """Выполнить инструмент с параметрами"""
        return self.function(**kwargs)

    def extract_parameters(self, user_input: str) -> Dict:
        """Извлечь параметры из запроса пользователя"""
        if self.parameter_extractor:
            return self.parameter_extractor(user_input)
        return {}

class ToolManager:
    """Менеджер инструментов с hybrid роутингом"""

    def __init__(self, tools_dir: str = "tools", llm_url: str = None, llm_model: str = None):
        self.tools_dir = tools_dir
        self.tools: Dict[str, Tool] = {}
        self.llm_url = llm_url
        self.llm_model = llm_model

        # Семантический роутер
        print("🧠 Загрузка модели для семантического роутинга...")
        self.embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.tool_embeddings = {}

        self.load_tools()
        self._prepare_semantic_routing()

    def load_tools(self):
        """Автоматическая загрузка всех инструментов из папки"""
        if not os.path.exists(self.tools_dir):
            os.makedirs(self.tools_dir)
            print(f"📁 Создана папка для инструментов: {self.tools_dir}")
            return

        for filename in os.listdir(self.tools_dir):
            if filename.endswith('.py') and filename != '__init__.py' and filename != 'tool_manager.py':
                tool_path = os.path.join(self.tools_dir, filename)
                self._load_tool_from_file(tool_path)

        print(f"🔧 Загружено инструментов: {len(self.tools)}")
        for tool_name, tool in self.tools.items():
            keywords_info = f" | Ключевые слова: {tool.keywords}" if tool.keywords else ""
            print(f"  ✅ {tool_name}{keywords_info}")

    def _load_tool_from_file(self, filepath: str):
        """Загрузить инструмент из файла"""
        try:
            module_name = os.path.basename(filepath)[:-3]
            spec = importlib.util.spec_from_file_location(module_name, filepath)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if hasattr(module, 'register_tool'):
                tool = module.register_tool()
                if isinstance(tool, Tool):
                    self.tools[tool.name] = tool
                else:
                    print(f"  ⚠️ {filepath}: register_tool() должна возвращать Tool")
            else:
                print(f"  ⚠️ {filepath}: нет функции register_tool()")

        except Exception as e:
            print(f"  ❌ Ошибка загрузки {filepath}: {e}")

    def _prepare_semantic_routing(self):
        """Подготовить эмбеддинги для семантического роутинга"""
        for tool_name, tool in self.tools.items():
            context_text = f"{tool.description}. {tool.usage_context}"
            if tool.keywords:
                context_text += f". Ключевые слова: {', '.join(tool.keywords)}"

            embedding = self.embedder.encode(context_text)
            self.tool_embeddings[tool_name] = embedding

        if self.tool_embeddings:
            print(f"🧠 Подготовлено {len(self.tool_embeddings)} семантических роутов")

    def _keyword_match(self, user_input: str) -> Optional[str]:
        """
        Быстрая проверка по ключевым словам + семантическая валидация
        (БЕЗ стоп-слов - только позитивная валидация)
        """
        user_lower = user_input.lower()

        # Собираем кандидатов по ключевым словам
        candidates = []

        for tool_name, tool in self.tools.items():
            if not tool.keywords:
                continue

            matched_keywords = [kw for kw in tool.keywords if kw.lower() in user_lower]

            if matched_keywords:
                candidates.append((tool_name, matched_keywords))

        if not candidates:
            return None

        # === ВАЛИДАЦИЯ: Сравниваем с ПРИМЕРАМИ использования ===
        best_candidate = None
        best_score = 0.0

        query_embedding = self.embedder.encode(user_input)

        for tool_name, matched_kw in candidates:
            if tool_name not in self.tool_embeddings:
                continue

            tool_embedding = self.tool_embeddings[tool_name]

            # Косинусное расстояние с усреднённым вектором примеров
            score = np.dot(query_embedding, tool_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(tool_embedding)
            )

            if score > best_score:
                best_score = score
                best_candidate = (tool_name, matched_kw)

        # Порог: 0.60 (выше чем раньше, т.к. сравниваем с примерами)
        if best_score > 0.75:
            tool_name, matched_kw = best_candidate
            print(
                f"⚡ Валидировано: {tool_name} | Ключевые слова: {matched_kw} | Схожесть с примерами: {best_score:.2f}")
            return tool_name

        if candidates:
            print(
                f"⚠️ Найдены ключевые слова {candidates}, но не похоже на примеры использования (схожесть: {best_score:.2f})")

        return None

    def _semantic_match(self, user_input: str, threshold: float = 0.75) -> Optional[str]:
        """Семантический поиск через векторное сходство"""
        if not self.tool_embeddings:
            return None

        query_embedding = self.embedder.encode(user_input)

        best_tool = None
        best_score = 0.0

        for tool_name, tool_embedding in self.tool_embeddings.items():
            score = np.dot(query_embedding, tool_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(tool_embedding)
            )

            if score > best_score:
                best_score = score
                best_tool = tool_name

        if best_score > threshold:
            print(f"🧠 Семантическое совпадение: {best_tool} (схожесть: {best_score:.2f})")
            return best_tool

        print(f"❌ Нет подходящего инструмента (макс схожесть: {best_score:.2f})")
        return None

    def route_to_tool(self, user_input: str) -> Optional[Dict]:
        """3-этапный hybrid роутинг"""
        if not self.tools:
            return None

        # ЭТАП 1: Ключевые слова с валидацией
        tool_name = self._keyword_match(user_input)

        # ЭТАП 2: Семантический поиск
        if not tool_name:
            tool_name = self._semantic_match(user_input, threshold=0.75)

        if tool_name:
            return {"tool": tool_name, "params": {}}

        return None

    def execute_tool(self, tool_name: str, params: Dict = None, user_input: str = None) -> str:
        """Выполнить инструмент с автоматическим извлечением параметров"""
        if tool_name not in self.tools:
            return f"❌ Инструмент '{tool_name}' не найден"

        tool = self.tools[tool_name]
        params = params or {}

        # === УНИВЕРСАЛЬНОЕ ИЗВЛЕЧЕНИЕ ПАРАМЕТРОВ ===
        if user_input and tool.parameter_extractor:
            try:
                extracted = tool.extract_parameters(user_input)
                if extracted:
                    params.update(extracted)
                    print(f"  📊 Извлечены параметры: {extracted}")
            except Exception as e:
                print(f"  ⚠️ Ошибка извлечения параметров: {e}")

        try:
            result = tool.execute(**params)
            print(f"✅ Инструмент {tool_name} выполнен: {result}")
            return str(result)
        except Exception as e:
            error_msg = f"❌ Ошибка выполнения {tool_name}: {e}"
            print(error_msg)
            return error_msg

    def process_request(self, user_input: str) -> Optional[str]:
        """Обработать запрос: определить инструмент, извлечь параметры и выполнить"""
        tool_decision = self.route_to_tool(user_input)

        if tool_decision:
            tool_name = tool_decision.get("tool")
            params = tool_decision.get("params", {})
            # ВАЖНО: передаём user_input
            return self.execute_tool(tool_name, params, user_input=user_input)

        return None

    def _prepare_semantic_routing(self):
        """Подготовить эмбеддинги для семантического роутинга"""
        for tool_name, tool in self.tools.items():
            # === НОВАЯ ЛОГИКА: используем примеры вместо описания ===
            if tool.examples:
                # Создаём эмбеддинг из УСРЕДНЁННЫХ примеров
                # Это даёт более точное представление о том, КАК спрашивают
                example_embeddings = [self.embedder.encode(ex) for ex in tool.examples]

                # Усредняем все примеры в один вектор
                tool_embedding = np.mean(example_embeddings, axis=0)
            else:
                # Fallback: если примеров нет - используем описание
                context_text = f"{tool.description}. {tool.usage_context}"
                tool_embedding = self.embedder.encode(context_text)

            self.tool_embeddings[tool_name] = tool_embedding

        if self.tool_embeddings:
            print(f"🧠 Подготовлено {len(self.tool_embeddings)} семантических роутов")


    def get_tools_description(self, tool_list: List[Dict]) -> str:
        """Формирует описание инструментов для промпта LLM"""
        if not tool_list:
            return ""

        desc = "Доступные инструменты:\n"
        for item in tool_list:
            tool = item['tool']
            desc += f"- {tool.name}: {tool.description}\n"
            if tool.parameters:
                params = ", ".join(tool.parameters.keys())
                desc += f"  Параметры: {params}\n"

        return desc

    def filter_by_categories(self, user_input: str, tools_list: List[Dict]) -> List[Dict]:
        """
        Убирает инструменты, которые явно не подходят по контексту

        Например: если запрос про расчёты - убираем get_time
        """
        lower_input = user_input.lower()

        # Правила исключения: если в запросе есть эти слова - исключаем категории
        exclude_rules = {
            "time": ["сколько будет", "посчитай", "вычисли", "плюс", "минус", "*", "/"],
            "datetime": ["сколько будет", "посчитай", "вычисли"],
            "calculation": ["сейчас времени", "который час", "какая дата", "сколько времени"],
            "math": ["сейчас времени", "который час", "какая дата"]
        }

        filtered = []
        for tool_item in tools_list:
            tool = tool_item['tool']
            exclude = False

            # Проверяем каждую категорию инструмента
            for category in tool.categories:
                if category in exclude_rules:
                    # Если хоть один паттерн совпал - исключаем
                    if any(pattern in lower_input for pattern in exclude_rules[category]):
                        exclude = True
                        print(f"  ❌ Исключён {tool.name} (категория '{category}' не подходит)")
                        break

            if not exclude:
                filtered.append(tool_item)

        return filtered

    def decide_tool_usage(self, user_input: str) -> Dict:
        """
        LLM решает, нужны ли инструменты для этого запроса

        Возвращает:
        {
            "needs_tools": True/False,
            "reasoning": "Почему нужны/не нужны инструменты",
            "suggested_tools": ["calculator", "get_time"] или []
        }
        """
        # Формируем список всех доступных инструментов
        tools_list = []
        for tool_name, tool in self.tools.items():
            tools_list.append(f"- {tool_name}: {tool.description}")

        tools_description = "\n".join(tools_list)

        decision_prompt = f"""Ты - анализатор запросов. Твоя задача: определить, нужны ли ИНСТРУМЕНТЫ для ответа на запрос пользователя.

    ДОСТУПНЫЕ ИНСТРУМЕНТЫ:
    {tools_description}

    ПРАВИЛА:
    1. Если запрос требует ВЫЧИСЛЕНИЙ, РАСЧЁТОВ, ТЕКУЩЕГО ВРЕМЕНИ/ДАТЫ - нужны инструменты
    2. Если это ОБЫЧНАЯ БЕСЕДА (приветствие, вопросы о самочувствии, благодарность) - инструменты НЕ нужны
    3. Если спрашивают общую информацию (что такое X, объясни Y) - инструменты НЕ нужны

    ФОРМАТ ОТВЕТА (СТРОГО):
    DECISION: YES или NO
    TOOLS: список инструментов через запятую (или NONE)
    REASON: краткое объяснение

    ПРИМЕРЫ:

    Запрос: "Сколько будет 15*3?"
    DECISION: YES
    TOOLS: calculator
    REASON: Требуется математическое вычисление

    Запрос: "Привет, как дела?"
    DECISION: NO
    TOOLS: NONE
    REASON: Обычное приветствие, не требует инструментов

    Запрос: "Который час?"
    DECISION: YES
    TOOLS: get_time
    REASON: Нужно узнать текущее время

    Запрос: "Что такое квантовая физика?"
    DECISION: NO
    TOOLS: NONE
    REASON: Общий вопрос, не требует специальных инструментов

    ТЕКУЩИЙ ЗАПРОС: {user_input}

    Твой ответ (строго по формату выше):"""

        try:
            payload = {
                "model": self.llm_model,
                "prompt": decision_prompt,
                "stream": False,
                "options": {"temperature": 0.1}  # Низкая температура для точности
            }

            response = requests.post(self.llm_url, json=payload, timeout=100)
            if response.status_code != 200:
                return {"needs_tools": False, "reasoning": "Ошибка LLM", "suggested_tools": []}

            llm_response = response.json().get("response", "").strip()

            # Парсим ответ
            import re
            decision_match = re.search(r'DECISION:\s*(YES|NO)', llm_response, re.IGNORECASE)
            tools_match = re.search(r'TOOLS:\s*([^\n]+)', llm_response)
            reason_match = re.search(r'REASON:\s*([^\n]+)', llm_response)

            needs_tools = decision_match.group(1).upper() == "YES" if decision_match else False

            suggested_tools = []
            if tools_match and tools_match.group(1).strip().upper() != "NONE":
                tools_str = tools_match.group(1).strip()
                suggested_tools = [t.strip() for t in tools_str.split(',') if t.strip() in self.tools]

            reasoning = reason_match.group(1).strip() if reason_match else "Не указано"

            return {
                "needs_tools": needs_tools,
                "reasoning": reasoning,
                "suggested_tools": suggested_tools
            }

        except Exception as e:
            print(f"⚠️ Ошибка роутера: {e}")
            return {"needs_tools": False, "reasoning": f"Ошибка: {e}", "suggested_tools": []}
