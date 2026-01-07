# memory_manager.py
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import uuid
from datetime import datetime
from typing import List, Dict, Optional
import requests
from core.language_utils import LanguageDetector
import json

class MemoryManager:
    def __init__(self, persist_dir="./memory_db", collection_name="assistant_memory"):
        """Инициализация системы памяти"""
        # Векторная модель для эмбеддингов (384 измерения, быстрая)
        self.embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        try:
            self.lang_detector = LanguageDetector()
        except:
            self.lang_detector = None

        # Обновите метод add_memory (добавьте только 2 строки):
        def add_memory(self, content: str, memory_type: str = "user_info",
                       metadata: Optional[Dict] = None) -> str:
            memory_id = str(uuid.uuid4())
            embedding = self.embedder.encode(content).tolist()

            # НОВОЕ: определяем язык
            detected_lang = self.lang_detector.detect(content) if self.lang_detector else 'unknown'

            mem_metadata = {
                "type": memory_type,
                "language": detected_lang,  # НОВОЕ
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }

            if metadata:
                mem_metadata.update(metadata)

            self.collection.add(
                ids=[memory_id],
                documents=[content],
                embeddings=[embedding],
                metadatas=[mem_metadata]
            )

            return memory_id

        # Обновите update_memory (добавьте 1 строку):
        def update_memory(self, memory_id: str, new_content: str, new_metadata: Optional[Dict] = None):
            old = self.collection.get(ids=[memory_id])
            if not old['ids']:
                print(f"❌ Память {memory_id} не найдена")
                return

            metadata = old['metadatas'][0]
            metadata['updated_at'] = datetime.now().isoformat()
            metadata['language'] = self.lang_detector.detect(new_content) if self.lang_detector else 'unknown'  # НОВОЕ

            if new_metadata:
                metadata.update(new_metadata)

            embedding = self.embedder.encode(new_content).tolist()
            self.collection.update(
                ids=[memory_id],
                documents=[new_content],
                embeddings=[embedding],
                metadatas=[metadata]
            )
            print(f"✅ Память обновлена: {memory_id[:8]}...")


        # ChromaDB клиент с постоянным хранением
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )

        # Коллекция для хранения памяти
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # косинусное расстояние для поиска
        )

    def add_memory(self, content: str, memory_type: str = "user_info",
                   metadata: Optional[Dict] = None) -> str:
        """Добавить новую запись в память"""
        memory_id = str(uuid.uuid4())

        embedding = self.embedder.encode(content).tolist()

        mem_metadata = {
            "type": memory_type,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        if metadata:
            mem_metadata.update(metadata)

        self.collection.add(
            ids=[memory_id],
            documents=[content],
            embeddings=[embedding],
            metadatas=[mem_metadata]
        )

        # Убрали print
        return memory_id

    def search_memory(self, query: str, top_k: int = 10,
                      memory_type: Optional[str] = None) -> List[Dict]:
        """
        Поиск релевантных воспоминаний

        Args:
            query: Поисковый запрос
            top_k: Количество результатов
            memory_type: Фильтр по типу памяти

        Returns:
            Список найденных воспоминаний
        """
        # Создаём эмбеддинг запроса
        query_embedding = self.embedder.encode(query).tolist()

        # Формируем фильтр
        where_filter = {"type": memory_type} if memory_type else None

        # Поиск
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        # Форматируем результаты
        memories = []
        if results['ids']:
            for i, doc_id in enumerate(results['ids'][0]):
                memories.append({
                    "id": doc_id,
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "relevance": 1 - results['distances'][0][i]  # конвертируем расстояние в релевантность
                })

        return memories

    def update_memory(self, memory_id: str, new_content: str,
                      new_metadata: Optional[Dict] = None):
        """
        Обновить существующую запись

        Args:
            memory_id: ID записи для обновления
            new_content: Новый текст
            new_metadata: Новые метаданные
        """
        # Получаем старую запись
        old = self.collection.get(ids=[memory_id])
        if not old['ids']:
            print(f"❌ Память {memory_id} не найдена")
            return

        # Обновляем метаданные
        metadata = old['metadatas'][0]
        metadata['updated_at'] = datetime.now().isoformat()
        if new_metadata:
            metadata.update(new_metadata)

        # Создаём новый эмбеддинг
        embedding = self.embedder.encode(new_content).tolist()

        # Обновляем
        self.collection.update(
            ids=[memory_id],
            documents=[new_content],
            embeddings=[embedding],
            metadatas=[metadata]
        )

        print(f"✅ Память обновлена: {memory_id[:8]}...")

    def delete_memory(self, memory_id: str):
        """Удалить запись из памяти"""
        self.collection.delete(ids=[memory_id])
        print(f"🗑️ Память удалена: {memory_id[:8]}...")

    def list_all_memories(self, memory_type: Optional[str] = None) -> List[Dict]:
        """
        Получить все записи в памяти

        Args:
            memory_type: Фильтр по типу

        Returns:
            Список всех воспоминаний
        """
        where_filter = {"type": memory_type} if memory_type else None

        results = self.collection.get(
            where=where_filter,
            include=["documents", "metadatas"]
        )

        memories = []
        if results['ids']:
            for i, doc_id in enumerate(results['ids']):
                memories.append({
                    "id": doc_id,
                    "content": results['documents'][i],
                    "metadata": results['metadatas'][i]
                })

        return memories

    def clear_all_memories(self):
        """Очистить всю память (осторожно!)"""
        # Удаляем и создаём коллекцию заново
        self.client.delete_collection(name=self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )
        print("🗑️ Вся память очищена")

    def extract_important_info(self, conversation_text: str) -> Optional[str]:
        """
        Извлечь важную информацию из диалога для запоминания
        (можно улучшить с помощью LLM)

        Args:
            conversation_text: Текст диалога

        Returns:
            Важная информация или None
        """
        # Простые ключевые слова для обнаружения важной информации
        important_keywords = [
            "меня зовут", "я работаю", "я люблю", "мой любимый",
            "я предпочитаю", "запомни", "важно", "всегда делай",
            "никогда не", "мне нравится", "я не люблю"
        ]

        text_lower = conversation_text.lower()
        for keyword in important_keywords:
            if keyword in text_lower:
                return conversation_text

        return None

    # memory_manager.py - добавь этот метод в класс MemoryManager



    def extract_with_llm(self, user_message: str, assistant_response: str,
                         llm_url: str, llm_model: str) -> List[str]:
        """
        Извлечение важной информации с помощью LLM

        Args:
            user_message: Сообщение пользователя
            assistant_response: Ответ ассистента
            llm_url: URL вашей Ollama
            llm_model: Модель (gemma3:4b)

        Returns:
            Список извлечённых фактов
        """
        extraction_prompt = f"""Проанализируй диалог и извлеки ТОЛЬКО важные долгосрочные факты о пользователе.

    ХОРОШИЕ примеры (извлекай):
    - "Пользователю нравятся кошки"
    - "Пользователь работает программистом"
    - "Пользователь предпочитает чай кофе"
    - "Пользователь живёт в Москве"
    - "У пользователя аллергия на орехи"

    ПЛОХИЕ примеры (НЕ извлекай):
    - Временные вопросы ("какая погода")
    - Текущие команды ("включи музыку")
    - Общие темы без личных предпочтений

    Диалог:
    Пользователь: {user_message}
    Ассистент: {assistant_response}

    Выведи 1-3 важных факта (каждый с новой строки) или напиши "НЕТ" если фактов нет.
    Формат: краткое утверждение без лишних слов."""



        payload = {
            "model": llm_model,
            "prompt": extraction_prompt,
            "stream": False
        }

        try:
            response = requests.post(llm_url, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                facts_text = result.get("response", "").strip()

                if facts_text and facts_text.upper() != "НЕТ":
                    # Разбиваем на отдельные факты
                    facts = [f.strip() for f in facts_text.split('\n') if f.strip()]
                    # Убираем нумерацию типа "1.", "2."
                    facts = [f.lstrip('0123456789.-) ') for f in facts]
                    return facts[:3]  # максимум 3 факта
        except Exception as e:
            print(f"⚠️ Ошибка извлечения памяти: {e}")

        return []

    def manage_memory_conflicts(self, new_fact: str, llm_url: str, llm_model: str) -> str:
        """
        Проверка новых фактов на противоречия со старыми и автоматическое обновление

        Args:
            new_fact: Новый факт для проверки
            llm_url: URL Ollama
            llm_model: Модель для анализа

        Returns:
            Действие: "add" (добавить), "update" (обновить), "skip" (пропустить)
        """
        import requests
        import json

        # Ищем похожие воспоминания
        similar_memories = self.search_memory(new_fact, top_k=3)

        if not similar_memories or similar_memories[0]['relevance'] < 0.7:
            # Нет похожих - просто добавляем
            return {"action": "add", "memory_id": None}

        # Формируем контекст старых воспоминаний
        old_facts_text = "\n".join([
            f"ID: {mem['id'][:8]}... | {mem['content']}"
            for mem in similar_memories if mem['relevance'] > 0.7
        ])

        # Просим LLM проанализировать конфликт
        analysis_prompt = f"""Ты управляешь памятью ИИ-ассистента. Проанализируй новый факт и существующие воспоминания.

    СУЩЕСТВУЮЩИЕ ВОСПОМИНАНИЯ:
    {old_facts_text}

    НОВЫЙ ФАКТ:
    {new_fact}

    ЗАДАЧА: Определи, что делать с новым фактом.

    ВАРИАНТЫ:
    1. ADD - если новый факт дополняет информацию (не противоречит старым)
    2. UPDATE - если новый факт обновляет/исправляет старый (например, "раньше любил кофе, теперь люблю чай")
    3. SKIP - если факт уже есть в памяти (полный дубликат)

    Ответь ТОЛЬКО в формате JSON:
    {{
      "action": "ADD/UPDATE/SKIP",
      "reason": "краткая причина",
      "update_id": "первые 8 символов ID для обновления (если UPDATE) или null"
    }}"""

        payload = {
            "model": llm_model,
            "prompt": analysis_prompt,
            "stream": False
        }

        try:
            response = requests.post(llm_url, json=payload, timeout=30)
            if response.status_code == 200:
                result_text = response.json().get("response", "").strip()

                # Извлекаем JSON из ответа
                import re
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if json_match:
                    decision = json.loads(json_match.group())

                    # Находим полный ID если нужно обновление
                    if decision['action'] == 'UPDATE' and decision.get('update_id'):
                        short_id = decision['update_id']
                        for mem in similar_memories:
                            if mem['id'].startswith(short_id):
                                decision['memory_id'] = mem['id']
                                break

                    return decision
        except Exception as e:
            print(f"⚠️ Ошибка анализа конфликта: {e}")

        # По умолчанию - добавляем
        return {"action": "add", "memory_id": None, "reason": "ошибка анализа"}

    def auto_deduplicate(self, llm_url: str, llm_model: str):
        """
        Автоматическое удаление дубликатов и объединение похожих воспоминаний
        """
        import requests
        import json

        all_memories = self.list_all_memories()

        if len(all_memories) < 2:
            return

        print("🔍 Ищу дубликаты в памяти...")

        processed_ids = set()

        for i, mem1 in enumerate(all_memories):
            if mem1['id'] in processed_ids:
                continue

            # Ищем похожие
            similar = self.search_memory(mem1['content'], top_k=5)
            duplicates = [
                s for s in similar
                if s['id'] != mem1['id']
                   and s['relevance'] > 0.85  # очень похожие
                   and s['id'] not in processed_ids
            ]

            if not duplicates:
                continue

            # Формируем список для объединения
            to_merge = [mem1] + [d for d in all_memories if d['id'] in [dup['id'] for dup in duplicates]]
            merge_text = "\n".join([f"- {m['content']}" for m in to_merge])

            # Просим LLM объединить
            merge_prompt = f"""Объедини эти похожие факты в ОДИН краткий факт:

    {merge_text}

    Ответь ТОЛЬКО объединённым фактом, без пояснений."""

            payload = {"model": llm_model, "prompt": merge_prompt, "stream": False}

            try:
                response = requests.post(llm_url, json=payload, timeout=20)
                if response.status_code == 200:
                    merged_fact = response.json().get("response", "").strip()

                    # Обновляем первый, удаляем остальные
                    self.update_memory(mem1['id'], merged_fact)
                    for dup in duplicates:
                        self.delete_memory(dup['id'])
                        processed_ids.add(dup['id'])

                    print(f"✅ Объединил {len(duplicates) + 1} воспоминаний: {merged_fact[:50]}...")
                    processed_ids.add(mem1['id'])

            except Exception as e:
                print(f"⚠️ Ошибка объединения: {e}")



    def extract_with_llm_verified(self, user_message: str, assistant_response: str,
                                  llm_url: str, llm_model: str) -> List[str]:
        """
        Извлечение с защитой от галлюцинаций
        """
        import requests
        import re

        extraction_prompt = f"""Ты - система извлечения фактов. КРИТИЧЕСКИ ВАЖНО: извлекай ТОЛЬКО факты, которые ЯВНО присутствуют в диалоге.

    ⛔ СТРОГО ЗАПРЕЩЕНО:
    - Придумывать информацию, которой НЕТ в диалоге
    - Додумывать детали
    - Использовать примеры из инструкции как факты

    ✅ РАЗРЕШЕНО извлекать:
    - Прямые утверждения пользователя о себе ("я работаю X", "мне нравится Y")
    - Явные предпочтения и факты

    ДИАЛОГ:
    Пользователь: {user_message}
    Ассистент: {assistant_response}

    ЗАДАЧА: Извлеки 1-2 конкретных факта о пользователе ИЗ ЭТОГО ДИАЛОГА или напиши "НИЧЕГО" если фактов нет.

    Формат ответа (без нумерации):
    Пользователь работает программистом
    Пользователю нравятся кошки"""

        payload = {
            "model": llm_model,
            "prompt": extraction_prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9
            }
        }

        try:
            self.log("Запрос извлечения фактов к LLM...", "INFO")
            response = requests.post(llm_url, json=payload, timeout=30)

            if response.status_code == 200:
                result = response.json()
                facts_text = result.get("response", "").strip()

                self.log(f"LLM ответил: '{facts_text[:100]}...'", "INFO")

                if not facts_text or facts_text.upper() == "НИЧЕГО" or "НИЧЕГО" in facts_text.upper():
                    self.log("Факты не найдены", "INFO")
                    return []

                facts = [f.strip() for f in facts_text.split('\n') if f.strip()]

                cleaned_facts = []
                for fact in facts:
                    fact = re.sub(r'^[\d\.\-\•\)\]\*\s]+', '', fact).strip()

                    if len(fact) > 10 and any(keyword in fact.lower() for keyword in
                                              ['пользовател', 'любит', 'нравится', 'работает', 'предпочитает', 'живет',
                                               'занимается', 'увлекается']):

                        user_words = set(user_message.lower().split())
                        fact_words = set(fact.lower().split())

                        common_words = user_words & fact_words - {'я', 'мне', 'меня', 'мой', 'моя', 'моё', 'мои', 'это',
                                                                  'что', 'и', 'в', 'на', 'с', 'к'}

                        if common_words or len(fact) < 50:
                            cleaned_facts.append(fact)
                            self.log(f"✓ Валидный факт: '{fact}'", "SUCCESS")
                        else:
                            self.log(f"✗ Подозрение на галлюцинацию: '{fact}'", "WARNING")
                    else:
                        self.log(f"✗ Отклонен: '{fact}'", "WARNING")

                return cleaned_facts[:2]

        except Exception as e:
            self.log(f"Ошибка извлечения: {e}", "ERROR")

        return []



    # ✅ ДОБАВЬ ЭТОТ МЕТОД СРАЗУ ПОСЛЕ __init__
    def log(self, message: str, level: str = "INFO"):
        """Логирование операций памяти"""
        if self.debug_mode:
            emoji = {"INFO": "ℹ️", "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌"}
            print(f"{emoji.get(level, 'ℹ️')} [MEMORY] {message}")