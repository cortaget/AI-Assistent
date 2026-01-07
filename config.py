# config.py
import os


class Config:
    """Централизованная конфигурация ассистента"""

    # LLM настройки
    LLM_URL = "http://127.0.0.1:11434/api/generate"
    LLM_MODEL = "gemma3:4b"

    # Пути к данным
    MEMORY_DB_PATH = "./memory_db"
    MEMORY_COLLECTION_NAME = "assistant_memory"
    VOSK_MODEL_PATH = "models\\vosk-model-small-ru-0.22"
    TOOLS_DIR = "tools"

    # Настройки аудио
    AUDIO_RATE = 16000
    AUDIO_CHUNK_SIZE = 4096

    # Настройки TTS
    TTS_VOICE_NAME = "irina"
    TTS_RATE = 160

    # Настройки памяти
    MEMORY_DEBUG_MODE = False
    MEMORY_TOP_K = 10
    MEMORY_RELEVANCE_THRESHOLD = 0.3

    # Настройки обработки памяти
    MEMORY_INTENT_DETECTION_TOKENS = 5
    MEMORY_FACT_EXTRACTION_TOKENS = 30
    MEMORY_SEARCH_QUERY_TOKENS = 20

    # Пороги релевантности
    DUPLICATE_THRESHOLD = 0.92
    SIMILAR_THRESHOLD = 0.80
    UPDATE_THRESHOLD = 0.70
    DELETE_THRESHOLD = 0.65

    # Настройки инструментов
    MAX_TOOL_ITERATIONS = 5
    TOOL_TIMEOUT = 100

    # Настройки LLM
    LLM_TEMPERATURE = 0.3
    LLM_QUICK_CALL_TEMPERATURE = 0.1
    LLM_QUICK_CALL_TOP_K = 10
    LLM_QUICK_CALL_TOP_P = 0.5



    ENABLE_MULTILINGUAL = True  # Включить/выключить мультиязычность
    DEFAULT_LANGUAGE = "ru"  # Или "en" если хотите английский по умолчанию