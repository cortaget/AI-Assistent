# language_detector.py
"""
Легковесный детектор языка для голосового ассистента
Использует langdetect (2 МБ) - быстро и эффективно
"""

from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Фиксируем seed для стабильных результатов
DetectorFactory.seed = 0


class LanguageDetector:
    def __init__(self, default_lang='ru'):
        """
        Инициализация детектора языка

        Args:
            default_lang: Язык по умолчанию при ошибке определения
        """
        self.default_lang = default_lang

        # Маппинг языков на коды Vosk
        self.lang_mapping = {
            'ru': 'ru',
            'en': 'en',
            'cs': 'cs',
            'sv': 'sv',
            'de': 'de',
            'fr': 'fr',
            'es': 'es',
            'it': 'it',
            'pl': 'pl',
            'uk': 'uk'
        }

        print("✅ Детектор языка инициализирован")

    def detect(self, text: str) -> str:
        """
        Определение языка текста

        Args:
            text: Текст для анализа

        Returns:
            Код языка (ru, en, cs, sv и т.д.)
        """
        if not text or len(text.strip()) < 3:
            return self.default_lang

        try:
            detected = detect(text)
            # Возвращаем язык, если он в списке поддерживаемых
            return self.lang_mapping.get(detected, self.default_lang)
        except LangDetectException:
            return self.default_lang

    def detect_batch(self, texts: list) -> list:
        """
        Определение языка для нескольких текстов

        Args:
            texts: Список текстов

        Returns:
            Список кодов языков
        """
        return [self.detect(text) for text in texts]

    def is_supported(self, lang_code: str) -> bool:
        """Проверка поддержки языка"""
        return lang_code in self.lang_mapping.values()
