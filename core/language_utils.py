# core/language_utils.py
"""Легковесный детектор языка (2 МБ, быстро)"""

try:
    from langdetect import detect, DetectorFactory

    DetectorFactory.seed = 0
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False


class LanguageDetector:
    def __init__(self, default_lang='ru'):
        self.default_lang = default_lang
        if not LANGDETECT_AVAILABLE:
            print("⚠️ langdetect не установлен: pip install langdetect")

    def detect(self, text: str) -> str:
        if not LANGDETECT_AVAILABLE or not text or len(text.strip()) < 3:
            return self.default_lang
        try:
            lang = detect(text)
            return lang if len(lang) == 2 else self.default_lang
        except:
            return self.default_lang
