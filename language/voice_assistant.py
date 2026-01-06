
import pyttsx3

class MultilingualTTS:
    """Мультиязычная система озвучивания"""

    def __init__(self):
        self.tts = pyttsx3.init()
        self.voices_map = self._build_voices_map()
        self.current_lang = 'ru'

        # Устанавливаем русский по умолчанию
        self.set_language('ru')

        print(f"🔊 Доступные голоса для озвучки: {list(self.voices_map.keys())}")

    def _build_voices_map(self) -> dict:
        """Создаёт карту языков и голосов"""
        voices = self.tts.getProperty('voices')
        lang_map = {}

        # Словарь соответствий (ID языка Windows -> наш код)
        language_patterns = {
            'ru': ['russian', 'irina', 'ru-ru', 'ru_ru'],
            'en': ['english', 'zira', 'david', 'en-us', 'en-gb', 'en_us'],
            'de': ['german', 'de-de', 'hedda', 'de_de'],
            'fr': ['french', 'fr-fr', 'hortense', 'fr_fr'],
            'es': ['spanish', 'es-es', 'helena', 'es_es'],
            'it': ['italian', 'it-it', 'it_it'],
            'pl': ['polish', 'pl-pl', 'pl_pl'],
            'cs': ['czech', 'cs-cz', 'cs_cz'],
            'sv': ['swedish', 'sv-se', 'sv_se'],
            'pt': ['portuguese', 'pt-pt', 'pt-br', 'pt_br'],
            'zh': ['chinese', 'zh-cn', 'zh_cn', 'huihui'],
            'ja': ['japanese', 'ja-jp', 'ja_jp', 'haruka']
        }

        for voice in voices:
            voice_name = voice.name.lower()
            voice_lang = voice.languages[0].lower() if voice.languages else ""

            for lang_code, patterns in language_patterns.items():
                if any(pattern in voice_name or pattern in voice_lang for pattern in patterns):
                    if lang_code not in lang_map:
                        lang_map[lang_code] = voice.id
                        print(f"  ✓ Найден голос для {lang_code}: {voice.name}")
                    break

        # Fallback на английский если нет русского
        if 'ru' not in lang_map and 'en' in lang_map:
            lang_map['ru'] = lang_map['en']

        return lang_map

    def set_language(self, lang_code: str):
        """Переключает язык озвучки"""
        if lang_code in self.voices_map:
            self.tts.setProperty('voice', self.voices_map[lang_code])
            self.current_lang = lang_code
        else:
            # Fallback на английский или первый доступный
            fallback = self.voices_map.get('en') or list(self.voices_map.values())[0]
            self.tts.setProperty('voice', fallback)
            print(f"⚠️ Голос для '{lang_code}' не найден, используется fallback")

    def speak(self, text: str, lang_code: str = None):
        """Озвучивает текст на нужном языке"""
        if lang_code and lang_code != self.current_lang:
            self.set_language(lang_code)

        self.tts.say(text)
        self.tts.runAndWait()


# ЗАМЕНИТЕ СТАРЫЙ speech_worker() НА ЭТОТ:

multilingual_tts = None


def speech_worker():
    """Поток для мультиязычной озвучивания"""
    global multilingual_tts
    multilingual_tts = MultilingualTTS()

    while True:
        item = speech_queue.get()
        if item is None:
            break

        # item теперь кортеж: (text, language)
        if isinstance(item, tuple):
            text, lang = item
            multilingual_tts.speak(text, lang)
        else:
            # Обратная совместимость
            multilingual_tts.speak(item)

        speech_queue.task_done()


# ОБНОВИТЕ ФУНКЦИЮ speak():

def speak(text: str, lang_code: str = None):
    """Добавить текст в очередь озвучки с языком"""
    speech_queue.put((text, lang_code) if lang_code else text)


# ОБНОВИТЕ query_llm_stream() - добавьте определение языка для озвучки:

def query_llm_stream(user_input: str):
    # ... существующий код ...

    # В конце, перед return reply:
    query_lang = lang_detector.detect(user_input)
    return reply, query_lang  # Возвращаем язык тоже


# ОБНОВИТЕ next_prompt():

def next_prompt(user_input: str, use_voice: bool = False, detect_lang: bool = False) -> str:
    print(f"\n👤 {user_input}")

    detected_lang = None
    if detect_lang:
        detected_lang = lang_detector.detect(user_input)
        print(f"🌍 Определён язык: {detected_lang}")

    # Проверка плагинов
    plugin_handlers = load_plugins()
    for handler in plugin_handlers:
        result = handler(user_input)
        if result:
            if use_voice:
                speak(result, detected_lang)
            return result

    # Основной запрос
    reply_result = query_llm_stream(user_input)

    # Распаковываем если возвращается кортеж
    if isinstance(reply_result, tuple):
        reply, reply_lang = reply_result
    else:
        reply = reply_result
        reply_lang = detected_lang

    if use_voice:
        speak(reply, reply_lang)

    return reply
