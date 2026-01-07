# core/speech_manager.py
import json
import queue
import threading
import pyaudio
import pyttsx3
from vosk import Model, KaldiRecognizer
from config import Config


class SpeechManager:
    def __init__(self, config: Config = None):
        self.config = config or Config()

        # Инициализация Vosk
        self.model = Model(self.config.VOSK_MODEL_PATH)
        self.recognizer = KaldiRecognizer(self.model, self.config.AUDIO_RATE)

        # PyAudio
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.config.AUDIO_RATE,
            input=True,
            frames_per_buffer=self.config.AUDIO_CHUNK_SIZE
        )
        self.stream.start_stream()

        # TTS очередь и поток
        self.speech_queue = queue.Queue()
        self.tts = None
        self.voice_map = {}
        self.speech_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.speech_thread.start()

    def _speech_worker(self):
        self.tts = pyttsx3.init()
        voices = self.tts.getProperty('voices')

        # Карта голосов для разных языков
        lang_patterns = {
            'ru': ['russian', 'irina', 'ru'],
            'en': ['english', 'zira', 'david', 'en'],
            'de': ['german', 'hedda', 'de'],
            'fr': ['french', 'hortense', 'fr'],
            'es': ['spanish', 'helena', 'es']
        }

        # Строим карту доступных голосов
        for voice in voices:
            voice_name_lower = voice.name.lower()
            for lang, patterns in lang_patterns.items():
                if any(p in voice_name_lower for p in patterns):
                    if lang not in self.voice_map:
                        self.voice_map[lang] = voice.id

        # Устанавливаем голос по умолчанию
        if self.config.TTS_VOICE_NAME:
            for voice in voices:
                if self.config.TTS_VOICE_NAME in voice.name.lower():
                    self.tts.setProperty('voice', voice.id)
                    break

        self.tts.setProperty('rate', self.config.TTS_RATE)

        # Обработка очереди
        while True:
            item = self.speech_queue.get()
            if item is None:
                break

            # Поддержка кортежа (text, lang) или просто text
            if isinstance(item, tuple):
                text, lang = item
                if lang in self.voice_map:
                    self.tts.setProperty('voice', self.voice_map[lang])
            else:
                text = item

            self.tts.say(text)
            self.tts.runAndWait()
            self.speech_queue.task_done()

    def speak(self, text: str, lang: str = None):
        if lang:
            self.speech_queue.put((text, lang))
        else:
            self.speech_queue.put(text)

    def listen_command(self) -> str:
        print("🎙️ Говори...")
        try:
            self.stream.read(self.stream.get_read_available(), exception_on_overflow=False)
        except:
            pass

        while True:
            data = self.stream.read(self.config.AUDIO_CHUNK_SIZE, exception_on_overflow=False)
            if self.recognizer.AcceptWaveform(data):
                result = json.loads(self.recognizer.Result())
                text = result.get("text", "")
                if text:
                    print(f"📝 {text}")
                    return text

    def stop(self):
        self.speech_queue.put(None)
        self.speech_thread.join(timeout=2)
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
