from TTS.api import TTS
import torch
import numpy as np
import sounddevice as sd
import os
import time
import queue
import threading

print("=" * 60)
print("ОПТИМИЗИРОВАННЫЙ XTTS БЕЗ АРТЕФАКТОВ")
print("=" * 60)


# 1. Определяем путь к модели
def get_model_path():
    if os.name == 'nt':
        username = os.getenv('USERNAME')
        base_path = f"C:\\Users\\{username}\\AppData\\Local\\tts"
    else:
        base_path = os.path.expanduser("~/.local/share/tts")

    return os.path.join(base_path, "tts_models--multilingual--multi-dataset--xtts_v2")


# 2. Загрузка модели
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n🖥️ Устройство: {device}")

print("\n📥 Загрузка XTTS-v2...")
try:
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    model_path = get_model_path()
    config_path = os.path.join(model_path, "config.json")

    if not os.path.exists(config_path):
        print("⚠️ Первый запуск, загрузка модели...")
        tts_api = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        model_path = get_model_path()

    config = XttsConfig()
    config.load_json(config_path)
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=model_path, eval=True)
    model.to(device)

    print(f"✅ Модель загружена!")

except Exception as e:
    print(f"❌ Ошибка: {e}")
    exit()

# 3. Референсный голос
reference_audio = r"..\voices\my_voice_reference.wav"
if not os.path.exists(reference_audio):
    print(f"❌ Файл '{reference_audio}' не найден!")
    exit()

print(f"✅ Референс: {reference_audio}")

# 4. Предвычисление латентов
print("\n🔄 Кэширование эмбеддингов...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
    audio_path=[reference_audio]
)
print("✅ Эмбеддинги готовы!")

# 5. ✅ КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: Непрерывный аудио-стрим с очередью
audio_queue = queue.Queue()
stream_active = threading.Event()


def audio_callback(outdata, frames, time_info, status):
    """Callback для непрерывного воспроизведения без пауз"""
    if status:
        print(f"⚠️ Статус: {status}")

    try:
        # Берём данные из очереди
        data = audio_queue.get_nowait()

        if len(data) < len(outdata):
            # Дополняем тишиной если данных мало
            outdata[:len(data)] = data.reshape(-1, 1)
            outdata[len(data):] = 0
        else:
            outdata[:] = data[:len(outdata)].reshape(-1, 1)
            # Возвращаем остаток обратно в очередь
            if len(data) > len(outdata):
                audio_queue.put(data[len(outdata):])

    except queue.Empty:
        # Тишина если очередь пуста
        outdata.fill(0)


# Создаём непрерывный output stream
output_stream = sd.OutputStream(
    samplerate=24000,
    channels=1,
    callback=audio_callback,
    blocksize=2048  # Оптимальный размер блока
)


def speak_smooth(text):
    """УЛУЧШЕННАЯ ОЗВУЧКА: Плавная без разрывов"""
    print(f"\n🎙️ Генерация: '{text[:50]}...'")
    t0 = time.time()

    try:
        # Генерируем чанки
        chunks = model.inference_stream(
            text,
            "ru",
            gpt_cond_latent,
            speaker_embedding,
            stream_chunk_size=20,  # Больше чанк = меньше артефактов
            enable_text_splitting=True
        )

        # Запускаем stream если ещё не запущен
        if not output_stream.active:
            output_stream.start()

        first = True
        total_samples = 0

        # Собираем все чанки в один массив для плавности
        all_audio = []

        for i, chunk in enumerate(chunks):
            if first:
                print(f"⚡ Первый чанк за: {(time.time() - t0) * 1000:.0f}мс")
                first = False

            chunk_array = chunk.squeeze().cpu().numpy()
            all_audio.append(chunk_array)
            total_samples += len(chunk_array)

        # Объединяем в один массив
        full_audio = np.concatenate(all_audio)

        print(f"✅ Сгенерировано: {total_samples} сэмплов за {(time.time() - t0):.2f}сек")
        print("🔊 Воспроизведение...")

        # Воспроизводим одним куском (БЕЗ пауз!)
        sd.play(full_audio, 24000)
        sd.wait()

        print(f"✅ Завершено: {(time.time() - t0):.2f}сек")

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


# 6. Тестирование
print("\n" + "=" * 60)
print("ТЕСТИРОВАНИЕ ПЛАВНОЙ ОЗВУЧКИ")
print("=" * 60)

test_phrases = [
    "Привет! Теперь звук должен быть плавным и без разрывов.",
    "Я использую улучшенный алгоритм стриминга для качественного звука.",
    "Финальная проверка качества озвучки и плавности воспроизведения."
]

for i, phrase in enumerate(test_phrases, 1):
    print(f"\n--- ТЕСТ {i} из {len(test_phrases)} ---")
    speak_smooth(phrase)

    if i < len(test_phrases):
        print("\n⏸️ Пауза 2 секунды...")
        time.sleep(2)

print("\n" + "=" * 60)
print("🎉 ТЕСТИРОВАНИЕ ЗАВЕРШЕНО!")
print("=" * 60)

# Очистка
if output_stream.active:
    output_stream.stop()
output_stream.close()
