from TTS.api import TTS
import torch
import wave
import os


# ============ ШАГ 1: АВТОМАТИЧЕСКИЙ ПОИСК ВСЕХ .WAV ФАЙЛОВ ============
def find_all_wav_files(root_directory):
    """
    Ищет все .wav файлы в указанной папке и всех подпапках

    Args:
        root_directory: Путь к главной папке

    Returns:
        Список путей ко всем найденным .wav файлам
    """
    wav_files = []

    print(f"🔍 Поиск .wav файлов в {root_directory}")

    # os.walk рекурсивно обходит все подпапки
    for root, dirs, files in os.walk(root_directory):
        for file in files:
            if file.lower().endswith('.wav'):  # .wav или .WAV
                full_path = os.path.join(root, file)
                wav_files.append(full_path)
                print(f"  ✅ Найден: {file} (в {os.path.basename(root)})")

    print(f"\n📊 Всего найдено файлов: {len(wav_files)}")
    return wav_files


# ============ ШАГ 2: ОБЪЕДИНЕНИЕ ФАЙЛОВ ============
def merge_wav_files(input_paths, output_path="merged_reference.wav"):
    """Объединяет несколько .wav файлов в один"""
    if not input_paths:
        print("❌ Не найдено .wav файлов для объединения!")
        return None

    data = []
    total_duration = 0

    print("\n🔗 Объединяю файлы...")
    for i, clip_path in enumerate(input_paths, 1):
        try:
            w = wave.open(clip_path, "rb")
            duration = w.getnframes() / w.getframerate()
            total_duration += duration

            data.append([w.getparams(), w.readframes(w.getnframes())])
            w.close()

            print(f"  {i}/{len(input_paths)}: {os.path.basename(clip_path)} ({duration:.1f}с)")
        except Exception as e:
            print(f"  ⚠️ Ошибка при чтении {os.path.basename(clip_path)}: {e}")
            continue

    if not data:
        print("❌ Не удалось прочитать ни один файл!")
        return None

    # Создаём итоговый файл
    output = wave.open(output_path, "wb")
    output.setparams(data[0][0])

    for params, frames in data:
        output.writeframes(frames)

    output.close()

    print(f"\n✅ Готово! Сохранено в {output_path}")
    print(f"⏱️ Общая длительность: {total_duration:.1f} секунд")

    # Проверка длительности
    if total_duration < 5:
        print("⚠️ ВНИМАНИЕ: Референсное аудио короче 5 секунд - качество может быть низким!")
    elif total_duration > 60:
        print("⚠️ ВНИМАНИЕ: Референсное аудио длиннее 60 секунд - будет использовано ~30 секунд")

    return output_path


# ============ ОСНОВНОЙ КОД ============

# 👇 УКАЖИ ЗДЕСЬ ПУТЬ К СВОЕЙ ГЛАВНОЙ ПАПКЕ
MAIN_FOLDER = r"C:\Users\Максим\Downloads\3284455619 Murder Drones - Cyn Tessa Hunter Voicelines\sound\player\hunter"  # Замени на свой путь!

# Ищем все .wav файлы во всех подпапках
all_wav_files = find_all_wav_files(MAIN_FOLDER)

if not all_wav_files:
    print("\n❌ В указанной папке не найдено .wav файлов!")
    print("Проверь путь к папке и наличие .wav файлов")
    exit()

# Объединяем все найденные файлы
reference_file = merge_wav_files(all_wav_files, output_path="../voices/my_voice_reference.wav")

if not reference_file:
    print("❌ Не удалось создать референсный файл!")
    exit()

# ============ ШАГ 3: ЗАГРУЗКА МОДЕЛИ И КЛОНИРОВАНИЕ ============
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n🖥️ Используется устройство: {device}")

print("📥 Загрузка модели XTTS-v2... (может занять 1-2 минуты)")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)


def clone_voice(text, output_path="output.wav"):
    """Генерирует речь твоим клонированным голосом"""
    print(f"\n🎙️ Генерирую: '{text[:50]}...'")
    tts.tts_to_file(
        text=text,
        file_path=output_path,
        speaker_wav=reference_file,
        language="ru"
    )
    print(f"✅ Готово! Сохранено в {output_path}")


# ============ ТЕСТ ============
print("\n" + "=" * 60)
print("🧪 ТЕСТ КЛОНИРОВАНИЯ ГОЛОСА")
print("=" * 60)

clone_voice(
    text="Привет! Это клонированная версия моего голоса. Теперь я могу говорить что угодно.",
    output_path="test_clone.wav"
)

print("\n🎧 Послушай файл test_clone.wav чтобы проверить результат!")
