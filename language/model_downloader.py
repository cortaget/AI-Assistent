# model_downloader.py
"""
Автоматическое скачивание и распаковка моделей Vosk
"""

import os
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm


class VoskModelDownloader:
    def __init__(self, models_dir="models"):
        """
        Инициализация загрузчика моделей

        Args:
            models_dir: Папка для хранения моделей
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)

        # База данных доступных моделей
        self.available_models = {
            'ru': {
                'url': 'https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip',
                'folder': 'vosk-model-small-ru-0.22',
                'size': '45 МБ'
            },
            'en': {
                'url': 'https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip',
                'folder': 'vosk-model-small-en-us-0.15',
                'size': '40 МБ'
            },
            'cs': {
                'url': 'https://alphacephei.com/vosk/models/vosk-model-small-cs-0.4-rhasspy.zip',
                'folder': 'vosk-model-small-cs-0.4-rhasspy',
                'size': '44 МБ'
            },
            'sv': {
                'url': 'https://alphacephei.com/vosk/models/vosk-model-small-sv-rhasspy-0.15.zip',
                'folder': 'vosk-model-small-sv-rhasspy-0.15',
                'size': '40 МБ'
            },
            'de': {
                'url': 'https://alphacephei.com/vosk/models/vosk-model-small-de-0.15.zip',
                'folder': 'vosk-model-small-de-0.15',
                'size': '45 МБ'
            },
            'fr': {
                'url': 'https://alphacephei.com/vosk/models/vosk-model-small-fr-0.22.zip',
                'folder': 'vosk-model-small-fr-0.22',
                'size': '41 МБ'
            },
            'es': {
                'url': 'https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip',
                'folder': 'vosk-model-small-es-0.42',
                'size': '39 МБ'
            },
            'it': {
                'url': 'https://alphacephei.com/vosk/models/vosk-model-small-it-0.22.zip',
                'folder': 'vosk-model-small-it-0.22',
                'size': '48 МБ'
            },
            'pl': {
                'url': 'https://alphacephei.com/vosk/models/vosk-model-small-pl-0.22.zip',
                'folder': 'vosk-model-small-pl-0.22',
                'size': '50 МБ'
            },
            'uk': {
                'url': 'https://alphacephei.com/vosk/models/vosk-model-small-uk-v3-small.zip',
                'folder': 'vosk-model-small-uk-v3-small',
                'size': '73 МБ'
            }
        }

    def is_model_downloaded(self, lang_code: str) -> bool:
        """
        Проверка наличия модели

        Args:
            lang_code: Код языка (ru, en, cs и т.д.)

        Returns:
            True если модель скачана
        """
        if lang_code not in self.available_models:
            return False

        model_path = self.models_dir / self.available_models[lang_code]['folder']
        return model_path.exists() and model_path.is_dir()

    def get_model_path(self, lang_code: str) -> str:
        """
        Получение пути к модели

        Args:
            lang_code: Код языка

        Returns:
            Путь к папке модели
        """
        if lang_code not in self.available_models:
            return None

        return str(self.models_dir / self.available_models[lang_code]['folder'])

    def download_model(self, lang_code: str, force=False) -> bool:
        """
        Скачивание и распаковка модели

        Args:
            lang_code: Код языка
            force: Принудительная перезагрузка

        Returns:
            True если успешно
        """
        if lang_code not in self.available_models:
            print(f"❌ Язык '{lang_code}' не поддерживается")
            print(f"   Доступные языки: {', '.join(self.available_models.keys())}")
            return False

        model_info = self.available_models[lang_code]
        model_path = self.models_dir / model_info['folder']

        # Проверяем наличие
        if model_path.exists() and not force:
            print(f"✅ Модель '{lang_code}' уже скачана: {model_path}")
            return True

        # Скачиваем
        zip_path = self.models_dir / f"{model_info['folder']}.zip"

        try:
            print(f"📥 Скачивание модели '{lang_code}' ({model_info['size']})...")
            print(f"   URL: {model_info['url']}")

            response = requests.get(model_info['url'], stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            # Скачиваем с прогресс-баром
            with open(zip_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Модель {lang_code}") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            print(f"📦 Распаковка модели '{lang_code}'...")

            # Распаковываем
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.models_dir)

            # Удаляем zip
            zip_path.unlink()

            print(f"✅ Модель '{lang_code}' успешно установлена: {model_path}")
            return True

        except Exception as e:
            print(f"❌ Ошибка при скачивании модели '{lang_code}': {e}")
            # Удаляем частично скачанные файлы
            if zip_path.exists():
                zip_path.unlink()
            return False

    def download_multiple(self, lang_codes: list) -> dict:
        """
        Скачивание нескольких моделей

        Args:
            lang_codes: Список кодов языков

        Returns:
            Словарь {lang: success}
        """
        results = {}

        for lang in lang_codes:
            results[lang] = self.download_model(lang)

        return results

    def list_downloaded_models(self) -> list:
        """Список скачанных моделей"""
        downloaded = []

        for lang_code, info in self.available_models.items():
            if self.is_model_downloaded(lang_code):
                downloaded.append(lang_code)

        return downloaded

    def list_available_models(self) -> list:
        """Список доступных для скачивания моделей"""
        return list(self.available_models.keys())


# Вспомогательная функция для использования в других модулях
def ensure_model_available(lang_code: str, models_dir="models") -> str:
    """
    Гарантирует наличие модели (скачивает если нет)

    Args:
        lang_code: Код языка
        models_dir: Папка моделей

    Returns:
        Путь к модели или None
    """
    downloader = VoskModelDownloader(models_dir)

    if not downloader.is_model_downloaded(lang_code):
        print(f"⚠️ Модель '{lang_code}' не найдена, начинаю скачивание...")
        if not downloader.download_model(lang_code):
            return None

    return downloader.get_model_path(lang_code)
