# core/plugin_loader.py
import os
import importlib.util

_loaded_plugins = {}  # имя -> функция handle

def load_plugins(plugin_folder="plugins"):
    for filename in os.listdir(plugin_folder):
        if filename.endswith(".py") and filename != "__init__.py":
            path = os.path.join(plugin_folder, filename)
            module_name = filename[:-3]

            spec = importlib.util.spec_from_file_location(module_name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if hasattr(module, "handle") and callable(module.handle):
                _loaded_plugins[module_name] = module.handle
                print(f"✅ Загрузен плагин: {module_name}")
            else:
                print(f"⚠️ Пропущен (нет функции handle): {module_name}")

    return list(_loaded_plugins.values())



def run_plugin(plugin_name: str, command: str) -> str | None:
    """
    Запускает конкретный плагин по имени, как обычную функцию.
    :param plugin_name: имя файла плагина без .py (например: hello_plugin)
    :param command: строка команды
    :return: результат работы плагина или None
    """
    # Если плагины ещё не загружены
    if not _loaded_plugins:
        load_plugins("plugins")

    handler = _loaded_plugins.get(plugin_name)
    if not handler:
        raise ValueError(f"❌ Плагин '{plugin_name}' не найден")

    return handler(command)