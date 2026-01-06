# plugins/memory_plugin.py

memory = {}

def handle(command: str) -> str | None:
    command_lower = command.lower()

    # Запоминание информации
    if "запомни" in command_lower:
        info = command_lower.split("запомни", 1)[1].strip()
        if info:
            memory_key = info  # простое хранение всей строки
            memory[memory_key] = info
            return f"Я запомнил: {info}"  # вернётся в main и добавится в историю
        else:
            return "Что именно мне запомнить?"

    # Вопросы о том, что запомнили
    for value in memory.values():
        if any(word in command_lower for word in value.split()):
            return f"Я помню: {value}"

    return None
