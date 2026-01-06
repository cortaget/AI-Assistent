def handle(command: str) -> str | None:
    if "привет" in command.lower():
        return "Привет, хозяин! Тебя зовут Максим, верно?"
    return None
