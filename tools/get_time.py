# tools/get_time.py
from datetime import datetime
from tools.tool_manager import Tool

def get_current_time() -> str:
    """Получить текущее время на ПК"""
    now = datetime.now()
    return now.strftime("%H:%M:%S, %d.%m.%Y")

def register_tool() -> Tool:
    """Регистрация инструмента в системе"""
    return Tool(
        name="get_time",
        description="Получает текущее время и дату с компьютера",
        usage_context="Когда пользователь спрашивает о текущем времени, дате или дне",
        function=get_current_time,
        parameters={},
        keywords=["время", "час", "дата", "сейчас", "который", "когда"],
        examples=[  # ← НОВОЕ: Примеры ПРАВИЛЬНЫХ запросов
            "сколько сейчас времени?",
            "который час?",
            "какое сегодня число?",
            "какая сейчас дата?",
            "когда сейчас?",
            "время покажи",
            "скажи время"
        ],
        parameter_extractor=None,  # ← Экстрактор не нужен!
        categories=["time", "datetime", "system"]
    )
