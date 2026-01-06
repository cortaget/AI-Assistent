from datetime import datetime

def get_current_time() -> str:
    """Получить текущее время на ПК"""
    now = datetime.now()
    return now.strftime("%H:%M:%S, %d.%m.%Y")


if __name__ == "__main__":
    print("Текущее время:", get_current_time())

