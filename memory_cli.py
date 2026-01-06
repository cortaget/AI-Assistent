# memory_cli.py
from memory_manager import MemoryManager
import sys


def main():
    memory = MemoryManager()

    while True:
        print("\n📝 Управление памятью ассистента")
        print("1. Показать всю память")
        print("2. Добавить запись")
        print("3. Поиск в памяти")
        print("4. Удалить запись")
        print("5. Очистить всю память")
        print("6. Выход")

        choice = input("\nВыберите действие: ")

        if choice == "1":
            memories = memory.list_all_memories()
            if not memories:
                print("❌ Память пуста")
            else:
                print(f"\n✅ Найдено {len(memories)} записей:\n")
                for mem in memories:
                    print(f"ID: {mem['id']}")
                    print(f"Содержание: {mem['content']}")
                    print(f"Тип: {mem['metadata'].get('type', 'unknown')}")
                    print(f"Создано: {mem['metadata'].get('created_at', 'unknown')}")
                    print("-" * 50)

        elif choice == "2":
            content = input("Введите текст для запоминания: ")
            mem_type = input("Тип памяти (user_info/rule/preference/fact): ") or "user_info"
            memory.add_memory(content, memory_type=mem_type)

        elif choice == "3":
            query = input("Поисковый запрос: ")
            results = memory.search_memory(query, top_k=5)
            if results:
                print(f"\n✅ Найдено {len(results)} релевантных записей:\n")
                for i, mem in enumerate(results, 1):
                    print(f"{i}. [{mem['relevance']:.2f}] {mem['content']}")
                    print(f"   ID: {mem['id']}")
                    print()
            else:
                print("❌ Ничего не найдено")

        elif choice == "4":
            mem_id = input("ID записи для удаления: ")
            memory.delete_memory(mem_id)

        elif choice == "5":
            confirm = input("⚠️ Удалить всю память? (да/нет): ")
            if confirm.lower() == "да":
                memory.clear_all_memories()

        elif choice == "6":
            break


if __name__ == "__main__":
    main()
