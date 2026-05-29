"""
Выдача варианта лабораторной работы по ФИО студента.
Запуск: python genvar.py -name ИвановИИ -group ИВТб-1301 [-save]
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

def load_config(path: str = "variant.json") -> dict:
    if not Path(path).exists():
        print(f" Файл {path} не найден. Поместите JSON в ту же директорию.")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_variant_id(full_name: str, group: str, total: int) -> int:
    seed = f"{full_name.strip().lower()}{group.strip().upper()}"
    h = int(hashlib.sha256(seed.encode("utf-8")).hexdigest(), 16)
    return (h % total) + 1

def format_variant(v: dict) -> str:
    lines = [
        f"ВАРИАНТ {v['id']}: {v['title']}",
        "=" * 60,
        f"Описание: {v['description']}",
        "",
        " Поля:",
    ]
    for field in v["fields"]:
        lines.append(f"  • {field}")
    
    lines.extend([
        "",
        f"Очистка: {v['cleanup_rule']}",
        f"Пояснение: {v['cleanup_note']}",
        "",
        f"Группировка: {v['grouping_rule']}",
        f"Пояснение: {v['grouping_note']}",
        "",
        f"Окно: {v['window_rule']}",
        f"Пояснение: {v['window_note']}",
        "",
        f"Акцент: {v['accent_rule']}",
        f"Пояснение: {v['accent_note']}",
        "=" * 60
    ])
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Выдача варианта лабораторной по NumPy")
    parser.add_argument("-name", required=True, help="ФИО студента (например, 'Иванов И.И.')")
    parser.add_argument("-group", required=True, help="Номер группы (например, 'ИВТб-1301')")
    parser.add_argument("-save", action="store_true", help="Сохранить задание в файл .txt")
    args = parser.parse_args()

    config = load_config()
    total = len(config["variants"])
    var_id = get_variant_id(args.name, args.group, total)
    variant = config["variants"][var_id - 1]

    text = f"Студент: {args.name} | Группа: {args.group}\nВариант: #{var_id}\n\n" + format_variant(variant)
    print(text)

    if args.save:
        safe_name = "".join(c if c.isalnum() or c in ".- " else "_" for c in args.name).replace(" ", "_")
        filename = f"variant_{safe_name}_{args.group}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)
        print(f" Задание сохранено в: {filename}")

main()