# Лабораторная 6 — NumPy-предобработка

**Студент:** Изместьев М.Н., гр. ИВТб-1302-06-00  
**Вариант:** 10 — спортивная аналитика  
**GitHub (upload):** `lab-6-numpy-preprocessing`

## Статус

| Компонент | Статус |
|-----------|--------|
| `genvar.py` + `variant.json` | ✅ |
| `gen_data.py` → `data.csv` (12000 строк) | ✅ |
| `lab_variant10_numpy.py` (10 этапов) | ✅ |
| `lab6_colab.ipynb` (Colab) | ✅ |
| Отчёт `report/main.tex` | ✅ |
| Скрин вывода Colab `report/img/1.jpeg` | ✅ |
| Выходные файлы `outputs/` | ✅ |
| Ссылка на GitHub в отчёте | ❌ |

## Поля варианта 10

| Поле | Тип | Смысл |
|------|-----|-------|
| `ts` | int32 | Unix timestamp |
| `athlete_id` | int16 | ID спортсмена |
| `dist` | float32 | дистанция, км |
| `pace` | float32 | темп, мин/км |
| `cal` | float32 | калории |
| `zone` | uint8 | пульсовая зона 1–5 |

## Структура

```
LABA_6/
├── genvar.py / variant.json
├── gen_data.py / data.csv
├── lab_variant10_numpy.py    # локальный запуск
├── lab6_colab.ipynb          # основной для Colab / сдачи
├── outputs/                  # результаты Colab + скрины
├── drawio/                   # схемы (не в отчёт)
└── report/
    ├── main.tex
    └── img/1.jpeg            # скрин вывода Colab
```

## Запуск

**Google Colab (рекомендуется):**
1. https://colab.research.google.com/
2. Файл → Загрузить блокнот → `lab6_colab.ipynb`
3. Среда выполнения → Выполнить все
4. Скачать `.npy`/`.csv` из последней ячейки

**Локально:**
```powershell
cd c:\Users\stud222640\Documents\proga\LABA_6
..\.venv\Scripts\python.exe lab_variant10_numpy.py --input data.csv
```

## Выходные файлы

- `stage1_structured.npy` — 12000 строк, 6 полей
- `stage3_normalized.npy` — + z_dist
- `conditional_stats.csv`, `conditional_stats_with_p90.csv`
- `final_preprocessed.npy` — 11559 строк, 13 полей

Пример вывода (Colab): 441 аномалия (3.67%), 50 групп, IQR 0%.

## Сдача

PDF из `report/main.tex` + ссылка на репозиторий (блокнот в репо).
