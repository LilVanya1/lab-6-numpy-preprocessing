"""
Генерация синтетического CSV для лабораторной.
Запуск: python gen_data.py -variant N -seed YYYYMMDD [-output data.csv]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


DTYPE_MAP = {
    "int16": np.int16,
    "int32": np.int32,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "float32": np.float32,
    "float64": np.float64,
}


def load_variant(variant_id: int, cfg_path: str = "variant.json") -> dict:
    path = Path(cfg_path)
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл {cfg_path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    variants = cfg.get("variants", [])
    if not variants:
        raise ValueError("В variant.json нет variants")
    if variant_id < 1 or variant_id > len(variants):
        raise ValueError(f"variant должен быть в диапазоне [1, {len(variants)}]")
    return variants[variant_id - 1]


def parse_field(field_text: str) -> tuple[str, str]:
    m = re.match(r"^\s*([A-Za-z0-9_]+)\s*\(([^)]+)\)\s*:", field_text)
    if not m:
        raise ValueError(f"Не удалось распарсить поле: {field_text}")
    return m.group(1), m.group(2).strip().lower()


def parse_hint_range(field_text: str) -> tuple[float, float] | None:
    normalized = field_text.replace("–", "-")
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)", normalized)
    if m:
        lo = float(m.group(1))
        hi = float(m.group(2))
        if lo < hi:
            return lo, hi
    m = re.search(r"\[(\-?\d+(?:\.\d+)?),\s*(\-?\d+(?:\.\d+)?)\]", normalized)
    if m:
        lo = float(m.group(1))
        hi = float(m.group(2))
        if lo < hi:
            return lo, hi
    return None


def base_float_by_name(name: str, rng: np.random.Generator, n: int) -> np.ndarray:
    lname = name.lower()
    if lname == "dist":
        return rng.gamma(shape=2.2, scale=2.2, size=n).astype(np.float32)
    if lname == "pace":
        return np.clip(rng.normal(loc=5.7, scale=1.1, size=n), 2.5, 12.0).astype(np.float32)
    if lname == "cal":
        return rng.normal(loc=430.0, scale=140.0, size=n).astype(np.float32)
    if "temp" in lname:
        return rng.normal(loc=24.0, scale=8.0, size=n).astype(np.float32)
    if "speed" in lname or lname == "sp":
        return np.clip(rng.normal(loc=38.0, scale=14.0, size=n), 0.0, None).astype(np.float32)
    if "price" in lname or "fare" in lname or "rate" in lname or "amt" in lname:
        return np.clip(rng.lognormal(mean=5.1, sigma=0.45, size=n), 1.0, None).astype(np.float32)
    if "press" in lname or "volt" in lname:
        return np.clip(rng.normal(loc=50.0, scale=12.0, size=n), 0.0, None).astype(np.float32)
    if "hum" in lname or "load" in lname or "eff" in lname:
        return np.clip(rng.normal(loc=55.0, scale=20.0, size=n), 0.0, 100.0).astype(np.float32)
    if "co2" in lname:
        return np.clip(rng.normal(loc=650.0, scale=180.0, size=n), 250.0, None).astype(np.float32)
    if "lat" in lname or "delay" in lname:
        return np.clip(rng.normal(loc=50.0, scale=30.0, size=n), 0.0, None).astype(np.float32)
    if "churn" in lname or "loss" in lname or "ctr" in lname or "err_r" in lname:
        return np.clip(rng.beta(a=2.0, b=12.0, size=n), 0.0, 1.0).astype(np.float32)
    return np.clip(rng.normal(loc=50.0, scale=18.0, size=n), 0.0, None).astype(np.float32)


def add_numeric_noise(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = values.astype(np.float32, copy=True)
    n = out.size
    if n == 0:
        return out
    idx_neg = rng.choice(n, max(1, n // 120), replace=False)
    idx_nan = rng.choice(n, max(1, n // 220), replace=False)
    idx_inf = rng.choice(n, max(1, n // 420), replace=False)
    out[idx_neg] *= -1.0
    out[idx_nan] = np.nan
    out[idx_inf] = np.inf
    return out


def build_column(
    variant_id: int,
    field_text: str,
    col_name: str,
    col_dtype: str,
    rng: np.random.Generator,
    n: int,
) -> np.ndarray:
    np_dtype = DTYPE_MAP.get(col_dtype)
    if np_dtype is None:
        raise ValueError(f"Неизвестный dtype: {col_dtype} ({col_name})")

    hint_range = parse_hint_range(field_text)
    lname = col_name.lower()

    if lname == "ts":
        start = 1700000000 + int(rng.integers(0, 500000))
        ts = start + np.arange(n, dtype=np.int64) * int(rng.integers(1, 8))
        return ts.astype(np_dtype, copy=False)

    if lname.endswith("_id") or lname in {"id", "srv_hash", "line_id", "well_id"}:
        return rng.integers(1, 81, size=n, dtype=np_dtype)

    if np.issubdtype(np_dtype, np.floating):
        vals = base_float_by_name(col_name, rng, n)
        if hint_range is not None:
            lo, hi = hint_range
            # Расширяем диапазон, чтобы присутствовали выбросы.
            vals = rng.normal(loc=(lo + hi) / 2.0, scale=(hi - lo) / 3.0 + 1e-6, size=n).astype(np.float32)
        vals = add_numeric_noise(vals, rng)
        return vals.astype(np_dtype, copy=False)

    if np_dtype == np.uint8:
        if hint_range is not None:
            lo, hi = hint_range
            lo_i = max(0, int(np.floor(lo)))
            hi_i = max(lo_i + 1, int(np.ceil(hi)) + 1)
            return rng.integers(lo_i, hi_i, size=n, dtype=np.uint8)
        if lname in {"zone", "diff"}:
            return rng.integers(1, 6, size=n, dtype=np.uint8)
        if lname in {"acc", "pump", "fault", "calib"}:
            return rng.integers(0, 2, size=n, dtype=np.uint8)
        return rng.integers(0, 16, size=n, dtype=np.uint8)

    if np.issubdtype(np_dtype, np.integer):
        lo_i, hi_i = 0, 1000
        if hint_range is not None:
            lo, hi = hint_range
            lo_i = int(np.floor(lo))
            hi_i = int(np.ceil(hi)) + 1
        if lname in {"steps", "items", "qty", "orders", "viewers"}:
            lo_i, hi_i = 0, 350
        if lname in {"attempts", "shows", "bpm"}:
            lo_i, hi_i = 1, 12
        return rng.integers(lo_i, max(lo_i + 1, hi_i), size=n, dtype=np_dtype)

    raise ValueError(f"Неподдерживаемый dtype для {col_name}: {col_dtype}")


def generate_data(variant: dict, variant_id: int, seed: int, n_rows: int = 12000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fields = [parse_field(x) for x in variant["fields"]]
    dtype = np.dtype([(name, DTYPE_MAP[dt]) for name, dt in fields], align=False)
    data = np.empty(n_rows, dtype=dtype)

    for raw_field, (name, dt) in zip(variant["fields"], fields):
        data[name] = build_column(variant_id, raw_field, name, dt, rng, n_rows)

    # Легкая предметная корреляция для варианта 10 (спорт).
    if variant_id == 10 and {"dist", "pace", "cal"}.issubset(data.dtype.names):
        dist = data["dist"].astype(np.float32)
        pace = data["pace"].astype(np.float32)
        cal = data["cal"].astype(np.float32)
        model_cal = (dist * (85.0 / np.maximum(pace, 1e-3)) * 6.0).astype(np.float32)
        blend = rng.uniform(0.35, 0.65, size=n_rows).astype(np.float32)
        data["cal"] = (blend * cal + (1.0 - blend) * model_cal).astype(np.float32)

    return data


def save_csv(data: np.ndarray, out_path: str) -> None:
    header = ",".join(data.dtype.names)
    matrix = np.column_stack([data[name] for name in data.dtype.names])
    fmt = []
    for dt in data.dtype.fields.values():
        kind = dt[0].kind
        if kind in {"i", "u"}:
            fmt.append("%d")
        else:
            fmt.append("%.6f")
    np.savetxt(out_path, matrix, delimiter=",", header=header, comments="", fmt=fmt)


def main() -> None:
    parser = argparse.ArgumentParser(description="Генератор CSV-данных для варианта лабораторной")
    parser.add_argument("-variant", type=int, required=True, help="Номер варианта")
    parser.add_argument("-seed", type=int, required=True, help="Дата рождения в формате YYYYMMDD")
    parser.add_argument("-output", default="data.csv", help="Имя выходного CSV")
    args = parser.parse_args()

    variant = load_variant(args.variant)
    data = generate_data(variant=variant, variant_id=args.variant, seed=args.seed)
    save_csv(data, args.output)
    print(f"Сгенерирован файл: {args.output}")
    print(f"Вариант: {args.variant} | Строк: {data.shape[0]} | Полей: {len(data.dtype.names)}")


if __name__ == "__main__":
    main()
