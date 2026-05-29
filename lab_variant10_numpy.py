"""
Лабораторная работа (вариант 10: Спортивная аналитика).
Запуск:
  python lab_variant10_numpy.py --input data.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main(input_csv: str) -> None:
    # 1) Загрузка и подготовка типов
    raw = np.genfromtxt(input_csv, delimiter=",", names=True, dtype=None, encoding=None)
    dtype = np.dtype(
        [
            ("ts", np.int32),
            ("athlete_id", np.int16),
            ("dist", np.float32),
            ("pace", np.float32),
            ("cal", np.float32),
            ("zone", np.uint8),
        ]
    )
    data = np.empty(raw.shape[0], dtype=dtype)
    for name in data.dtype.names:
        data[name] = raw[name].astype(data.dtype[name], copy=False)

    n_rows = data.shape[0]
    bytes_used = data.nbytes
    mb_used = bytes_used / (1024.0**2)
    print(f"Строк: {n_rows}")
    print(f"Память: {bytes_used} байт ({mb_used:.3f} Мбайт)")

    numeric_fields = ["dist", "pace", "cal"]
    nan_counts = {f: int(np.isnan(data[f]).sum()) for f in numeric_fields}
    inf_counts = {f: int(np.isinf(data[f]).sum()) for f in numeric_fields}
    total_bad = sum(nan_counts.values()) + sum(inf_counts.values())
    bad_share = total_bad / (n_rows * len(numeric_fields))
    print(f"NaN по полям: {nan_counts}")
    print(f"Inf по полям: {inf_counts}")
    if bad_share > 0.03:
        print("ПРЕДУПРЕЖДЕНИЕ: доля NaN/Inf > 3%")

    np.save("stage1_structured.npy", data)

    # 2) Векторизованная фильтрация и очистка
    anomaly_mask = (
        (data["dist"] < 0)
        | (data["pace"] < 0)
        | np.isnan(data["dist"])
        | np.isnan(data["pace"])
        | np.isnan(data["cal"])
        | np.isinf(data["dist"])
        | np.isinf(data["pace"])
        | np.isinf(data["cal"])
    )
    an_count = int(anomaly_mask.sum())
    print(f"Аномалий: {an_count} ({an_count / n_rows:.2%})")

    cleaned = data[~anomaly_mask].copy()
    cleaned["cal"] = np.where(cleaned["cal"] < 0, 0.0, cleaned["cal"])

    q10, q90 = np.nanpercentile(cleaned["pace"], [10, 90])
    cleaned["pace"] = np.clip(cleaned["pace"], q10, q90)

    # 3) Группировка + нормализация
    group_ids, group_counts = np.unique(cleaned["athlete_id"], return_counts=True)
    print(f"Групп: {group_ids.size}")
    print(f"Размеры групп: min={group_counts.min()}, max={group_counts.max()}")

    mean_dist = np.empty(group_ids.size, dtype=np.float32)
    std_dist = np.empty(group_ids.size, dtype=np.float32)
    mean_pace = np.empty(group_ids.size, dtype=np.float32)
    min_pace = np.empty(group_ids.size, dtype=np.float32)

    for i, gid in enumerate(group_ids):
        gm = cleaned["athlete_id"] == gid
        g_dist = cleaned["dist"][gm]
        g_pace = cleaned["pace"][gm]
        mean_dist[i] = np.nanmean(g_dist)
        std_dist[i] = np.nanstd(g_dist)
        mean_pace[i] = np.nanmean(g_pace)
        min_pace[i] = np.nanmin(g_pace)

    z_dist = np.empty(cleaned.shape[0], dtype=np.float32)
    for i, gid in enumerate(group_ids):
        gm = cleaned["athlete_id"] == gid
        z_dist[gm] = (cleaned["dist"][gm] - mean_dist[i]) / (std_dist[i] + 1e-8)

    norm_dtype = cleaned.dtype.descr + [("z_dist", np.float32)]
    norm_data = np.empty(cleaned.shape[0], dtype=norm_dtype)
    for n in cleaned.dtype.names:
        norm_data[n] = cleaned[n]
    norm_data["z_dist"] = z_dist
    np.save("stage3_normalized.npy", norm_data)

    # 4) Скользящее окно и разница
    ratio = norm_data["cal"] / np.maximum(norm_data["dist"], 1e-8)
    k = 30
    csum = np.cumsum(np.insert(ratio, 0, 0.0))
    mov = (csum[k:] - csum[:-k]) / k
    mov_full = np.pad(mov, (k - 1, 0), mode="edge")

    pace_diff = np.diff(norm_data["pace"])
    pace_diff_full = np.insert(pace_diff, 0, 0.0).astype(np.float32)

    win_dtype = norm_data.dtype.descr + [("cal_per_km_ma30", np.float32), ("pace_diff", np.float32)]
    win_data = np.empty(norm_data.shape[0], dtype=win_dtype)
    for n in norm_data.dtype.names:
        win_data[n] = norm_data[n]
    win_data["cal_per_km_ma30"] = mov_full.astype(np.float32)
    win_data["pace_diff"] = pace_diff_full

    # 5) Feature engineering
    speed_kmh = win_data["dist"] / np.maximum(win_data["pace"] / 60.0, 1e-8)
    cal_per_min = win_data["cal"] / np.maximum(win_data["pace"], 1e-8)

    speed_kmh = np.where(np.isfinite(speed_kmh), speed_kmh, 0.0).astype(np.float32)
    cal_per_min = np.where(np.isfinite(cal_per_min), cal_per_min, np.nanmedian(cal_per_min)).astype(np.float32)

    feat_dtype = win_data.dtype.descr + [("speed_kmh", np.float32), ("cal_per_min", np.float32)]
    feat_data = np.empty(win_data.shape[0], dtype=feat_dtype)
    for n in win_data.dtype.names:
        feat_data[n] = win_data[n]
    feat_data["speed_kmh"] = speed_kmh
    feat_data["cal_per_min"] = cal_per_min

    # 6) Условная агрегация по группам
    # Условие: тренировка в высокой зоне пульса и выше медианной дистанции.
    global_dist_median = np.nanmedian(feat_data["dist"])
    cond_mask = (feat_data["zone"] >= 4) & (feat_data["dist"] > global_dist_median) & (feat_data["pace"] < np.nanmedian(feat_data["pace"]))

    conditional_stats = np.empty((group_ids.size, 3), dtype=np.float64)
    conditional_stats_p90 = np.empty((group_ids.size, 4), dtype=np.float64)
    for i, gid in enumerate(group_ids):
        gm = (feat_data["athlete_id"] == gid) & cond_mask
        vals = feat_data["cal_per_min"][gm]
        if vals.size == 0:
            conditional_stats[i] = [gid, np.nan, np.nan]
            conditional_stats_p90[i] = [gid, np.nan, np.nan, np.nan]
        else:
            m = np.nanmean(vals)
            med = np.nanmedian(vals)
            p90 = np.nanpercentile(vals, 90)
            conditional_stats[i] = [gid, m, med]
            conditional_stats_p90[i] = [gid, m, med, p90]

    np.savetxt(
        "conditional_stats.csv",
        conditional_stats,
        delimiter=",",
        header="group_id,mean_conditional,median_conditional",
        comments="",
        fmt=["%d", "%.6f", "%.6f"],
    )
    np.savetxt(
        "conditional_stats_with_p90.csv",
        conditional_stats_p90,
        delimiter=",",
        header="group_id,mean_conditional,median_conditional,p90_conditional",
        comments="",
        fmt=["%d", "%.6f", "%.6f", "%.6f"],
    )

    # 7) Лаговые признаки и анализ сдвигов
    prev_pace = np.roll(feat_data["pace"], 1)
    prev_pace[0] = feat_data["pace"][0]
    pace_lag_diff = feat_data["pace"] - prev_pace

    lag_dtype = feat_data.dtype.descr + [("pace_lag1", np.float32), ("pace_delta_lag1", np.float32)]
    lag_data = np.empty(feat_data.shape[0], dtype=lag_dtype)
    for n in feat_data.dtype.names:
        lag_data[n] = feat_data[n]
    lag_data["pace_lag1"] = prev_pace.astype(np.float32)
    lag_data["pace_delta_lag1"] = pace_lag_diff.astype(np.float32)

    grew = np.mean(pace_lag_diff > 0)
    fell = np.mean(pace_lag_diff < 0)
    same = np.mean(pace_lag_diff == 0)
    print(f"Доля роста pace: {grew:.2%}, падения: {fell:.2%}, без изменений: {same:.2%}")

    signs, sign_counts = np.unique(np.sign(pace_lag_diff).astype(np.int8), return_counts=True)
    print(f"Распределение sign: {dict(zip(signs.tolist(), sign_counts.tolist()))}")

    # 8) Групповая робастная замена выбросов (IQR) по pace
    replaced_per_group = []
    replaced_mask_total = np.zeros(lag_data.shape[0], dtype=bool)
    robust_pace = lag_data["pace"].copy()

    for gid in group_ids:
        gm = lag_data["athlete_id"] == gid
        vals = robust_pace[gm]
        q1, q3 = np.nanpercentile(vals, [25, 75])
        iqr = q3 - q1
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        out = gm & ((robust_pace < lo) | (robust_pace > hi))
        med = np.nanmedian(vals)
        robust_pace[out] = med
        replaced_per_group.append((int(gid), int(out.sum())))
        replaced_mask_total |= out

    lag_data["pace"] = robust_pace
    replaced_share = replaced_mask_total.mean()
    print(f"Доля замен после IQR: {replaced_share:.2%}")
    print(f"Замены по группам (первые 10): {replaced_per_group[:10]}")

    # 9) Проверка согласованности и логической целостности
    # Логика: зона 5 не должна идти с очень медленным темпом, а зона 1 - с экстремально быстрым.
    invalid_logic = ((lag_data["zone"] == 5) & (lag_data["pace"] > 8.5)) | ((lag_data["zone"] == 1) & (lag_data["pace"] < 3.2))
    invalid_count = int(invalid_logic.sum())
    print(f"Логических нарушений: {invalid_count} ({invalid_count / lag_data.shape[0]:.2%})")

    lag_data["zone"] = np.where((lag_data["pace"] < 4.0) & (lag_data["zone"] < 3), 3, lag_data["zone"]).astype(np.uint8)
    lag_data["zone"] = np.where((lag_data["pace"] > 8.0) & (lag_data["zone"] > 3), 3, lag_data["zone"]).astype(np.uint8)

    # 10) Частотный анализ и сжатие редких категорий
    cats, cnts = np.unique(lag_data["zone"], return_counts=True)
    freq = cnts / lag_data.shape[0]
    rare_cats = cats[freq < 0.01]
    is_rare = np.isin(lag_data["zone"], rare_cats)
    lag_data["zone"] = np.where(is_rare, 0, lag_data["zone"]).astype(np.uint8)
    print(f"Редкие категории zone: {rare_cats.tolist()}")

    np.save("final_preprocessed.npy", lag_data)
    print(
        "Готово. Сохранены файлы: stage1_structured.npy, stage3_normalized.npy, "
        "conditional_stats.csv, conditional_stats_with_p90.csv, final_preprocessed.npy"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data.csv")
    args = parser.parse_args()
    if not Path(args.input).exists():
        raise FileNotFoundError(f"Не найден файл {args.input}")
    main(args.input)
