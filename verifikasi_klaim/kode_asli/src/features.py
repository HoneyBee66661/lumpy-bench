"""5.4 Fitur Random Forest — WAJIB leakage-safe.

Aturan yang dipakai (spec v2 5.4):
- `lag_1`, `lag_7`, `lag_14`, `lag_28`  = demand hari sebelumnya (t-L).
- `rolling_mean_7`, `rolling_mean_28`, `rolling_std_28` dihitung dari `demand.shift(1)`
  lebih dulu, jadi hari t hanya memakai hari <= t-1.
- Kalender: `dow`, `month`, `is_weekend`, `is_event` (event_name_1/2 tidak kosong),
  `snap` (kolom snap sudah spesifik state SKU itu, dari hasil reshape M1).
- Target: `demand` hari t.

Fitur identitas (tambahan v2 untuk model GLOBAL, Bagian 5.4):
- `sku_code`  : label encoding integer dari `sku_id` ("FOODS_1_128__CA_4").
- `cat_code`, `dept_code`, `store_code`, `state_code`: label encoding atribut M5 yang
  diturunkan langsung dari kolom id (item_id/store_id), lihat `sku_metadata()`.
- Semua integer code dibangun deterministik dari nilai unik TERURUT, sehingga urutan code
  tidak bergantung urutan baris/pembacaan file. Pemetaan disimpan ke
  `results/tables/rf_identity_mapping.csv` supaya arti tiap angka bisa ditelusuri.
- Random Forest berbasis split pohon, jadi code integer dipakai apa adanya (tanpa one-hot),
  sesuai spec.

Konsekuensi yang dipakai di M3: fitur hari t TIDAK pernah menyentuh demand hari t, jadi
prediksi periode uji sah dilakukan satu langkah ke depan memakai actual yang sudah
terealisasi (walk-forward, spec 5.5) tanpa melatih ulang model tiap hari.

`tests/test_features.py` membuktikan dua hal: rumus tiap kolom benar, dan mengubah demand
hari t tidak mengubah satu pun fitur hari t (uji mutasi = deteksi leakage langsung).
"""
from __future__ import annotations

import pandas as pd

LAGS = (1, 7, 14, 28)
ROLL_MEAN_WINDOWS = (7, 28)
ROLL_STD_WINDOW = 28

FEATURE_COLS = [
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_std_28",
    "dow",
    "month",
    "is_weekend",
    "is_event",
    "snap",
]

# Fitur identitas (spec v2 5.4): nama kolom hasil encoding -> kolom sumber di dataframe.
IDENTITY_SOURCE = {
    "sku_code": "sku_id",
    "cat_code": "cat_id",
    "dept_code": "dept_id",
    "store_code": "store_id",
    "state_code": "state_id",
}
IDENTITY_COLS = list(IDENTITY_SOURCE)
GLOBAL_FEATURE_COLS = FEATURE_COLS + IDENTITY_COLS

TARGET_COL = "target"
MIN_HISTORY = max(max(LAGS), max(ROLL_MEAN_WINDOWS), ROLL_STD_WINDOW)


def sku_metadata(sku_id: str) -> dict:
    """Turunkan atribut M5 dari id gabungan `ITEM__STORE`, tanpa file metadata terpisah.

    Contoh: "FOODS_1_128__CA_4" -> item_id FOODS_1_128, cat_id FOODS, dept_id FOODS_1,
    store_id CA_4, state_id CA. Struktur id M5 memang memuat hierarki kategori di dalam
    item_id dan state di dalam store_id.
    """
    item, store = str(sku_id).split("__")
    item_parts = item.split("_")
    if len(item_parts) < 3:
        raise ValueError(f"item_id tidak sesuai pola M5: {item!r}")
    return {
        "sku_id": str(sku_id),
        "item_id": item,
        "cat_id": item_parts[0],
        "dept_id": "_".join(item_parts[:2]),
        "store_id": store,
        "state_id": store.split("_")[0],
    }


def build_features(daily: pd.DataFrame, sku_id: str | None = None) -> pd.DataFrame:
    """Bangun fitur untuk SATU SKU.

    daily: kolom wajib `day`, `date`, `demand`, `snap`; opsional `event_name_1`,
    `event_name_2` (kalau tidak ada, `is_event` = 0). Baris tidak dibuang di sini —
    baris hangat (28 hari pertama) dibiarkan NaN supaya pemanggil yang memutuskan.

    Kalau `sku_id` diberikan, kolom identitas mentah (`sku_id`, `cat_id`, `dept_id`,
    `store_id`, `state_id`) ikut ditambahkan untuk dipakai `encode_identity()`.
    """
    need = {"day", "date", "demand", "snap"}
    missing = need - set(daily.columns)
    if missing:
        raise ValueError(f"kolom wajib tidak ada: {sorted(missing)}")

    df = daily.sort_values("day").reset_index(drop=True).copy()
    demand = df["demand"].astype("float64")

    for lag in LAGS:
        df[f"lag_{lag}"] = demand.shift(lag)

    shifted = demand.shift(1)  # spec 5.4: shift dulu, baru rolling
    for window in ROLL_MEAN_WINDOWS:
        df[f"rolling_mean_{window}"] = shifted.rolling(window, min_periods=window).mean()
    df[f"rolling_std_{ROLL_STD_WINDOW}"] = shifted.rolling(ROLL_STD_WINDOW, min_periods=ROLL_STD_WINDOW).std(ddof=1)

    date = pd.to_datetime(df["date"])
    df["dow"] = date.dt.dayofweek.astype("int8")
    df["month"] = date.dt.month.astype("int8")
    df["is_weekend"] = (date.dt.dayofweek >= 5).astype("int8")
    if {"event_name_1", "event_name_2"} <= set(df.columns):
        df["is_event"] = (df["event_name_1"].notna() | df["event_name_2"].notna()).astype("int8")
    else:
        df["is_event"] = 0
        df["is_event"] = df["is_event"].astype("int8")
    df["snap"] = df["snap"].astype("int8")

    if sku_id is not None:
        for key, value in sku_metadata(sku_id).items():
            df[key] = value

    df[TARGET_COL] = demand
    return df


def identity_mapping(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Pemetaan deterministik nilai unik (terurut) -> integer untuk tiap fitur identitas."""
    mapping: dict[str, dict[str, int]] = {}
    for code_col, src_col in IDENTITY_SOURCE.items():
        if src_col not in df.columns:
            raise ValueError(f"kolom identitas {src_col!r} tidak ada — panggil build_features(..., sku_id=...)")
        values = sorted(str(v) for v in df[src_col].unique())
        mapping[code_col] = {v: i for i, v in enumerate(values)}
    return mapping


def encode_identity(
    df: pd.DataFrame, mapping: dict[str, dict[str, int]] | None = None
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Label-encode fitur identitas menjadi kolom `*_code` (int32).

    Pemetaan dibuat dari nilai unik terurut kalau tidak diberikan, jadi hasilnya sama
    walaupun urutan baris berbeda.
    """
    out = df.copy()
    if mapping is None:
        mapping = identity_mapping(df)
    for code_col, src_col in IDENTITY_SOURCE.items():
        series = out[src_col].astype(str)
        unknown = sorted(set(series.unique()) - set(mapping[code_col]))
        if unknown:
            raise ValueError(f"nilai {src_col!r} di luar pemetaan: {unknown[:5]}")
        out[code_col] = series.map(mapping[code_col]).astype("int32")
    return out, mapping


def feature_matrix(feat: pd.DataFrame, feature_cols: list[str] | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Ambil X, y untuk baris yang fiturnya lengkap (di luar warm-up)."""
    cols = list(FEATURE_COLS if feature_cols is None else feature_cols)
    full = feat.dropna(subset=[c for c in cols if c in FEATURE_COLS])
    return full[cols].reset_index(drop=True), full[TARGET_COL].reset_index(drop=True)


def leakage_report(feat: pd.DataFrame, feature_cols: list[str] | None = None) -> pd.DataFrame:
    """|korelasi| tiap kolom fitur terhadap target — jaring kasar untuk leakage.

    Korelasi ~1,0 berarti fitur memuat informasi target hari yang sama (mis. demand hari t).
    """
    cols = list(FEATURE_COLS if feature_cols is None else feature_cols)
    corr = feat[cols + [TARGET_COL]].corr(numeric_only=True)[TARGET_COL].drop(TARGET_COL)
    out = corr.abs().rename("abs_corr_with_target").to_frame()
    out["feature"] = out.index
    return out[["feature", "abs_corr_with_target"]].sort_values(
        "abs_corr_with_target", ascending=False
    ).reset_index(drop=True)
