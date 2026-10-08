"""features_v3.py — Fase 2: fitur untuk target DIRECT MULTI-HORIZON periode proteksi.

Keputusan desain (menjawab temuan audit A2 'look-ahead'):
  * Target model = JUMLAH demand selama periode proteksi: sum demand[t+1 .. t+H], dengan H = R + L
    (default 8 hari). Model global dilatih langsung pada besaran yang dipakai kebijakan, jadi tidak
    perlu menjumlahkan forecast satu-langkah (yang di v2 membocorkan 6 hari demand masa depan).
  * Fitur hanya memakai informasi <= t: lag_1 = demand hari t sendiri (keputusan order dibuat di
    AKHIR hari t setelah demand teramati), rolling statistik sampai t.
  * Fitur kalender untuk jendela t+1..t+H (jumlah event, jumlah hari SNAP, jumlah akhir pekan)
    memang "masa depan", tetapi kalender M5 bersifat eksogen dan diketahui di muka — bukan leakage
    demand. Ini dinyatakan eksplisit di Metode.
  * Fitur identitas SKU (kategori/departemen/toko/state/sku) seperti v2, supaya model global bisa
    cross-learning antar SKU.

Uji leakage (tests/test_features_v3.py): mengubah demand SETELAH hari t tidak mengubah satu pun fitur
hari t (target memang memakai masa depan — target bukan fitur).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))          # pakai ulang encoder identitas yang sudah teruji (v2)
from features import (  # noqa: E402
    IDENTITY_COLS,
    IDENTITY_SOURCE,
    encode_identity,
    identity_mapping,
    sku_metadata,
)

LAGS = (1, 7, 14, 28)
ROLL_MEAN_WINDOWS = (7, 28)
ROLL_STD_WINDOW = 28
TARGET_COL = "target_proteksi"          # sum demand[t+1 .. t+H]

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
    "is_event_hari_ini",
    "snap_hari_ini",
    # kalender jendela proteksi (eksogen, diketahui di muka)
    "event_h",
    "snap_h",
    "weekend_h",
]
GLOBAL_FEATURE_COLS = FEATURE_COLS + list(IDENTITY_COLS)
MIN_HISTORY = max(max(LAGS), max(ROLL_MEAN_WINDOWS), ROLL_STD_WINDOW)


def _fwd_window_sum(series: pd.Series, h: int) -> pd.Series:
    """Jumlah series pada jendela t+1..t+h (None/NaN untuk h baris terakhir)."""
    return series.rolling(h, min_periods=h).sum().shift(-h)


def build_features_v3(daily: pd.DataFrame, sku_id: str, horizon: int = 8) -> pd.DataFrame:
    """Bangun fitur + target proteksi untuk SATU SKU.

    daily wajib punya kolom: day, date, demand, snap; opsional event_name_1/2.
    """
    need = {"day", "date", "demand", "snap"}
    missing = need - set(daily.columns)
    if missing:
        raise ValueError(f"kolom wajib tidak ada: {sorted(missing)}")

    df = daily.sort_values("day").reset_index(drop=True).copy()
    demand = df["demand"].astype("float64")

    for lag in LAGS:
        df[f"lag_{lag}"] = demand.shift(lag - 1)          # lag_1 = demand hari t

    for w in ROLL_MEAN_WINDOWS:
        df[f"rolling_mean_{w}"] = demand.rolling(w, min_periods=w).mean()
    df[f"rolling_std_{ROLL_STD_WINDOW}"] = demand.rolling(ROLL_STD_WINDOW, min_periods=ROLL_STD_WINDOW).std(ddof=1)

    date = pd.to_datetime(df["date"])
    df["dow"] = date.dt.dayofweek.astype("int8")
    df["month"] = date.dt.month.astype("int8")
    df["is_weekend"] = (date.dt.dayofweek >= 5).astype("int8")
    if {"event_name_1", "event_name_2"} <= set(df.columns):
        is_event = (df["event_name_1"].notna() | df["event_name_2"].notna()).astype("int8")
    else:
        is_event = pd.Series(np.zeros(len(df), dtype="int8"), index=df.index)
    df["is_event_hari_ini"] = is_event.astype("int8")
    df["snap_hari_ini"] = df["snap"].astype("int8")

    # kalender jendela proteksi t+1..t+H (eksogen)
    df["event_h"] = _fwd_window_sum(is_event.astype("float64"), horizon).fillna(np.nan)
    df["snap_h"] = _fwd_window_sum(df["snap"].astype("float64"), horizon)
    df["weekend_h"] = _fwd_window_sum(df["is_weekend"].astype("float64"), horizon)

    # target: jumlah demand t+1..t+H
    df[TARGET_COL] = _fwd_window_sum(demand, horizon)

    for key, value in sku_metadata(sku_id).items():
        df[key] = value
    return df


def features_and_target(
    df: pd.DataFrame, horizon: int = 8, butuh_target: bool = True
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Ambil X (baris dengan fitur lengkap) dan y (target) bila diminta."""
    mask = df[GLOBAL_FEATURE_COLS].notna().all(axis=1)
    if butuh_target:
        mask &= df[TARGET_COL].notna()
    sub = df.loc[mask]
    y = sub[TARGET_COL] if butuh_target else None
    return sub[GLOBAL_FEATURE_COLS].reset_index(drop=True), (y.reset_index(drop=True) if y is not None else None)


def encode_pooled(frames: list[pd.DataFrame], mapping: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Gabungkan fitur beberapa SKU lalu label-encode fitur identitas + kolom `*_code`.

    Pemetaan dibuat dari gabungan (konsisten antar SKU); kalau `mapping` diberikan, dipakai apa adanya
    sehingga SKU baru tidak mengubah kode SKU lama.
    """
    pooled = pd.concat(frames, ignore_index=True)
    pooled, used = encode_identity(pooled, mapping=mapping)
    return pooled, used


def skema_versi() -> dict:
    return {
        "horizon": "diisi pemanggil (default = R+L = 8)",
        "fitur": GLOBAL_FEATURE_COLS,
        "target": TARGET_COL,
        "encoder_identitas": "dipakai ulang dari src/features.py (IDENTITY_SOURCE)",
        "leakage": "fitur <= t; kalender jendela proteksi eksogen; target punya masa depan (label)",
        "catatan": "lag_1 = demand hari t (keputusan order di akhir hari t)",
    }
