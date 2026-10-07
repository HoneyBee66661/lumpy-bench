"""Fitur untuk peramalan langsung (direct multi-horizon) jendela proteksi.

Fitur hanya memakai informasi sampai hari t (lag_1 = permintaan hari t, karena keputusan
pemesanan dibuat pada akhir hari t), sedangkan target memakai hari t+1..t+H.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LAG_DEFAULT = (1, 7, 14, 28)
ROLLING_DEFAULT = (7, 28)
ROLLING_STD_DEFAULT = 28


def daftar_fitur(lag=LAG_DEFAULT, rolling=ROLLING_DEFAULT,
                 rolling_std: int = ROLLING_STD_DEFAULT) -> list[str]:
    """Nama kolom fitur untuk konfigurasi jendela tertentu (urutan kolom = matriks fitur)."""
    return (
        [f"lag_{int(l)}" for l in lag]
        + [f"rolling_mean_{int(w)}" for w in rolling]
        + [f"rolling_std_{int(rolling_std)}"]
        + ["is_event_hari_ini", "snap_hari_ini", "event_h", "snap_h", "hari_minggu"]
    )


FITUR = daftar_fitur()          # daftar baku (kompatibilitas)


def dari_config(blok: dict | None) -> dict:
    """Ubah blok `features` di config.yaml menjadi argumen siap pakai."""
    b = blok or {}
    return {
        "lag": tuple(int(x) for x in b.get("lag", LAG_DEFAULT)),
        "rolling": tuple(int(x) for x in b.get("rolling", ROLLING_DEFAULT)),
        "rolling_std": int(b.get("rolling_std", ROLLING_STD_DEFAULT)),
    }


def fitur_sku(seri: pd.DataFrame, horizon: int = 8, lag=LAG_DEFAULT, rolling=ROLLING_DEFAULT,
              rolling_std: int = ROLLING_STD_DEFAULT) -> pd.DataFrame:
    """Bangun fitur + target H-jumlah untuk satu SKU. seri wajib punya kolom day, demand, snap."""
    df = seri.sort_values("day").reset_index(drop=True).copy()
    demand = df["demand"].astype(float)
    for l in lag:
        df[f"lag_{int(l)}"] = demand.shift(int(l) - 1)
    for w in rolling:
        df[f"rolling_mean_{int(w)}"] = demand.rolling(int(w), min_periods=int(w)).mean()
    df[f"rolling_std_{int(rolling_std)}"] = demand.rolling(int(rolling_std),
                                                            min_periods=int(rolling_std)).std(ddof=1)
    event = df["event_name_1"].notna().astype(int) if "event_name_1" in df.columns else pd.Series(0, index=df.index)
    df["is_event_hari_ini"] = event
    df["snap_hari_ini"] = df["snap"].astype(int) if "snap" in df.columns else 0
    df["hari_minggu"] = pd.to_datetime(df["date"]).dt.dayofweek if "date" in df.columns else df["day"] % 7
    # agregat jendela t+1..t+H
    df["event_h"] = event[::-1].rolling(horizon, min_periods=horizon).sum()[::-1].shift(-1)
    df["snap_h"] = df["snap_hari_ini"][::-1].rolling(horizon, min_periods=horizon).sum()[::-1].shift(-1)
    # target: jumlah permintaan t+1..t+H
    df["target"] = demand[::-1].rolling(horizon, min_periods=horizon).sum()[::-1].shift(-1)
    return df


def bangun_matriks(panjang: pd.DataFrame, sku_ids: list[str], horizon: int = 8,
                   blok_fitur: dict | None = None) -> pd.DataFrame:
    """Matriks fitur untuk SKU terpilih; blok_fitur = blok `features` dari config.yaml."""
    p = dari_config(blok_fitur)
    bagian = [fitur_sku(g, horizon, p["lag"], p["rolling"], p["rolling_std"]).assign(sku_id=s)
              for s, g in panjang[panjang["sku_id"].isin(sku_ids)].groupby("sku_id", observed=True)]
    return pd.concat(bagian, ignore_index=True)


def potong_latih_uji(matriks: pd.DataFrame, batas_latih: int, batas_uji_akhir: int,
                     kolom_fitur: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pisahkan baris latih (day <= batas_latih) dan uji (day <= batas_uji_akhir)."""
    kolom = kolom_fitur or FITUR
    siap = matriks.dropna(subset=kolom + ["target"])
    latih = siap[siap["day"] <= batas_latih]
    uji = siap[siap["day"] <= batas_uji_akhir]
    return latih, uji
