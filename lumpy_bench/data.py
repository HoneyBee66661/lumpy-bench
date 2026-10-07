"""Pemuatan data dan penentuan periode aktif.

Skema yang diharapkan: item_id, store_id, day, demand, sell_price (kolom snap, event_name_1,
cat_id, state_id opsional). Berkas M5 asli dapat dipakai setelah diubah ke bentuk panjang.
"""
from __future__ import annotations

import pandas as pd

KOLOM_WAJIB = ["item_id", "store_id", "day", "demand", "sell_price"]


def muat_long(path: str, kolom_tambahan: list[str] | None = None) -> pd.DataFrame:
    kolom = list(KOLOM_WAJIB) + [k for k in (kolom_tambahan or []) if k not in KOLOM_WAJIB]
    df = pd.read_csv(path, usecols=lambda c: c in set(kolom) | {"date", "snap", "event_name_1", "cat_id", "state_id"})
    kurang = [k for k in KOLOM_WAJIB if k not in df.columns]
    if kurang:
        raise ValueError(f"kolom wajib tidak ada: {kurang}")
    df["sku_id"] = df["item_id"].astype(str) + "__" + df["store_id"].astype(str)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["day"] = df["day"].astype(int)
    df["demand"] = df["demand"].astype(float)
    df["cat_id"] = df["cat_id"].astype(str) if "cat_id" in df.columns else "CAT"
    df["state_id"] = df["state_id"].astype(str) if "state_id" in df.columns else "STATE"
    return df.sort_values(["sku_id", "day"]).reset_index(drop=True)


def potong_periode_aktif(df: pd.DataFrame) -> pd.DataFrame:
    """Periode aktif = hari yang punya harga jual (peluncuran s.d. penghentian)."""
    return df[df["sell_price"].notna()].copy()


def muat_m5(path_csv: str, path_harga: str, path_kalender: str | None = None) -> pd.DataFrame:
    """Muat berkas M5 asli (sales_train_*.csv + sell_prices.csv) menjadi bentuk panjang.

    Catatan etika data: berkas M5 tidak boleh didistribusikan ulang. Fungsi ini hanya
    membantu pengguna yang mengunduh sendiri dari Kaggle dan menyetujui ketentuannya.
    """
    penjualan = pd.read_csv(path_csv)
    harga = pd.read_csv(path_harga)
    hari = [c for c in penjualan.columns if c.startswith("d_")]
    panjang = penjualan.melt(id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
                             value_vars=hari, var_name="d", value_name="demand")
    panjang["day"] = panjang["d"].str.replace("d_", "").astype(int)
    if path_kalender:
        kal = pd.read_csv(path_kalender)
        kal = kal.rename(columns={"d": "d"})[["d", "date", "snap", "event_name_1"]]
        panjang = panjang.merge(kal, on="d", how="left")
        panjang["date"] = pd.to_datetime(panjang["date"], errors="coerce")
    panjang = panjang.merge(harga, on=["store_id", "item_id", "d"], how="left")
    return muat_long_frame(panjang)


def muat_long_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Rapikan kerangka panjang yang sudah jadi (tanpa membaca berkas)."""
    df = df.copy()
    df["sku_id"] = df["item_id"].astype(str) + "__" + df["store_id"].astype(str)
    df["demand"] = df["demand"].astype(float)
    df["day"] = df["day"].astype(int)
    for k, v in (("cat_id", "CAT"), ("state_id", "STATE")):
        df[k] = df[k].astype(str) if k in df.columns else v
    df["snap"] = df["snap"].fillna(0).astype(int) if "snap" in df.columns else 0
    return df.sort_values(["sku_id", "day"]).reset_index(drop=True)
