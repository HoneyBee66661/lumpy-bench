"""Pembuat data contoh sintetis berskema sama dengan data ritel (tanpa data M5).

Tujuannya: siapa pun dapat menjalankan seluruh alur tanpa mengunduh data kompetisi, lalu
mengganti berkasnya dengan data asli bila diinginkan.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def buat_data_contoh(
    n_sku: int = 150,
    n_hari: int = 900,
    n_toko: int = 4,
    n_kategori: int = 3,
    seed: int = 20261007,
    mulai: str = "2023-01-01",
) -> pd.DataFrame:
    """Buat deret harian item-toko dengan campuran pola: lancar, intermiten, dan lumpy.

    Kolom keluaran: item_id, store_id, day, date, demand, sell_price, snap, event_name_1,
    cat_id, state_id.
    """
    rng = np.random.default_rng(seed)
    tanggal = pd.date_range(mulai, periods=n_hari, freq="D")
    baris = []
    for i in range(n_sku):
        toko = f"STORE_{i % n_toko + 1}"
        kategori = f"CAT_{i % n_kategori + 1}"
        sku = f"ITEM_{i:03d}__{toko}"
        # profil permintaan beragam: lancar, intermiten, dan lumpy dengan berbagai tingkat kesulitan
        r = i % 10
        if r < 3:
            p, skala = 0.85, 4.0          # lancar
        elif r < 6:
            p, skala = 0.45, 4.0          # intermiten ringan
        elif r < 8:
            p, skala = 0.30, 3.0          # lumpy sedang (masih mungkin mencapai fill tinggi)
        else:
            p, skala = 0.14, 6.0          # lumpy berat
        # permintaan
        muncul = rng.random(n_hari) < p
        # musiman mingguan
        mingguan = 1.0 + 0.25 * np.sin(2 * np.pi * np.arange(n_hari) / 7)
        demand = np.where(muncul, rng.negative_binomial(3, 3 / (3 + skala * mingguan)), 0).astype(float)
        # event dan SNAP
        event = np.array([f"EVENT_{1 + (k // 37) % 4}" if (k % 97 == 0) else None for k in range(n_hari)], dtype=object)
        snap = (rng.random(n_hari) < 0.18).astype(int)
        demand = np.where(event != None, demand * rng.integers(2, 5, n_hari), demand)  # noqa: E711
        # harga jual: periode aktif dibatasi (peluncuran dan penghentian)
        aktif_mulai = int(rng.integers(0, 60))
        aktif_selesai = n_hari - int(rng.integers(0, 90))
        harga = np.full(n_hari, np.nan)
        harga[aktif_mulai:aktif_selesai] = np.round(rng.uniform(1.5, 9.0), 2)
        demand = np.where(np.isnan(harga), 0.0, demand)
        baris.append(pd.DataFrame({
            "item_id": f"ITEM_{i:03d}", "store_id": toko, "sku_id": sku,
            "day": np.arange(1, n_hari + 1), "date": tanggal,
            "demand": demand, "sell_price": harga, "snap": snap, "event_name_1": event,
            "cat_id": kategori, "state_id": f"STATE_{i % 2 + 1}",
        }))
    return pd.concat(baris, ignore_index=True)


def simpan(path: str, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False, date_format="%Y-%m-%d")
