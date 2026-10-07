"""Pengambilan sampel berstrata dari populasi lumpy."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _kuartil(s: pd.Series) -> pd.Series:
    return pd.qcut(s.rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"]).astype(str)


def ambil_sampel(
    stat: pd.DataFrame,
    meta: pd.DataFrame,
    n_sku: int = 40,
    seed: int = 20261007,
    kolom_strata: tuple[str, ...] = ("cat_id", "state_id"),
) -> pd.DataFrame:
    """Ambil sampel berstrata: kuartil ADI x kuartil CV2 x kategori x negara bagian."""
    populasi = stat[stat["lumpy"]].merge(meta, on="sku_id", how="left")
    if populasi.empty:
        raise ValueError("populasi lumpy kosong; periksa ambang ADI/CV2 atau data masukan")
    populasi = populasi.copy()
    populasi["adi_q"] = _kuartil(populasi["adi"])
    populasi["cv2_q"] = _kuartil(populasi["cv2"])
    strata = ["adi_q", "cv2_q"] + [k for k in kolom_strata if k in populasi.columns]
    populasi["stratum"] = populasi[strata].astype(str).agg("-".join, axis=1)
    rng = np.random.default_rng(seed)
    bagian = []
    for _, g in populasi.groupby("stratum", observed=True):
        ambil = min(len(g), max(1, int(round(n_sku * len(g) / len(populasi)))))
        idx = rng.choice(g.index.to_numpy(), size=ambil, replace=False)
        bagian.append(g.loc[idx])
    sampel = pd.concat(bagian, ignore_index=True) if bagian else populasi
    if len(sampel) > n_sku:                      # pangkas ke jumlah yang diminta
        sampel = sampel.sort_values(["adi_q", "cv2_q", "sku_id"]).head(n_sku)
    return sampel.sort_values("sku_id").reset_index(drop=True)
