"""Klasifikasi pola permintaan menurut ADI dan CV2 (Syntetos dkk., 2005)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def statistik_sku(aktif: pd.DataFrame, min_nonzero: int = 2) -> pd.DataFrame:
    """ADI = jumlah hari periode aktif / jumlah hari berpermintaan; CV2 = (sd/mean)^2 ukuran."""
    g = aktif.groupby("sku_id", observed=True)
    out = pd.DataFrame({
        "n_hari_aktif": g["demand"].size(),
        "n_nonzero": g["demand"].apply(lambda s: int((s > 0).sum())),
        "sum_pos": g["demand"].apply(lambda s: float(s[s > 0].sum())),
        "sumsq_pos": g["demand"].apply(lambda s: float((s[s > 0] ** 2).sum())),
    })
    out = out[out["n_nonzero"] >= int(min_nonzero)].copy()
    out["mean_pos"] = out["sum_pos"] / out["n_nonzero"]
    var = (out["sumsq_pos"] - out["n_nonzero"] * out["mean_pos"] ** 2) / (out["n_nonzero"] - 1)
    out["cv2"] = (np.sqrt(var.clip(lower=0)) / out["mean_pos"]) ** 2
    out["adi"] = out["n_hari_aktif"] / out["n_nonzero"]
    return out.reset_index()


def tandai_lumpy(stat: pd.DataFrame, adi_min: float = 1.32, cv2_min: float = 0.49) -> pd.DataFrame:
    stat = stat.copy()
    stat["lumpy"] = (stat["adi"] > adi_min) & (stat["cv2"] > cv2_min)
    return stat


def klasifikasi(aktif: pd.DataFrame, adi_min: float = 1.32, cv2_min: float = 0.49,
                min_nonzero: int = 2) -> pd.DataFrame:
    return tandai_lumpy(statistik_sku(aktif, min_nonzero), adi_min, cv2_min)
