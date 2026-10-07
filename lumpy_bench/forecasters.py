"""Dua metode yang dibandingkan: Random Forest global dan Croston-SBA.

Antarmuka seragam: keduanya meramalkan jumlah permintaan H hari ke depan per SKU per hari.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from statsforecast.models import CrostonSBA
    ADA_STATSFORECAST = True
except Exception:                                   # pragma: no cover
    ADA_STATSFORECAST = False


def latih_rf(latih: pd.DataFrame, kolom_fitur: list[str], horizon: int, params: dict, seed: int):
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(
        n_estimators=int(params.get("n_estimators", 200)),
        max_depth=params.get("max_depth", 10),
        min_samples_leaf=int(params.get("min_samples_leaf", 5)),
        random_state=seed, n_jobs=int(params.get("n_jobs", 1)),
    )
    model.fit(latih[kolom_fitur].to_numpy(float), latih["target"].to_numpy(float))
    return model


def ramal_rf(model, uji: pd.DataFrame, kolom_fitur: list[str]) -> pd.DataFrame:
    pred = model.predict(uji[kolom_fitur].to_numpy(float))
    return pd.DataFrame({"sku_id": uji["sku_id"].to_numpy(), "day": uji["day"].to_numpy(),
                         "prediksi": pred, "aktual": uji["target"].to_numpy(float)})


def ramal_croston_sba(seri_sku: pd.DataFrame, hari_uji: np.ndarray, horizon: int,
                      alpha: float = 0.10, pakai_statsforecast: bool = False) -> pd.DataFrame:
    """Croston-SBA atas permintaan harian, dikalikan horizon agar sebanding dengan target H-jumlah.

    Implementasi mengikuti statsforecast bila tersedia (pemulusan alpha = 0,10 dan faktor
    debiasing 0,95 = 1 - alpha/2); bila pustaka tidak ada, dipakai implementasi ekuivalen.
    """
    seri = seri_sku.sort_values("day")
    y = seri["demand"].to_numpy(float)
    baris = []
    for hari in hari_uji:
        pos = int(np.searchsorted(seri["day"].to_numpy(), hari, side="right"))
        hist = y[:pos]
        if pakai_statsforecast and ADA_STATSFORECAST:
            model = CrostonSBA()
            model.fit(hist)
            laju = float(model.predict(h=1)["mean"][0])
        else:                                        # ekuivalen manual
            laju = _croston_sba_manual(hist, alpha)
        baris.append({"sku_id": seri["sku_id"].iloc[0], "day": int(hari), "laju": laju})
    out = pd.DataFrame(baris)
    out["prediksi"] = out["laju"] * horizon
    return out.drop(columns=["laju"])


def _croston_sba_manual(hist: np.ndarray, alpha: float, debias: float = 0.95) -> float:
    pos = hist[hist > 0]
    if pos.size == 0:
        return 0.0
    interval = np.diff(np.flatnonzero(np.concatenate([[0], hist > 0])))
    interval = interval[interval > 0]
    z = _ses(pos, alpha)
    p = _ses(interval.astype(float), alpha)
    return debias * (z / p) if p > 0 else debias * z


def _ses(x: np.ndarray, alpha: float) -> float:
    if x.size == 0:
        return 0.0
    f = float(x[0])
    for v in x[1:]:
        f = alpha * float(v) + (1 - alpha) * f
    return alpha * float(x[-1]) + (1 - alpha) * f
