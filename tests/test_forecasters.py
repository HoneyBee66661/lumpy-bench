"""Kesetaraan implementasi Croston-SBA dan determinisme Random Forest."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lumpy_bench import forecasters


def test_croston_manual_setara_pustaka():
    if not forecasters.ADA_STATSFORECAST:
        pytest.skip("statsforecast tidak terpasang")
    rng = np.random.default_rng(5)
    y = rng.poisson(2.0, 400).astype(float)
    y[rng.random(400) < 0.4] = 0.0
    seri = pd.DataFrame({"sku_id": "X", "day": np.arange(1, 401), "demand": y})
    manual = forecasters.ramal_croston_sba(seri, np.array([401]), 8, 0.10, pakai_statsforecast=False)
    pustaka = forecasters.ramal_croston_sba(seri, np.array([401]), 8, 0.10, pakai_statsforecast=True)
    assert manual["prediksi"].iloc[0] == pytest.approx(pustaka["prediksi"].iloc[0], rel=1e-9)


def test_rf_deterministik():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(400, 6))
    y = X[:, 0] * 2 + rng.normal(size=400)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    df["target"] = y
    kolom = [c for c in df.columns if c != "target"]
    m1 = forecasters.latih_rf(df, kolom, 8, {"n_estimators": 60, "n_jobs": 1}, seed=1)
    m2 = forecasters.latih_rf(df, kolom, 8, {"n_estimators": 60, "n_jobs": 1}, seed=1)
    a = m1.predict(df[kolom].to_numpy(float))
    b = m2.predict(df[kolom].to_numpy(float))
    assert np.array_equal(a, b)
