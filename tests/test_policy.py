"""Invarian simulator order-up-to dan penyetaraan tingkat layanan."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lumpy_bench import policy


def _data(n=60, seed=7):
    rng = np.random.default_rng(seed)
    demand = rng.poisson(3.0, n).astype(float)
    demand[rng.random(n) < 0.4] = 0.0
    prediksi = np.full(n, 9.0)
    galat = rng.normal(0, 2.0, 200)
    return demand, prediksi, galat


def test_mass_balance_dan_tidak_ada_stok_negatif():
    demand, prediksi, galat = _data()
    ss = policy.kuantil_galat(galat, [0.95])[0.95]
    hasil = policy.simulasi_order_up_to(demand, prediksi, ss, warmup=0)
    # neraca massa: permintaan = terlayani + tunggakan akhir
    assert hasil["served_total"] + hasil["backlog_akhir"] == pytest.approx(hasil["demand_total"])
    # tidak ada stok maupun kekurangan yang bernilai negatif
    assert (hasil["jejak_on_hand"] >= -1e-9).all()
    assert (hasil["jejak_unmet"] >= -1e-9).all()


def test_fill_rate_tidak_turun_ketika_ss_dinaikkan():
    demand, prediksi, galat = _data(seed=11)
    ss = policy.kuantil_galat(galat, [0.50, 0.80, 0.95])
    fill = [policy.simulasi_order_up_to(demand, prediksi, ss[t], warmup=0)["fill_rate"]
            for t in (0.50, 0.80, 0.95)]
    assert fill[0] <= fill[-1] + 1e-9


def test_penyetaraan_interpolasi_dan_batas_cakupan():
    sweep = pd.DataFrame({
        "tau": [0.5, 0.9], "fill_rate": [0.80, 0.97],
        "unmet_total": [10.0, 2.0], "on_hand_total": [100.0, 300.0]})
    sweep = policy.biaya(sweep, 5.0, 1.0)             # biaya: 150 dan 310
    hasil = policy.setarakan_layanan(sweep, [0.90, 0.995])
    baris90 = hasil[hasil["fill_target"] == 0.90].iloc[0]
    baris99 = hasil[hasil["fill_target"] == 0.995].iloc[0]
    harap = 150.0 + (0.90 - 0.80) / (0.97 - 0.80) * (310.0 - 150.0)
    assert baris90["tercakup"] and baris90["cost_at_target"] == pytest.approx(harap, rel=1e-9)
    assert not baris99["tercakup"] and np.isnan(baris99["cost_at_target"])
