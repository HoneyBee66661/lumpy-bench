"""Simulasi kebijakan order-up-to periodik dan penyetaraan tingkat layanan.

Model kekurangan stok: backorder (permintaan yang belum terlayani ditunggak dan dipenuhi saat
barang datang). Persediaan pengaman diambil dari kuantil empiris galat validasi tiap SKU,
bukan dari rumus normal atau dari rasio kritis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def simulasi_order_up_to(
    demand: np.ndarray,
    prediksi_kumulatif: np.ndarray,
    ss: float,
    *,
    review: int = 1,
    lead: int = 7,
    warmup: int = 0,
) -> dict:
    """Jalankan order-up-to harian dengan tenggang waktu `lead` hari.

    prediksi_kumulatif[t] = ramalan jumlah permintaan review+lead hari ke depan pada hari t.
    ss = persediaan pengaman (unit). Mengembalikan fill rate, biaya komponen, dan jejak harian.
    """
    n = len(demand)
    level = float(prediksi_kumulatif[0]) if n else 0.0
    on_hand, backlog = float(np.ceil(max(level + ss, 0.0))), 0.0
    datang = np.zeros(n + lead + 1)
    unmet_jejak, on_hand_jejak, demand_jejak = [], [], []
    for t in range(n):
        on_hand += datang[t]                     # pesanan tiba lebih dulu
        dibayar = min(on_hand, backlog)          # tunggakan lama dilunasi dulu
        on_hand -= dibayar
        backlog -= dibayar
        served = min(on_hand, demand[t])
        on_hand -= served
        unmet = demand[t] - served
        backlog += unmet
        if t % review == 0:                      # keputusan pemesanan
            posisi = on_hand - backlog
            level = float(prediksi_kumulatif[t]) + ss
            qty = max(level - posisi, 0.0)
            if t + lead < len(datang):
                datang[t + lead] += qty
        unmet_jejak.append(unmet)
        on_hand_jejak.append(on_hand)
        demand_jejak.append(demand[t])
    unmet_jejak = np.array(unmet_jejak)
    on_hand_jejak = np.array(on_hand_jejak)
    demand_jejak = np.array(demand_jejak)
    potong = slice(warmup, n)
    dem = demand_jejak[potong].sum()
    un = unmet_jejak[potong].sum()
    fill = float(1 - un / dem) if dem > 0 else np.nan
    return {
        "fill_rate": fill,
        "unmet_total": float(un),
        "on_hand_total": float(on_hand_jejak[potong].sum()),
        "backlog_akhir": float(backlog),
        "demand_total": float(dem),
        "served_total": float(dem - un),
        "jejak_on_hand": on_hand_jejak,
        "jejak_unmet": unmet_jejak,
        "jejak_demand": demand_jejak,
    }


def kuantil_galat(galat_validasi: np.ndarray, taus: list[float]) -> dict[float, float]:
    """Persediaan pengaman = kuantil empiris galat validasi (galat positif = permintaan di atas ramalan)."""
    g = np.asarray(galat_validasi, float)
    g = g[~np.isnan(g)]
    return {t: float(np.quantile(g, t)) if g.size else 0.0 for t in taus}


def sapu_kebijakan(
    demand: np.ndarray,
    prediksi: np.ndarray,
    galat_validasi: np.ndarray,
    taus: list[float],
    **kw,
) -> pd.DataFrame:
    """Sapu grid kuantil: satu baris hasil simulasi per level tau."""
    ss = kuantil_galat(galat_validasi, taus)
    baris = []
    for t, nilai in ss.items():
        hasil = simulasi_order_up_to(demand, prediksi, nilai, **kw)
        baris.append({"tau": t, "ss": nilai, **hasil})
    return pd.DataFrame(baris)


def setarakan_layanan(sweep: pd.DataFrame, target: list[float]) -> pd.DataFrame:
    """Biaya pada fill rate target = interpolasi linear kurva (fill_rate, biaya) per SKU-model."""
    out = []
    fr = sweep["fill_rate"].to_numpy(float)
    cost = sweep["cost_total"].to_numpy(float)
    urut = np.argsort(fr)
    fr_s, cost_s = fr[urut], cost[urut]
    for t in target:
        if fr_s.size == 0 or fr_s.min() > t or fr_s.max() < t:
            out.append({"fill_target": t, "cost_at_target": np.nan, "tercakup": False,
                        "fill_min": float(fr_s.min()) if fr_s.size else np.nan,
                        "fill_max": float(fr_s.max()) if fr_s.size else np.nan})
        else:
            out.append({"fill_target": t, "cost_at_target": float(np.interp(t, fr_s, cost_s)),
                        "tercakup": True, "fill_min": float(fr_s.min()), "fill_max": float(fr_s.max())})
    return pd.DataFrame(out)


def biaya(sweep: pd.DataFrame, cs: float, ch: float) -> pd.DataFrame:
    out = sweep.copy()
    out["cost_stockout"] = out["unmet_total"] * cs
    out["cost_holding"] = out["on_hand_total"] * ch
    out["cost_total"] = out["cost_stockout"] + out["cost_holding"]
    out["cs"], out["ch"] = cs, ch
    return out
