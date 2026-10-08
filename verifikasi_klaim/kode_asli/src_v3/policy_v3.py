"""policy_v3.py — Fase 2 & 4: kebijakan (R,S) berbasis KUANTIL EMPIRIS + perbandingan service-equalized.

Inti (menjawab audit):
  * Level order-up-to S_t = pred_sum_t + SS_sku(tau), dengan SS dari KUANTIL EMPIRIS error periode
    proteksi pada jendela validasi — bukan z·sigma·sqrt(L+R) dengan asumsi Normal (audit C: residual
    skew ~1,8 dan Shapiro menolak Normal di semua SKU).
  * tau disapu pada kisi (0,50..0,99). Sapuan ini yang menghasilkan KURVA BIAYA vs FILL RATE, sehingga
    dua model bisa dibandingkan pada TINGKAT LAYANAN YANG SAMA (95% utama, 90/98% pendukung) — bukan
    pada safety factor yang sama seperti v2 (audit: perbandingan v2 tidak adil bagi kedua model).
  * Keputusan order TIDAK bergantung pada (Cs, Ch): biaya hanya mengubah penilaian, bukan perilaku.
    Karena itu satu simulasi per (SKU, model, tau) cukup untuk semua skenario biaya — dihitung dari
    komponen (unmet, backlog-hari, on-hand) supaya kedua konvensi biaya bisa dilaporkan dari run yang sama.
"""
from __future__ import annotations

import sys
from pathlib import Path

from dataclasses import replace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src_v3"))
from simulator_v3 import Policy, simulate, simulate_aggregate  # noqa: E402

MIN_OBS_SS = 30          # minimal observasi error per SKU untuk kuantil per-SKU; kurang -> pooled


def safety_stock_table(
    err: pd.DataFrame,
    quantile_levels: list[float],
    group_col: str = "cat_id",
    min_obs: int = MIN_OBS_SS,
) -> tuple[dict[float, pd.DataFrame], pd.DataFrame]:
    """SS per SKU per tau dari kuantil empiris error validasi; SKU dengan data kurang -> pooled grup.

    err: kolom sku_id, err (error periode proteksi pada validasi), cat_id (untuk fallback).
    Mengembalikan ({tau: DataFrame(sku_id, ss, sumber)}, tabel_diagnostik).
    """
    out: dict[float, pd.DataFrame] = {}
    diag_rows = []
    pooled_all = {tau: float(np.quantile(err["err"].to_numpy(), tau)) for tau in quantile_levels}
    grp = err.groupby(group_col, sort=False)["err"]
    pooled_grp = {
        tau: grp.apply(lambda s: float(np.quantile(s.to_numpy(), tau))).to_dict() for tau in quantile_levels
    }
    sigma_err = err.groupby("sku_id", sort=False)["err"].agg(["count", "std"])

    for tau in quantile_levels:
        rows = []
        n_pool = 0
        per_sku = err.groupby("sku_id", sort=False)["err"].apply(lambda s: float(np.quantile(s.to_numpy(), tau)))
        cat_of = err.groupby("sku_id", sort=False)[group_col].first()
        for sku, ss in per_sku.items():
            n = int(sigma_err.loc[sku, "count"])
            if n >= min_obs:
                rows.append({"sku_id": sku, "ss": float(ss), "sumber": f"sku (n={n})", "tau": tau})
            else:
                kat = cat_of.loc[sku]
                val = pooled_grp[tau].get(kat, pooled_all[tau])
                rows.append({"sku_id": sku, "ss": float(val), "sumber": f"pooled {group_col}={kat} (n={n})", "tau": tau})
                n_pool += 1
        tbl = pd.DataFrame(rows)
        out[tau] = tbl
        diag_rows.append({"tau": tau, "n_sku_pakai_sku": len(tbl) - n_pool, "n_sku_pooled": n_pool,
                          "ss_median": float(tbl["ss"].median()), "ss_min": float(tbl["ss"].min()),
                          "ss_max": float(tbl["ss"].max()), "ss_negatif": int((tbl["ss"] < 0).sum())})
    return out, pd.DataFrame(diag_rows)


def _fill_boundary_level(level: pd.Series, policy_base: Policy) -> pd.Series:
    """Isi NaN level di HARI-HARI DI LUAR JENDELA PENAGIHAN: kalender jendela proteksi (t+1..t+H)
    belum tersedia karena melampaui ujung data.

    Kenapa ini tidak mempengaruhi hasil: `billing_window()` membatasi hari yang dibebankan biaya ke
    t <= max(days) - H, sehingga keputusan di hari-hari yang diisi di sini tidak pernah masuk
    perhitungan biaya (diverifikasi di tests/smoke_v3.py). Nilai diisi dengan level terakhir yang
    tersedia supaya deret tidak NaN — bukan untuk mengarang keputusan yang berpengaruh.
    """
    if level.isna().any():
        level = level.ffill().bfill()
    return level


def billing_window(days: np.ndarray, policy_base: Policy) -> Policy:
    """Kebijakan dengan jendela penagihan biaya yang aman terhadap batas data.

    Aturan: hari yang dibebankan maksimum = max(days) - H dengan H = R + L (hari proteksi). Alasannya
    bukan kosmetik: keputusan di hari t memakai fitur jendela proteksi t+1..t+H, dan order yang dipesan
    di hari t tiba di t+L. Untuk t > max(days) - H, jendela proteksi melampaui data sehingga fitur
    kalender jendela tidak lengkap, dan untuk t = max(days) - L order-nya masih tiba tepat di hari
    terakhir (jadi tidak bisa diabaikan begitu saja). Semua model dinilai pada hari yang sama.
    """
    H = int(policy_base.review_period_days) + int(policy_base.lead_time_days)
    return replace(policy_base, bill_until_day=int(np.max(days)) - H)


def build_level(pred_sum: pd.Series, ss: float, days: np.ndarray, policy_base: Policy) -> pd.Series:
    """Deret level order-up-to S_t = pred_sum_t + SS, di-clip >= 0, batas akhir data ditangani."""
    return _fill_boundary_level((pred_sum.reindex(days) + float(ss)).clip(lower=0.0), policy_base)


def run_sweep(
    pred: pd.DataFrame,
    ss_tables: dict[float, pd.DataFrame],
    demand_by_sku: dict[str, pd.Series],
    days: np.ndarray,
    policy_base: Policy,
    model_name: str,
) -> pd.DataFrame:
    """Satu model: simulasi untuk tiap SKU x tau -> tabel agregat (biaya dihitung dari komponen).

    pred: kolom sku_id, day, pred_sum (prediksi jumlah demand periode proteksi, info <= t).
    """
    rows = []
    pred_pivot = {sku: g.set_index("day")["pred_sum"] for sku, g in pred.groupby("sku_id", sort=False)}
    pol = billing_window(days, policy_base)
    for tau, tbl in ss_tables.items():
        ss_map = tbl.set_index("sku_id")["ss"].to_dict()
        for sku, d in demand_by_sku.items():
            p = pred_pivot.get(sku)
            if p is None or sku not in ss_map:
                continue
            level = build_level(p, ss_map[sku], days, pol)
            agg = simulate_aggregate(d, level, pol, days)
            rows.append(
                {
                    "sku_id": sku,
                    "model": model_name,
                    "tau": float(tau),
                    "ss": float(ss_map[sku]),
                    "demand": agg["demand_dibebankan"],
                    "unmet": agg["stockout_units_total"],
                    "backlog_hari": agg["backlog_hari_total"],
                    "on_hand": agg["on_hand_total"],
                    "fill_rate": agg["fill_rate"],
                    "pis": agg["pis"],
                    "hari_order": agg["hari_order"],
                    "n_hari_dibebankan": agg["n_hari_dibebankan"],
                    "hari_on_hand_di_atas_level": agg["hari_on_hand_di_atas_level"],
                    "level_median": float(level.median()),
                }
            )
    return pd.DataFrame(rows)


def add_costs(df: pd.DataFrame, scenarios: list[tuple[float, float]], convention: str) -> pd.DataFrame:
    """Hitung biaya tiap skenario dari komponen (tidak perlu simulasi ulang)."""
    frames = []
    for cs, ch in scenarios:
        sub = df.copy()
        if convention == "per_unit_day":
            sub["cost_stockout"] = sub["backlog_hari"] * float(cs)
        else:
            sub["cost_stockout"] = sub["unmet"] * float(cs)
        sub["cost_holding"] = sub["on_hand"] * float(ch)
        sub["cost_total"] = sub["cost_stockout"] + sub["cost_holding"]
        sub["cs"] = float(cs)
        sub["ch"] = float(ch)
        sub["ratio_cs_ch"] = float(cs) / float(ch)
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)


def service_equalized(df: pd.DataFrame, fill_targets: list[float]) -> pd.DataFrame:
    """Biaya pada fill rate target (interpolasi linear antar titik sapuan tau, per SKU).

    Untuk tiap SKU x model x skenario: biaya pada fill rate target = interpolasi kurva
    (fill_rate(tau), cost(tau)). SKU yang kurvanya tidak mencakup target dilaporkan terpisah
    (tidak dipaksa) supaya tidak ada angka karangan.
    """
    rows = []
    for (sku, model, cs), g in df.groupby(["sku_id", "model", "cs"], sort=False):
        g = g.sort_values("fill_rate")
        fr = g["fill_rate"].to_numpy(float)
        cost = g["cost_total"].to_numpy(float)
        for target in fill_targets:
            if fr.min() > target or fr.max() < target:
                rows.append({"sku_id": sku, "model": model, "cs": cs, "fill_target": target,
                             "cost_at_target": np.nan, "tercakup": False,
                             "fill_min": fr.min(), "fill_max": fr.max()})
                continue
            # interpolasi pada fill rate yang meningkat; kurva tidak selalu monoton -> pakai rata-rata
            # biaya di sekitar target (interpolasi hanya pada selang yang mengurung target)
            idx = np.argsort(fr)
            fr_s, cost_s = fr[idx], cost[idx]
            val = float(np.interp(target, fr_s, cost_s))
            rows.append({"sku_id": sku, "model": model, "cs": cs, "fill_target": target,
                         "cost_at_target": val, "tercakup": True, "fill_min": fr.min(), "fill_max": fr.max()})
    return pd.DataFrame(rows)


def sample_trace(
    pred: pd.DataFrame,
    ss_tables: dict[float, pd.DataFrame],
    demand_by_sku: dict[str, pd.Series],
    days: np.ndarray,
    policy_base: Policy,
    skus: list[str],
    tau: float,
    model_name: str = "trace",
) -> pd.DataFrame:
    """Trace harian (untuk gambar/anekdot) pada SKU terpilih dan satu tau saja."""
    ss_map = ss_tables[tau].set_index("sku_id")["ss"].to_dict()
    pred_pivot = {sku: g.set_index("day")["pred_sum"] for sku, g in pred.groupby("sku_id", sort=False)}
    policy_base = billing_window(days, policy_base)
    frames = []
    for sku in skus:
        if sku not in pred_pivot or sku not in ss_map:
            continue
        level = build_level(pred_pivot[sku], ss_map[sku], days, policy_base)
        tr, _ = simulate(demand_by_sku[sku], level, policy_base, days)
        tr.insert(0, "sku_id", sku)
        tr.insert(1, "model", model_name)
        tr.insert(2, "tau", tau)
        frames.append(tr)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
