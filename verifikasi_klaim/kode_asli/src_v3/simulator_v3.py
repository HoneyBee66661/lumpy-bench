"""simulator_v3.py — simulator persediaan v3 (Fase 1). Akuntansi NET + tes invarian.

Perbedaan pokok dari simulator v2 (yang diaudit):
  1. Backorder BENAR: stok yang datang melunasi backlog lebih dulu.
       net_t   = net_{t-1} + received_t - demand_t          (net negatif = backlog)
       on_hand = max(net_t, 0) ; backlog = max(-net_t, 0)
       stockout hari t = demand yang tidak terpenuhi HARI ITU (backlog lama sudah dilunasi dulu)
  2. Tidak ada look-ahead: `target` (level order-up-to) disuplai pemanggil dari model yang hanya
     memakai informasi <= t (forecast periode proteksi R+L, lihat features_v3/forecast_v3).
  3. Konvensi biaya eksplisit: 'per_unit_once' (unit gagal dipenuhi hari itu) atau 'per_unit_day'
     (unit backlog yang masih tertunggak di akhir hari).
  4. Stok awal, warm-up, pembulatan order (pecahan/integer/MOQ), dan review period R semuanya
     parameter — bukan asumsi tersembunyi.
  5. `simulate_legacy()` mereproduksi simulator v2 apa adanya (bug + look-ahead yang diberikan
     pemanggil) untuk regresi: angka v2 harus bisa direplikasi, bukan diperdebatkan.

Invariant yang diuji di tests/test_simulator_v3.py:
  I1 mass balance: total demand = total terlayani + backlog akhir
  I2 IP = net + pesanan terbuka
  I3 on_hand <= level kecuali level turun (jumlah hari dilaporkan)
  I4 backlog turun ketika barang datang
  I5 biaya dihitung ulang dari trace = biaya ringkasan
  I6 tidak ada stok negatif; net = on_hand - backlog
  I7 kausalitas: mengubah demand setelah hari t tidak mengubah keputusan order s/d hari t
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Policy:
    """Parameter kebijakan + konvensi biaya. Tidak ada nilai default yang tersembunyi di loop."""

    review_period_days: int = 1          # R
    lead_time_days: int = 7              # L
    stockout_cost: float = 5.0           # Cs per unit
    holding_cost: float = 1.0            # Ch per unit (per hari)
    cost_convention: str = "per_unit_once"   # 'per_unit_once' | 'per_unit_day'
    initial_rule: str = "target_level"   # 'target_level' | 'zero'
    rounding: str = "pecahan"            # 'pecahan' | 'integer_ceil' | 'moq'
    moq: float = 0.0
    warmup_days: int = 28                # hari awal periode uji tidak dihitung dalam biaya
    # Hari terakhir yang boleh DIBEBANKAN biaya. Dipakai untuk aturan batas data: hari yang jendela
    # proteksinya (t+1..t+H) tidak lengkap tidak dihitung, supaya semua model dinilai pada hari-hari
    # yang informasinya benar-benar tersedia (lihat policy_v3.billing_window).
    bill_until_day: int | None = None
    extra: dict = field(default_factory=dict)

    @property
    def protection_days(self) -> int:
        return int(self.review_period_days) + int(self.lead_time_days)


TRACE_COLS = [
    "day",
    "demand",
    "target_level",
    "received",
    "on_hand_awal",
    "backlog_awal",
    "settle_backlog",
    "served",
    "unmet",
    "stockout_units",
    "on_hand",
    "backlog_end",
    "net_position",
    "inventory_position",
    "ip_after_order",
    "order_qty",
    "order_arrival_day",
    "cost_stockout",
    "cost_holding",
    "cost_day",
]


def _apply_rounding(qty: float, policy: Policy) -> float:
    if qty <= 0:
        return 0.0
    if policy.rounding == "integer_ceil":
        return float(math.ceil(qty))
    if policy.rounding == "moq":
        moq = float(policy.moq)
        if moq <= 0:
            return qty
        return float(math.ceil(qty / moq) * moq)
    return float(qty)


def _iterate(d: pd.Series, S: pd.Series, policy: Policy, days: np.ndarray, initial_net: float):
    """Generator hari-per-hari (satu implementasi akuntansi untuk semua pemakai).

    Menghasilkan dict per hari dengan semua besaran audit-able, supaya `simulate()` (butuh trace)
    dan `simulate_aggregate()` (butuh ringkasan cepat) tidak pernah berbeda logika.
    """
    L = int(policy.lead_time_days)
    R = int(policy.review_period_days)
    cs = float(policy.stockout_cost)
    ch = float(policy.holding_cost)
    first = int(days[0])
    net = float(initial_net)
    on_hand = max(net, 0.0)
    backlog = max(-net, 0.0)
    arrivals: dict[int, float] = {}
    for t in days:
        t = int(t)
        received = arrivals.pop(t, 0.0)
        on_hand_awal = on_hand
        backlog_awal = backlog
        on_hand_setelah_datang = on_hand_awal + received
        settle_backlog = min(on_hand_setelah_datang, backlog_awal)
        backlog_setelah_settle = backlog_awal - settle_backlog
        tersedia = on_hand_setelah_datang - settle_backlog
        demand_t = float(d.loc[t])
        served = min(tersedia, demand_t)
        unmet = demand_t - served
        on_hand = tersedia - served
        backlog = backlog_setelah_settle + unmet
        net = on_hand - backlog

        is_review = ((t - first) % R == 0)
        open_orders = sum(arrivals.values())
        position = net + open_orders
        order = 0.0
        if is_review:
            order = _apply_rounding(max(0.0, float(S.loc[t]) - position), policy)
        if order > 0:
            arrivals[t + L] = arrivals.get(t + L, 0.0) + order

        if policy.cost_convention == "per_unit_day":
            cost_so = backlog * cs
        elif policy.cost_convention == "per_unit_once":
            cost_so = unmet * cs
        else:
            raise ValueError(f"cost_convention tidak dikenal: {policy.cost_convention}")

        yield {
            "day": t,
            "demand": demand_t,
            "target_level": float(S.loc[t]),
            "received": received,
            "on_hand_awal": on_hand_awal,
            "backlog_awal": backlog_awal,
            "settle_backlog": settle_backlog,
            "served": served,
            "unmet": unmet,
            "stockout_units": unmet,
            "on_hand": on_hand,
            "backlog_end": backlog,
            "net_position": net,
            "inventory_position": position,
            "ip_after_order": position + order,
            "order_qty": order,
            "order_arrival_day": (t + L) if order > 0 else np.nan,
            "cost_stockout": cost_so,
            "cost_holding": on_hand * ch,
        }


def simulate(
    demand: pd.Series,
    target: pd.Series,
    policy: Policy,
    days: np.ndarray | None = None,
    initial_net: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Jalankan satu deret (satu SKU) dengan kebijakan (R,S) dan akuntansi net yang benar.

    demand : Series index = hari, nilai = demand aktual.
    target : Series index = hari, nilai = level order-up-to S_t (dari model, info <= t).
    days   : urutan hari yang disimulasikan (default: index demand).
    initial_net : stok awal (net). Default mengikuti policy.initial_rule.
    """
    if days is None:
        days = demand.index.to_numpy()
    days = np.asarray(days)

    d = demand.reindex(days).astype(float)
    S = target.reindex(days).astype(float)

    # gagal keras, jangan diam-diam menghasilkan NaN (audit v2 menemukan angka "hantu" dari asumsi
    # yang tidak diperiksa; di v3 input tidak lengkap harus menghentikan run)
    bad_d = np.isnan(d.to_numpy()).sum()
    bad_s = np.isnan(S.to_numpy()).sum()
    if bad_d or bad_s:
        raise ValueError(
            f"input mengandung NaN: demand {bad_d} hari, target/level {bad_s} hari "
            f"(hari pertama {int(days[0])}) — perbaiki deret target/demand sebelum simulasi"
        )

    first = int(days[0])
    if initial_net is None:
        initial_net = float(S.loc[first]) if policy.initial_rule == "target_level" else 0.0

    rows = list(_iterate(d, S, policy, days, initial_net))
    trace = pd.DataFrame(rows)
    trace["cost_day"] = trace["cost_stockout"] + trace["cost_holding"]

    warm = int(policy.warmup_days)
    billed = trace.iloc[warm:] if warm > 0 else trace
    if policy.bill_until_day is not None:
        billed = billed.loc[billed["day"] <= int(policy.bill_until_day)]
    summary = {
        "n_days": int(len(trace)),
        "n_days_dibebankan": int(len(billed)),
        "demand_total": float(trace["demand"].sum()),
        "demand_dibebankan": float(billed["demand"].sum()),
        "served_total": float(trace["served"].sum()),
        "unmet_total": float(trace["unmet"].sum()),
        "settle_backlog_total": float(trace["settle_backlog"].sum()),
        "stockout_units_total": float(billed["stockout_units"].sum()),
        "backlog_end": float(trace["backlog_end"].iloc[-1]),
        "on_hand_total": float(billed["on_hand"].sum()),
        "cost_stockout": float(billed["cost_stockout"].sum()),
        "cost_holding": float(billed["cost_holding"].sum()),
        "cost_total": float(billed["cost_day"].sum()),
        "fill_rate": float(1 - billed["stockout_units"].sum() / billed["demand"].sum()) if billed["demand"].sum() else np.nan,
        "hari_on_hand_di_atas_level": int((trace["on_hand"] > trace["target_level"] + 1e-9).sum()),
        "initial_net": float(initial_net),
        "hari_order": int((trace["order_qty"] > 0).sum()),
        "pis": float((trace["on_hand"] > 1e-12).mean()),
    }
    return trace, summary


def simulate_aggregate(
    demand: pd.Series,
    target: pd.Series,
    policy: Policy,
    days: np.ndarray,
    initial_net: float | None = None,
) -> dict:
    """Versi cepat: hanya ringkasan (tanpa menyimpan trace). Logika identik dengan `simulate()`."""
    d = demand.reindex(days).astype(float)
    S = target.reindex(days).astype(float)
    bad_d = np.isnan(d.to_numpy()).sum()
    bad_s = np.isnan(S.to_numpy()).sum()
    if bad_d or bad_s:
        raise ValueError(f"input mengandung NaN: demand {bad_d} hari, target {bad_s} hari")
    first = int(days[0])
    if initial_net is None:
        initial_net = float(S.loc[first]) if policy.initial_rule == "target_level" else 0.0

    warm = int(policy.warmup_days)
    tot = {
        "demand_total": 0.0,
        "demand_dibebankan": 0.0,
        "served_total": 0.0,
        "unmet_total": 0.0,
        "settle_backlog_total": 0.0,
        "stockout_units_total": 0.0,
        "on_hand_total": 0.0,
        "backlog_hari_total": 0.0,
        "cost_stockout": 0.0,
        "cost_holding": 0.0,
        "n_hari": 0,
        "n_hari_dibebankan": 0,
        "n_hari_ada_stok": 0,
        "backlog_end": 0.0,
        "hari_on_hand_di_atas_level": 0,
        "hari_order": 0,
    }
    for i, row in enumerate(_iterate(d, S, policy, days, initial_net)):
        dibebankan = i >= warm
        if dibebankan and policy.bill_until_day is not None and row["day"] > int(policy.bill_until_day):
            dibebankan = False
        tot["demand_total"] += row["demand"]
        tot["served_total"] += row["served"]
        tot["unmet_total"] += row["unmet"]
        tot["settle_backlog_total"] += row["settle_backlog"]
        tot["n_hari"] += 1
        if row["on_hand"] > 1e-12:
            tot["n_hari_ada_stok"] += 1
        if row["on_hand"] > row["target_level"] + 1e-9:
            tot["hari_on_hand_di_atas_level"] += 1
        if row["order_qty"] > 0:
            tot["hari_order"] += 1
        if dibebankan:
            tot["demand_dibebankan"] += row["demand"]
            tot["stockout_units_total"] += row["stockout_units"]
            tot["on_hand_total"] += row["on_hand"]
            tot["backlog_hari_total"] += row["backlog_end"]
            tot["cost_stockout"] += row["cost_stockout"]
            tot["cost_holding"] += row["cost_holding"]
            tot["n_hari_dibebankan"] += 1
    tot["backlog_end"] = float(row["backlog_end"])
    tot["cost_total"] = tot["cost_stockout"] + tot["cost_holding"]
    tot["fill_rate"] = (1 - tot["stockout_units_total"] / tot["demand_dibebankan"]) if tot["demand_dibebankan"] else np.nan
    tot["pis"] = tot["n_hari_ada_stok"] / max(tot["n_hari"], 1)
    tot["initial_net"] = float(initial_net)
    return tot


def simulate_legacy(
    demand: pd.Series,
    leaky_target: pd.Series,
    safety_stock: float,
    initial_inventory: float,
    policy: Policy,
    days: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Reproduksi simulator v2 apa adanya (backlog tidak pernah dilunasi + target look-ahead).

    Dipakai HANYA untuk regresi: membuktikan angka v2 bisa direplikasi dari kode v3 saat flag ini
    dinyalakan, sehingga perbaikan hanya mengubah yang dimaksud (prompt v3 Fase 1).
    """
    if days is None:
        days = demand.index.to_numpy()
    days = np.asarray(days)
    L = int(policy.lead_time_days)
    cs = float(policy.stockout_cost)
    ch = float(policy.holding_cost)
    d = demand.reindex(days).astype(float)
    rl = leaky_target.reindex(days).astype(float)

    on_hand = float(initial_inventory)
    backlog = 0.0
    arrivals: dict[int, float] = {}
    rows: list[dict] = []
    for t in days:
        t = int(t)
        received = arrivals.pop(t, 0.0)
        on_hand += received
        demand_t = float(d.loc[t])
        filled = min(on_hand, demand_t)
        on_hand -= filled
        stockout_t = demand_t - filled
        backlog += stockout_t
        holding = max(on_hand, 0.0)
        position = on_hand + sum(arrivals.values()) - backlog
        order = max(0.0, float(rl.loc[t]) + float(safety_stock) - position)
        if order > 0:
            arrivals[t + L] = arrivals.get(t + L, 0.0) + order
        rows.append(
            {
                "day": t,
                "demand": demand_t,
                "received": received,
                "served": filled,
                "unmet": stockout_t,
                "stockout_units": stockout_t,
                "settle_backlog": 0.0,          # <-- BUG v2: barang datang tidak pernah melunasi backlog
                "on_hand": holding,
                "backlog_end": backlog,
                "order_qty": order,
                "cost_stockout": stockout_t * cs,
                "cost_holding": holding * ch,
            }
        )
    trace = pd.DataFrame(rows)
    trace["cost_day"] = trace["cost_stockout"] + trace["cost_holding"]
    summary = {
        "n_days": int(len(trace)),
        "demand_total": float(trace["demand"].sum()),
        "stockout_units_total": float(trace["stockout_units"].sum()),
        "on_hand_total": float(trace["on_hand"].sum()),
        "cost_stockout": float(trace["cost_stockout"].sum()),
        "cost_holding": float(trace["cost_holding"].sum()),
        "cost_total": float(trace["cost_day"].sum()),
        "fill_rate": float(1 - trace["stockout_units"].sum() / trace["demand"].sum()),
        "initial_inventory": float(initial_inventory),
    }
    return trace, summary


def leaky_protection_target(forecast: pd.Series, protection_days: int) -> pd.Series:
    """Target gaya v2: jumlah forecast[t..t+H-1] (mengandung look-ahead) — HANYA untuk regresi legacy."""
    fe = np.r_[forecast.to_numpy(float), np.repeat(float(forecast.iloc[-1]), protection_days)]
    n = len(forecast)
    vals = np.array([fe[t: t + protection_days].sum() for t in range(n)])
    return pd.Series(vals, index=forecast.index)


# ------------------------------------------------------------------ invariant helpers (dipakai tes)
def cek_invarian(trace: pd.DataFrame, summary: dict, tol: float = 1e-9) -> dict:
    """Periksa I1-I6 pada satu trace. Mengembalikan dict nama -> (lulus, nilai aktual)."""
    out: dict[str, tuple[bool, float]] = {}

    def rnd(x: float) -> float:
        return round(float(x), 6)

    # I1 mass balance (satuan DEMAND): setiap demand yang datang menjadi terlayani atau unmet.
    out["I1_mass_balance"] = (
        abs(float(trace["demand"].sum() + max(-summary.get("initial_net", 0.0), 0.0)
                  - trace["served"].sum() - trace["unmet"].sum())) < 1e-6,
        float(trace["demand"].sum() + max(-summary.get("initial_net", 0.0), 0.0)
              - trace["served"].sum() - trace["unmet"].sum()),
    )

    # I1b neraca unit: net_akhir = net_awal + Σkedatangan − Σdemand
    pred = summary.get("initial_net", 0.0) + float(trace["received"].sum()) - float(trace["demand"].sum())
    out["I1b_neraca_unit"] = (abs(pred - float(trace["net_position"].iloc[-1])) < 1e-6,
                              pred - float(trace["net_position"].iloc[-1]))

    # I1c neraca backlog: backlog_akhir = backlog_awal + Σunmet − Σyang dilunasi
    backlog_awal0 = max(-summary.get("initial_net", 0.0), 0.0)
    pred_b = backlog_awal0 + float(trace["unmet"].sum()) - float(trace["settle_backlog"].sum())
    out["I1c_neraca_backlog"] = (abs(pred_b - float(trace["backlog_end"].iloc[-1])) < 1e-6,
                                 pred_b - float(trace["backlog_end"].iloc[-1]))

    # I4 backlog dilunasi ketika barang datang: hari dengan received>0 & backlog_awal>0 harus settle>0
    mask = (trace["received"] > tol) & (trace["backlog_awal"] > tol)
    if mask.any():
        lunas = bool((trace.loc[mask, "settle_backlog"] > tol).all())
    else:
        lunas = True
    out["I4_backlog_dilunasi_saat_datang"] = (lunas, float(mask.sum()))

    # I5 biaya dari trace == ringkasan (dibebankan setelah warm-up)
    warm = summary["n_days"] - summary["n_days_dibebankan"]
    billed = trace.iloc[warm:]
    out["I5_biaya_konsisten"] = (
        abs(float(billed["cost_day"].sum()) - summary["cost_total"]) < 1e-6,
        float(billed["cost_day"].sum()) - summary["cost_total"],
    )

    # I6 net = on_hand - backlog, tidak ada stok negatif
    ok6 = bool(np.allclose(trace["net_position"], trace["on_hand"] - trace["backlog_end"], atol=1e-9)
               and (trace["on_hand"] >= -1e-12).all() and (trace["backlog_end"] >= -1e-12).all())
    out["I6_net_konsisten"] = (ok6, float((trace["net_position"] - (trace["on_hand"] - trace["backlog_end"])).abs().max()))

    # I3 on_hand di atas level. Wajib ~0 HANYA bila level konstan (target berubah-ubah wajar
    # membuat stok lama melebihi level baru — itu konsekuensi kebijakan, bukan bug).
    level_konstan = trace["target_level"].nunique() == 1
    n_over = summary["hari_on_hand_di_atas_level"]
    if level_konstan:
        out["I3_on_hand_di_atas_level_konstan"] = (n_over <= 0.05 * len(trace), float(n_over))
    else:
        out["I3_info_on_hand_di_atas_level_bervariasi"] = (True, float(n_over))
    return out
