"""5.7-5.8 Simulasi persediaan order-up-to + kalkulasi biaya asimetris (spec v2: tidak berubah
selain dua jenis selisih biaya yang wajib dipisah).

Asumsi yang HARUS ditulis eksplisit di Metode (spec 5.7):
- Kebijakan order-up-to, review period R = 1 hari, lead time L = 7 hari (tetap).
- Backorder diperbolehkan: permintaan yang tidak terpenuhi dicatat sebagai unit stockout,
  tidak hilang dari sistem (masuk backlog dan tetap dipenuhi di hari berikutnya).
- Inventori awal = ekspektasi demand selama (R+L) pada awal periode uji, dihitung dari
  FORECAST model (bukan dari demand uji) supaya tidak ada leakage:
    RF      : jumlah forecast 1 langkah untuk R+L hari pertama periode uji
    Croston : laju hari pertama periode uji x (R+L)

Algoritma harian (spec 5.7), untuk hari t di periode uji:
  1. forecast_R_L  = jumlah forecast R+L hari ke depan (RF) atau laju x (R+L) (Croston)
  2. sigma         = std residual jendela validasi (per SKU, per model)
  3. z             = norm.ppf(Cs / (Cs + Ch))
  4. safety_stock  = z * sigma * sqrt(R + L)
  5. level         = forecast_R_L + safety_stock
  6. terima kiriman yang datang hari ini, penuhi demand aktual (catat stockout kalau kurang),
     hitung holding = stok akhir hari, lalu pesan sampai `level` dihitung dari posisi
     inventori (stok + pesanan terbuka - backlog)
  7. ulangi untuk tiap skenario rasio biaya di config.yaml

Biaya (5.8): total = sum_hari [stockout_units * Cs + holding_units * Ch].

DUA JENIS SELISIH (spec v2 5.8) — beda secara matematis, jadi disimpan terpisah di
cost_summary.csv (kolom sama untuk kedua baris model pada satu skenario, supaya bisa
digabung tanpa pivot):
  selisih_median_berpasangan_per_sku = median_i(RF_i - Croston_i)   (dihitung per SKU dulu)
  selisih_median_agregat             = median(RF) - median(Croston) (selisih dua median)
Nilai negatif berarti RF lebih murah.

Tidak ada angka acak di modul ini — simulasi deterministik dari input CSV hasil M3/M4.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_rf import split_bounds  # noqa: E402

TRACE_COLS = [
    "sku_id",
    "model",
    "cs",
    "ch",
    "day",
    "demand",
    "forecast",
    "forecast_rl",
    "safety_stock",
    "order_up_to_level",
    "inventory_position",
    "received",
    "order_qty",
    "stockout_units",
    "holding_units",
    "backlog_end",
    "cost_stockout",
    "cost_holding",
    "cost_day",
]
SUMMARY_COLS = [
    "sku_id",
    "model",
    "cs",
    "ch",
    "ratio_cs_ch",
    "n_days",
    "demand_total",
    "stockout_units_total",
    "holding_units_total",
    "cost_stockout",
    "cost_holding",
    "cost_total",
    "fill_rate",
    "initial_inventory",
    "safety_stock_day1",
    # dua jenis selisih (spec v2 5.8) — konstan per skenario, diulang di kedua baris model
    "selisih_median_berpasangan_per_sku",
    "selisih_median_agregat",
    "selisih_mean_berpasangan_per_sku",
    "rf_lebih_murah_n_sku",
    "n_sku_skenario",
]


def load_forecasts(results_dir: Path) -> dict[str, pd.DataFrame]:
    """Kumpulkan forecast + sigma per SKU per model dari keluaran M3/M4."""
    rf_fc = pd.read_csv(results_dir / "tables" / "rf_test_forecast.csv")[["sku_id", "day", "demand", "forecast"]]
    rf_res = pd.read_csv(results_dir / "tables" / "rf_val_residuals.csv")
    rf_sigma = rf_res.groupby("sku_id")["residual"].std(ddof=1).rename("sigma")
    cr_fc = pd.read_csv(results_dir / "tables" / "croston_test_forecast.csv")[["sku_id", "day", "demand", "forecast"]]
    cr_res = pd.read_csv(results_dir / "tables" / "croston_val_residuals.csv")
    cr_sigma = cr_res.groupby("sku_id")["residual"].std(ddof=1).rename("sigma")
    return {
        "rf": rf_fc.join(rf_sigma, on="sku_id"),
        "croston": cr_fc.join(cr_sigma, on="sku_id"),
    }


def forecast_sum(fc: pd.Series, days: np.ndarray, t: int, horizon: int) -> float:
    """Jumlah forecast untuk `horizon` hari mulai hari t.

    Kalau horizon melewati hari terakhir yang punya forecast (ujung periode uji), hari
    sisanya diisi dengan forecast terakhir yang tersedia — dipakai untuk keperluan
    order-up-to, bukan untuk mengevaluasi akurasi.
    """
    last_val = float(fc.iloc[-1])
    total = 0.0
    for k in range(horizon):
        d = t + k
        if d in fc.index:
            total += float(fc.loc[d])
        else:
            total += last_val
    return total


def simulate(
    demand: pd.Series,
    forecast: pd.Series,
    sigma: float,
    days: np.ndarray,
    review_period: int,
    lead_time: int,
    cs: float,
    ch: float,
) -> tuple[pd.DataFrame, dict]:
    """Jalankan satu SKU x satu model x satu skenario. Kembalikan (trace harian, ringkasan)."""
    rl = review_period + lead_time
    critical = cs / (cs + ch)
    z = float(norm.ppf(critical))
    safety = z * float(sigma) * float(np.sqrt(rl))

    first = int(days[0])
    fc_indexed = forecast.reindex(days).astype(float)
    # inventori awal = ekspektasi demand R+L (berbasis forecast), tanpa safety stock
    on_hand = forecast_sum(fc_indexed, days, first, rl)
    initial_inventory = on_hand
    backlog = 0.0
    arrivals: dict[int, float] = {}
    rows: list[dict] = []

    for t in days:
        t = int(t)
        received = arrivals.pop(t, 0.0)
        on_hand += received
        d = float(demand.loc[t])
        filled = min(on_hand, d)
        on_hand -= filled
        stockout = d - filled
        backlog += stockout
        holding = max(on_hand, 0.0)

        open_orders = sum(arrivals.values())
        position = on_hand + open_orders - backlog
        fc_rl = forecast_sum(fc_indexed, days, t, rl)
        level = fc_rl + safety
        order = max(0.0, level - position)
        if order > 0:
            arrivals[t + lead_time] = arrivals.get(t + lead_time, 0.0) + order

        rows.append(
            {
                "day": t,
                "demand": d,
                "forecast": float(fc_indexed.loc[t]),
                "forecast_rl": fc_rl,
                "safety_stock": safety,
                "order_up_to_level": level,
                "inventory_position": position,
                "received": received,
                "order_qty": order,
                "stockout_units": stockout,
                "holding_units": holding,
                "backlog_end": backlog,
                "cost_stockout": stockout * cs,
                "cost_holding": holding * ch,
            }
        )

    trace = pd.DataFrame(rows)
    trace["cost_day"] = trace["cost_stockout"] + trace["cost_holding"]
    summary = {
        "n_days": len(trace),
        "demand_total": float(trace["demand"].sum()),
        "stockout_units_total": float(trace["stockout_units"].sum()),
        "holding_units_total": float(trace["holding_units"].sum()),
        "cost_stockout": float(trace["cost_stockout"].sum()),
        "cost_holding": float(trace["cost_holding"].sum()),
        "cost_total": float(trace["cost_day"].sum()),
        "fill_rate": float(1 - trace["stockout_units"].sum() / trace["demand"].sum()) if trace["demand"].sum() else np.nan,
        "initial_inventory": float(initial_inventory),
        "safety_stock_day1": safety,
    }
    return trace, summary


def run(config_path: Path, results_dir: Path | None = None) -> pd.DataFrame:
    cfg = yaml.safe_load(config_path.read_text())
    results = Path(results_dir or cfg["paths"]["results"])
    (results / "tables").mkdir(parents=True, exist_ok=True)

    R = int(cfg["inventory_policy"]["review_period_days"])
    L = int(cfg["inventory_policy"]["lead_time_days"])
    scenarios = [(float(cs), float(ch)) for cs, ch in cfg["cost_scenarios"]]
    test_days = int(cfg["split"]["test_days"])

    data = load_forecasts(results)
    total_days = int(max(int(df["day"].max()) for df in data.values()))
    val_end = total_days - test_days
    days = np.arange(val_end + 1, total_days + 1)
    skus = pd.read_csv(results / "tables" / "sku_sample.csv")["sku_id"].tolist()
    print(f"skenario (Cs,Ch): {scenarios}")
    print(f"kebijakan: R={R}, L={L}, backorder=ya, hari uji {days[0]}..{days[-1]} ({len(days)} hari)")

    traces: list[pd.DataFrame] = []
    summaries: list[dict] = []
    for model, df in data.items():
        df = df[df["sku_id"].isin(skus)]
        for sku, g in df.groupby("sku_id"):
            g = g.sort_values("day")
            demand = g.set_index("day")["demand"]
            forecast = g.set_index("day")["forecast"]
            sigma = float(g["sigma"].iloc[0])
            for cs, ch in scenarios:
                trace, summ = simulate(demand, forecast, sigma, days, R, L, cs, ch)
                trace.insert(0, "sku_id", sku)
                trace.insert(1, "model", model)
                trace.insert(2, "cs", cs)
                trace.insert(3, "ch", ch)
                traces.append(trace)
                summaries.append({"sku_id": sku, "model": model, "cs": cs, "ch": ch,
                                  "ratio_cs_ch": cs / ch, **summ})

    summ_all = pd.DataFrame(summaries)

    # --- dua jenis selisih biaya per skenario (spec v2 5.8) ---
    sel_rows: list[dict] = []
    for (cs, ch), sub in summ_all.groupby(["cs", "ch"], sort=True):
        piv = sub.pivot_table(index="sku_id", columns="model", values="cost_total")
        diff = piv["rf"] - piv["croston"]
        sel_rows.append(
            {
                "cs": float(cs),
                "ch": float(ch),
                "selisih_median_berpasangan_per_sku": float(np.median(diff)),
                "selisih_mean_berpasangan_per_sku": float(diff.mean()),
                "selisih_median_agregat": float(np.median(piv["rf"]) - np.median(piv["croston"])),
                "rf_lebih_murah_n_sku": int((diff < 0).sum()),
                "n_sku_skenario": int(len(piv)),
            }
        )
    sel = pd.DataFrame(sel_rows)
    summ_all = summ_all.merge(sel, on=["cs", "ch"], how="left")
    summ_all = summ_all[SUMMARY_COLS]

    trace_all = pd.concat(traces, ignore_index=True)[TRACE_COLS]
    # trace harian disimpan ter-gzip (10 MB mentah -> ~1,5 MB) supaya repo tidak membengkak;
    # pandas membaca .gz otomatis dari ekstensi.
    trace_all.to_csv(results / "tables" / "inventory_daily_trace.csv.gz", index=False, compression="gzip")
    summ_all.to_csv(results / "tables" / "cost_summary.csv", index=False)

    print("\nbiaya per skenario (selisih negatif = RF lebih murah):")
    for r in sel.itertuples():
        print(
            f"  Cs:Ch {r.cs:.0f}:{int(r.ch)} | median biaya per SKU -> RF "
            f"{summ_all[(summ_all['cs'] == r.cs) & (summ_all['model'] == 'rf')]['cost_total'].median():,.1f} vs "
            f"Croston {summ_all[(summ_all['cs'] == r.cs) & (summ_all['model'] == 'croston')]['cost_total'].median():,.1f} "
            f"| selisih berpasangan per SKU {r.selisih_median_berpasangan_per_sku:,.1f} "
            f"| selisih agregat (median RF - median Croston) {r.selisih_median_agregat:,.1f} "
            f"| RF lebih murah di {r.rf_lebih_murah_n_sku}/{r.n_sku_skenario} SKU"
        )

    print(f"\ntrace harian: {len(trace_all):,} baris | ringkasan: {len(summ_all)} baris "
          f"({len(skus)} SKU x 2 model x {len(scenarios)} skenario)")
    return summ_all


def main() -> None:
    ap = argparse.ArgumentParser(description="Simulasi persediaan order-up-to + biaya asimetris (M5).")
    ap.add_argument("--config", type=Path, default=Path("config.yaml"))
    ap.add_argument("--results", type=Path, default=None)
    a = ap.parse_args()
    run(a.config, a.results)


if __name__ == "__main__":
    main()
