"""M4 — Croston-SBA sebagai pembanding RF (spec 5.6, tetap per SKU di v2).

Implementasi memakai `statsforecast.models.CrostonSBA` apa adanya, tanpa override parameter.
Spec v2 Bagian 4 & 8 mewajibkan parameter AKTUAL (alpha smoothing + faktor debiasing SBA)
DIVERIFIKASI dari source pustaka dan dicatat di results/run_log.txt — bukan diasumsikan.

`croston_library_params()` membaca source `statsforecast.models._croston_classic` dan
`_croston_sba` saat runtime, mengekstrak nilai alpha SES dan faktor debiasing, memastikan
CrostonSBA tidak punya parameter alpha di konstruktornya, lalu berhenti keras (SystemExit)
kalau nilainya berbeda dari yang tercatat di dokumentasi proyek ini. Jadi kalau pustaka
berubah, run ini gagal — bukan diam-diam menghasilkan angka yang salah.

Keluaran (results/tables/):
  croston_val_residuals.csv   residual (actual - forecast) periode validasi, per SKU per hari
  croston_test_forecast.csv   forecast periode uji (hari 1762..1941), per SKU per hari
  croston_summary.csv         MAE validasi/uji + sigma residual validasi + laju akhir per SKU
  mae_comparison.csv          MAE uji RF (model global) vs Croston per SKU (bahan tabel Hasil)
"""
from __future__ import annotations

import argparse
import inspect
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_rf import SERIES_COLS, load_series, split_bounds  # noqa: E402
from runlog import write as write_run_log  # noqa: E402

ALPHA_DIHAARAPKAN = [0.1]
FAKTOR_SBA_DIHARAPKAN = 0.95


def croston_library_params() -> dict:
    """Ekstrak parameter Croston-SBA langsung dari source pustaka (verifikasi, bukan asumsi)."""
    import statsforecast
    from statsforecast import models as sf

    src_classic = inspect.getsource(sf._croston_classic)
    src_sba = inspect.getsource(sf._croston_sba)
    src_init = inspect.getsource(sf.CrostonSBA.__init__)

    alpha_calls = re.findall(r"_ses_forecast\(\s*[^,]+,\s*([0-9.]+)\s*\)", src_classic)
    alphas = sorted({float(a) for a in alpha_calls})
    faktor = re.search(r'out\["mean"\]\s*\*=\s*([0-9.]+)', src_sba)

    params = {
        "versi_statsforecast": statsforecast.__version__,
        "alpha_ses": alphas,
        "n_pemanggilan_ses": len(alpha_calls),
        "faktor_debiasing_sba": None if faktor is None else float(faktor.group(1)),
        "constructor": src_init.strip().splitlines()[0].strip(),
        "constructor_punya_param_alpha": "alpha" in src_init.split(")")[0],
        "ses_fitted_baris_pertama": "fitted[0] = x[0]" in inspect.getsource(sf._ses_forecast),
    }

    if params["alpha_ses"] != ALPHA_DIHAARAPKAN:
        raise SystemExit(f"alpha SES Croston tidak lagi {ALPHA_DIHAARAPKAN}: {params['alpha_ses']}")
    if params["n_pemanggilan_ses"] != 2:
        raise SystemExit(f"jumlah pemanggilan SES di _croston_classic berubah: {params['n_pemanggilan_ses']}")
    if params["faktor_debiasing_sba"] != FAKTOR_SBA_DIHARAPKAN:
        raise SystemExit(f"faktor debiasing SBA tidak lagi {FAKTOR_SBA_DIHARAPKAN}: {params['faktor_debiasing_sba']}")
    if params["constructor_punya_param_alpha"]:
        raise SystemExit("CrostonSBA sekarang punya parameter alpha di konstruktor — cek ulang spec 5.6")
    return params


def croston_params_log(params: dict) -> list[str]:
    """Baris-baris bukti parameter Croston untuk results/run_log.txt."""
    return [
        f"croston_pustaka   : statsforecast {params['versi_statsforecast']} — statsforecast.models.CrostonSBA",
        f"croston_alpha_ses : {params['alpha_ses']} (dibaca dari source _croston_classic: "
        f"{params['n_pemanggilan_ses']}x _ses_forecast(..., 0.1) — ukuran demand & selang)",
        f"croston_sba_factor: {params['faktor_debiasing_sba']} (dibaca dari source _croston_sba: out['mean'] *= 0.95)",
        f"croston_constructor: {params['constructor']} — tidak ada parameter alpha, jadi alpha TIDAK di-override",
        f"croston_ses_awal  : 'fitted[0] = x[0]' ada di source _ses_forecast: {params['ses_fitted_baris_pertama']}",
        "croston_verifikasi: nilai di atas dicek ulang saat runtime; kalau pustaka berubah, run ini gagal (SystemExit)",
    ]


def croston_rates(demand: np.ndarray, days: np.ndarray, first_day: int, last_day: int) -> np.ndarray:
    """Laju Croston-SBA untuk tiap hari di [first_day, last_day], memakai demand hari < t."""
    from statsforecast.models import CrostonSBA

    model = CrostonSBA()
    order = np.argsort(days)
    days_sorted = np.asarray(days)[order]
    y_sorted = np.asarray(demand, dtype=float)[order]
    out = np.empty(last_day - first_day + 1, dtype=float)
    for i, d in enumerate(range(first_day, last_day + 1)):
        hist = y_sorted[days_sorted < d]
        out[i] = float(model.forecast(y=hist, h=1)["mean"][0])
    return out


def run(config_path: Path, parquet: Path, sample_csv: Path, rf_sku_csv: Path | None = None) -> pd.DataFrame:
    import statsforecast

    cfg = yaml.safe_load(config_path.read_text())
    seed = int(cfg["seed"])
    results = Path(cfg["paths"]["results"])
    (results / "tables").mkdir(parents=True, exist_ok=True)

    params = croston_library_params()
    sample = pd.read_csv(sample_csv)
    sku_ids = sample["sku_id"].tolist()
    series = load_series(parquet, sku_ids)
    total_days = int(max(s["day"].max() for s in series.values()))
    train_end, val_end = split_bounds(total_days, int(cfg["split"]["test_days"]), int(cfg["split"]["validation_days"]))
    print(f"statsforecast {statsforecast.__version__}")
    print(f"SKU: {len(sku_ids)} | latih+validasi <= {val_end}, uji {val_end + 1}..{total_days}")
    print(
        f"Croston-SBA (parameter aktual dari source pustaka): alpha SES {params['alpha_ses']}, "
        f"faktor debiasing {params['faktor_debiasing_sba']}, konstruktor {params['constructor']}"
    )
    print("walk-forward, re-estimasi tiap hari\n")

    t_start = time.time()
    val_rows: list[dict] = []
    test_rows: list[dict] = []
    summary: list[dict] = []

    for i, sku in enumerate(sku_ids, start=1):
        t0 = time.time()
        s = series[sku]
        demand = s["demand"].to_numpy(dtype=float)
        days = s["day"].to_numpy()
        dates = dict(zip(days.tolist(), pd.to_datetime(s["date"]).dt.strftime("%Y-%m-%d").tolist()))

        fc_val = croston_rates(demand, days, train_end + 1, val_end)
        fc_test = croston_rates(demand, days, val_end + 1, total_days)

        look_val = dict(zip(days.tolist(), demand.tolist()))
        for d, f in zip(range(train_end + 1, val_end + 1), fc_val):
            val_rows.append(
                {"sku_id": sku, "day": d, "date": dates[d], "demand": look_val[d], "forecast": float(f),
                 "residual": float(look_val[d]) - float(f)}
            )
        for d, f in zip(range(val_end + 1, total_days + 1), fc_test):
            test_rows.append(
                {"sku_id": sku, "day": d, "date": dates[d], "demand": look_val[d], "forecast": float(f)}
            )

        mae_val = float(np.mean(np.abs(fc_val - np.array([look_val[d] for d in range(train_end + 1, val_end + 1)]))))
        mae_test = float(np.mean(np.abs(fc_test - np.array([look_val[d] for d in range(val_end + 1, total_days + 1)]))))
        resid_val = np.array([look_val[d] for d in range(train_end + 1, val_end + 1)]) - fc_val
        summary.append(
            {
                "sku_id": sku,
                "mae_val": mae_val,
                "mae_test": mae_test,
                "sigma_resid_val": float(resid_val.std(ddof=1)),
                "mean_resid_val": float(resid_val.mean()),
                "rate_akhir": float(fc_test[-1]),
                "n_demand_nol_uji": int((fc_test == 0).sum()),
                "n_test": int(fc_test.size),
                "seconds": round(time.time() - t0, 1),
            }
        )
        if i % 10 == 0 or i == len(sku_ids):
            print(f"  [{i:2d}/{len(sku_ids)}] {sku:24s} MAE val {mae_val:.4f} | MAE uji {mae_test:.4f} | {time.time() - t0:.0f}s")

    resid_df = pd.DataFrame(val_rows)[["sku_id", "day", "date", "demand", "forecast", "residual"]]
    fc_df = pd.DataFrame(test_rows)[["sku_id", "day", "date", "demand", "forecast"]]
    summ = pd.DataFrame(summary)

    resid_df.to_csv(results / "tables" / "croston_val_residuals.csv", index=False)
    fc_df.to_csv(results / "tables" / "croston_test_forecast.csv", index=False)
    summ.to_csv(results / "tables" / "croston_summary.csv", index=False)

    if rf_sku_csv is not None and rf_sku_csv.exists():
        rf = pd.read_csv(rf_sku_csv)[["sku_id", "mae_val", "mae_test"]].rename(
            columns={"mae_val": "mae_val_rf", "mae_test": "mae_test_rf"}
        )
        cmp = summ[["sku_id", "mae_val", "mae_test"]].rename(
            columns={"mae_val": "mae_val_croston", "mae_test": "mae_test_croston"}
        ).merge(rf, on="sku_id", how="left")
        cmp["selisih_mae_uji_rf_minus_croston"] = cmp["mae_test_rf"] - cmp["mae_test_croston"]
        cmp["rf_lebih_baik"] = cmp["selisih_mae_uji_rf_minus_croston"] < 0
        cmp.to_csv(results / "tables" / "mae_comparison.csv", index=False)
        print(
            f"\nMAE uji per SKU median: RF (global) {cmp['mae_test_rf'].median():.4f} vs "
            f"Croston {cmp['mae_test_croston'].median():.4f}"
        )
        print(f"SKU di mana RF lebih baik (MAE): {int(cmp['rf_lebih_baik'].sum())}/{len(cmp)}")
    else:
        print(f"\n[peringatan] {rf_sku_csv} tidak ada — mae_comparison.csv dilewati")

    dur = time.time() - t_start
    print(f"\nM4 selesai dalam {dur:.1f} s")
    print("Ringkas Croston-SBA per SKU (MAE validasi / MAE uji / sigma residual):")
    print(summ[["sku_id", "mae_val", "mae_test", "sigma_resid_val"]].to_string(index=False))

    extra = "\n".join(
        [
            f"skema split   : latih+validasi <= {val_end}, uji {val_end + 1}..{total_days}",
            f"sku_diuji     : {len(sku_ids)} (Croston TETAP per SKU, spec v2 5.6)",
            f"walk-forward  : re-estimasi tiap hari, {len(sku_ids) * (val_end - train_end + total_days - val_end)} forecast satu-langkah",
            f"mae_val_median: {float(summ['mae_val'].median()):.6f}",
            f"mae_test_median: {float(summ['mae_test'].median()):.6f}",
            f"durasi        : {dur:.1f} s",
        ]
        + croston_params_log(params)
    )
    write_run_log(results / "run_log.txt", "M4 model_croston Croston-SBA (spec 5.6)", seed, t_start, extra=extra)
    return summ


def main() -> None:
    ap = argparse.ArgumentParser(description="Croston-SBA walk-forward (M4).")
    ap.add_argument("--config", type=Path, default=Path("config.yaml"))
    ap.add_argument("--parquet", type=Path, default=None)
    ap.add_argument("--sample", type=Path, default=None)
    ap.add_argument("--rf-sku", type=Path, default=None, help="tabel MAE per SKU dari M3 (rf_test_mae_per_sku.csv)")
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text())
    parquet = a.parquet or (Path(cfg["paths"]["data_interim"]) / "sales_long.parquet")
    sample = a.sample or (Path(cfg["paths"]["results"]) / "tables" / "sku_sample.csv")
    rf_sku = a.rf_sku or (Path(cfg["paths"]["results"]) / "tables" / "rf_test_mae_per_sku.csv")
    run(a.config, parquet, sample, rf_sku)


if __name__ == "__main__":
    main()
