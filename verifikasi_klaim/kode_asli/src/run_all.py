"""Orkestrasi pipeline (spec Bagian 3): semua tahap dari satu titik, memakai config.yaml.

    .venv/bin/python src/run_all.py --stages m5        # simulasi + Wilcoxon + gambar
    .venv/bin/python src/run_all.py --stages all       # M1 s.d. M5 berurutan

Tahap yang tersedia: m1 (data_prep), m2 (classify_lumpy), m3 (model_rf), m4 (model_croston),
m5 (simulate_inventory + stats_tests + make_figures).

Catatan lingkungan: run M2/M3 penuh untuk 59 juta baris lebih baik dijalankan di Kaggle
(RAM host ini 2 GB) — lihat README, bagian "M2 di Kaggle Notebook". run_all.py tetap
memanggilnya supaya alur end-to-end bisa direproduksi di mesin yang memadai.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

STAGES = ["m1", "m2", "m3", "m4", "m5", "m6"]


def run_stage(name: str, config: Path, parquet: Path, results: Path, sample: Path) -> float:
    t0 = time.time()
    print(f"\n=== tahap {name.upper()} ===")
    if name == "m1":
        import data_prep

        cfg_path = config
        import yaml

        cfg = yaml.safe_load(config.read_text())
        data_prep.run(Path(cfg["paths"]["data_raw"]), Path(cfg["paths"]["data_interim"]), chunk_rows=300)
    elif name == "m2":
        import classify_lumpy

        classify_lumpy.run(config, batch_rows=500_000)
    elif name == "m3":
        import model_rf

        model_rf.run(config, parquet, sample)
    elif name == "m4":
        import model_croston

        model_croston.run(config, parquet, sample, results / "tables" / "rf_test_mae_per_sku.csv")
    elif name == "m5":
        import make_figures
        import pandas as pd
        import runlog
        import simulate_inventory
        import stats_tests
        import yaml

        t_sim = time.time()
        summarize = simulate_inventory.run(config, results)
        dur_sim = time.time() - t_sim
        t_st = time.time()
        wil = stats_tests.run(config, results)
        dur_st = time.time() - t_st
        t_fig = time.time()
        figs = make_figures.run(config, results)
        dur_fig = time.time() - t_fig
        cfg = yaml.safe_load(config.read_text())
        extra = "\n".join(
            [
                f"durasi_simulasi_s : {dur_sim:.1f}",
                f"durasi_wilcoxon_s : {dur_st:.1f}",
                f"durasi_figur_s    : {dur_fig:.1f}",
                f"kebijakan         : order-up-to, R={cfg['inventory_policy']['review_period_days']}, "
                f"L={cfg['inventory_policy']['lead_time_days']}, backorder=ya",
                "skenario          : " + "; ".join(f"{cs}:{ch}" for cs, ch in cfg["cost_scenarios"]),
                f"sku x model       : {summarize['sku_id'].nunique()} SKU x {summarize['model'].nunique()} model "
                f"x {summarize['ratio_cs_ch'].nunique()} skenario = {len(summarize)} baris biaya",
                f"gambar            : {len(figs)} berkas",
                "selisih biaya     : cost_summary.csv memuat DUA kolom selisih terpisah — "
                "selisih_median_berpasangan_per_sku (per SKU dulu) dan selisih_median_agregat (median RF - median Croston)",
            ]
            + [
                f"skenario {r.ratio_cs_ch:.0f}:1 : median biaya RF {r.median_cost_rf:,.1f} vs Croston "
                f"{r.median_cost_croston:,.1f}; selisih berpasangan {r.selisih_median_berpasangan_per_sku:,.1f}; "
                f"selisih agregat {r.selisih_median_agregat:,.1f}; W={r.statistic:.0f} p={r.p_value:.4g} "
                f"({'signifikan' if r.signifikan_5pct else 'tidak signifikan'}); r_rb={r.rank_biserial_r:+.3f} "
                f"(efek {r.ukuran_efek})"
                for r in wil.itertuples()
            ]
        )
        runlog.write(results / "run_log.txt", "M5 simulasi inventori + biaya + Wilcoxon + sensitivitas (spec 5.7-5.10)", int(cfg["seed"]), t0, extra=extra)
    elif name == "m6":
        import subprocess

        rc = subprocess.call(
            [sys.executable, str(ROOT / "collect_results.py"), "--config", str(config), "--results", str(results)]
        )
        if rc != 0:
            raise SystemExit(f"M6: ada artefak wajib yang hilang/kolom kurang (exit {rc})")
    else:
        raise SystemExit(f"tahap tidak dikenal: {name}")
    dur = time.time() - t0
    print(f"tahap {name.upper()} selesai dalam {dur:.1f} s")
    return dur


def main() -> None:
    ap = argparse.ArgumentParser(description="Orkestrasi pipeline RF vs Croston.")
    ap.add_argument("--config", type=Path, default=Path("config.yaml"))
    ap.add_argument("--stages", default="all", help="all | m1 | m2 | m3 | m4 | m5 (boleh dipisah koma)")
    a = ap.parse_args()

    import yaml

    cfg = yaml.safe_load(a.config.read_text())
    results = Path(cfg["paths"]["results"])
    parquet = Path(cfg["paths"]["data_interim"]) / "sales_long.parquet"
    sample = results / "tables" / "sku_sample.csv"

    stages = STAGES if a.stages == "all" else [s.strip().lower() for s in a.stages.split(",")]
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise SystemExit(f"tahap tidak dikenal: {unknown} (pilihan: {STAGES})")

    total = 0.0
    for stage in stages:
        total += run_stage(stage, a.config, parquet, results, sample)
    print(f"\nsemua tahap ({', '.join(stages)}) selesai dalam {total:.1f} s")


if __name__ == "__main__":
    main()
