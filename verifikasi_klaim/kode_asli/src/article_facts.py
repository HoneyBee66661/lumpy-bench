#!/usr/bin/env python3
"""Ringkas SEMUA angka kunci dari results/ ke satu file datar (spec v2 Bagian 6, opsional tapi disarankan).

Tujuannya: saat bagian Hasil/Pembahasan ditulis di chat, angkanya tinggal disalin dari
results/article_facts.txt — tidak ada yang perlu dihitung ulang atau dikutip dari ingatan.

Isi file HANYA angka + label (spec v2 Bagian 7: jangan tulis narasi/kesimpulan di sini).

Jalankan:  .venv/bin/python src/article_facts.py
Keluaran:  cetak ke layar + results/article_facts.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from model_rf import split_bounds  # noqa: E402
from classify_lumpy import MIN_NONZERO_DAYS  # noqa: E402

TOTAL_DAYS = 1941
BARIS_M1 = 59_181_090


def f(x: float, nd: int = 4) -> str:
    """Format angka gaya Indonesia: pemisah ribuan '.' dan desimal ','."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)) or pd.isna(x):
        return "NA"
    s = f"{float(x):,.{nd}f}"
    return s.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("config.yaml"))
    ap.add_argument("--results", type=Path, default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text())
    res = Path(a.results or cfg["paths"]["results"])
    tabs = res / "tables"
    out: list[str] = []

    def p(line: str = "") -> None:
        print(line)
        out.append(line)

    def h(title: str) -> None:
        p("")
        p(f"=== {title} ===")

    seed = int(cfg["seed"])
    test_days = int(cfg["split"]["test_days"])
    val_days = int(cfg["split"]["validation_days"])
    train_end, val_end = split_bounds(TOTAL_DAYS, test_days, val_days)

    def csv(name: str) -> pd.DataFrame:
        return pd.read_csv(tabs / name)

    missing: list[str] = []

    def need(name: str) -> pd.DataFrame | None:
        try:
            return csv(name)
        except FileNotFoundError:
            missing.append(name)
            return None

    h("DATA (M1)")
    p(f"baris parquet long            : {BARIS_M1:,}".replace(",", "."))
    p(f"total hari (versi evaluation) : {TOTAL_DAYS}")
    p(f"split waktu                   : latih 1-{train_end}, validasi {train_end + 1}-{val_end}, uji {val_end + 1}-{TOTAL_DAYS}")
    p(f"seed random_state             : {seed}")

    h("KLASIFIKASI & SAMPLING (M2)")
    stats = need("all_sku_stats.csv")
    sample = need("sku_sample.csv")
    lumpy = res.parent / "data/interim/lumpy_all.csv"
    adi_min = float(cfg["lumpy_threshold"]["adi_min"])
    cv2_min = float(cfg["lumpy_threshold"]["cv2_min"])
    p(f"ambang lumpy                  : ADI > {adi_min} dan CV2 > {cv2_min} (Syntetos dkk., 2005)")
    if stats is not None:
        p(f"SKU total                     : {len(stats):,}".replace(",", "."))
        two = int(((stats.adi > adi_min) & (stats.cv2 > cv2_min)).sum())
        p(f"SKU lolos 2 ambang saja       : {two:,}".replace(",", "."))
        p(f"kriteria penuh (dipakai)      : 2 ambang + n_nonzero >= {MIN_NONZERO_DAYS} hari ber-demand "
          f"(menyaring SKU dengan demand nyaris tidak ada)")
    if lumpy.exists():
        p(f"SKU lolos kriteria penuh      : {len(pd.read_csv(lumpy)):,}".replace(",", "."))
        p(f"catatan                       : angka ini (5.899) muncul dari kriteria penuh; kalau hanya 2 ambang "
          f"dipakai angkanya 5.900 — bedanya 1 SKU dengan 2 hari ber-demand. Jangan tertukar saat menulis.")
        p(f"periode perhitungan ADI/CV2   : hari 1-{val_end} untuk SEMUA SKU (n_days = 1761 seragam). "
          f"Konsekuensi: nol struktural/pra-peluncuran ikut menaikkan ADI; perlu dihitung ulang dari "
          f"periode aktif pada revisi berikutnya.")
    if sample is not None:
        p(f"SKU ter-sampling (n_sku)      : {len(sample)}")
        p(f"prosedur sampling             : sampel acak sederhana dari SKU yang lolos kriteria penuh, "
          f"seed {seed} (pandas .sample(n, random_state=seed); BUKAN stratified per kategori/toko)")
        p(f"komposisi sampel              : kategori " + ", ".join(f"{k}={v}" for k, v in sample["item_id"].str.split("_").str[0].value_counts().items())
          + f" | toko {sample['store_id'].str.split('_').str[0].nunique()} state | {sample['store_id'].nunique()} toko | "
          f"{sample['item_id'].str.split('_').str[:2].str.join('_').nunique()} departemen")
        p(f"ADI sampel                    : min {f(sample['adi'].min())} | median {f(sample['adi'].median())} | maks {f(sample['adi'].max())}")
        p(f"CV2 sampel                    : min {f(sample['cv2'].min())} | median {f(sample['cv2'].median())} | maks {f(sample['cv2'].max())}")
        p(f"n_nonzero sampel (hari demand): min {int(sample['n_nonzero'].min())} | median {int(sample['n_nonzero'].median())} | maks {int(sample['n_nonzero'].max())} dari {val_end} hari latih+validasi")
    fc = need("rf_test_forecast.csv")
    if fc is not None:
        per_sku_hari = fc.groupby("sku_id")["demand"].apply(lambda s: int((s > 0).sum()))
        p(f"hari uji ber-demand per sampel: min {int(per_sku_hari.min())}, median {int(per_sku_hari.median())}, maks {int(per_sku_hari.max())} dari {test_days}")

    h("RANDOM FOREST GLOBAL (M3, spec v2 5.5)")
    tune = need("rf_tuning.csv")
    best = need("rf_best_params.csv")
    if tune is not None:
        p(f"kombinasi grid                : {len(tune)} (dari config rf.*)")
        p(f"total fit                     : {len(tune)} (SATU fit per kombinasi untuk seluruh {len(sample) if sample is not None else 'n'} SKU; versi v1: 1.080 fit)")
        p(f"MAE validasi GABUNGAN         : min {f(tune['mae_val_pooled'].min())} | median {f(tune['mae_val_pooled'].median())} | maks {f(tune['mae_val_pooled'].max())}")
        top = tune.sort_values("mae_val_pooled").head(3)
        p("3 kombinasi terbaik (MAE validasi gabungan):")
        for r in top.itertuples():
            p(f"    n_estimators={r.n_estimators}, max_depth={'None' if pd.isna(r.max_depth) else int(r.max_depth)}, "
              f"min_samples_leaf={r.min_samples_leaf} -> {f(r.mae_val_pooled)} (median per SKU {f(r.mae_val_median_per_sku)})")
        leaf = tune.groupby("min_samples_leaf")["mae_val_pooled"].median()
        p("median MAE validasi gabungan per min_samples_leaf: " + " | ".join(f"leaf={int(k)}: {f(v)}" for k, v in leaf.items()))
        depth = tune.assign(dkey=tune["max_depth"].fillna(0)).groupby("dkey")["mae_val_pooled"].median()
        p("median MAE validasi gabungan per max_depth: " + " | ".join(
            f"depth={'tanpa batas' if int(k) == 0 else int(k)}: {f(v)}" for k, v in depth.items()))
    if best is not None:
        b = best.iloc[0]
        p(f"kombinasi terpilih            : n_estimators={int(b.n_estimators)}, max_depth={'None' if pd.isna(b.max_depth) else int(b.max_depth)}, min_samples_leaf={int(b.min_samples_leaf)}")
        p(f"jumlah fitur model            : {int(b.n_features)}")
        p(f"MAE validasi gabungan         : {f(b.mae_val_pooled)} (median per SKU {f(b.mae_val_median_per_sku)})")
        p(f"MAE uji gabungan              : {f(b.mae_test_pooled)}")
        p(f"MAE uji per SKU               : min {f(b.mae_test_min_per_sku)} | median {f(b.mae_test_median_per_sku)} | maks {f(b.mae_test_max_per_sku)}")
        p(f"sigma residual validasi per SKU: min {f(b.sigma_resid_val_min_per_sku)} | median {f(b.sigma_resid_val_median_per_sku)} | maks {f(b.sigma_resid_val_max_per_sku)}")
        p(f"jumlah baris latih/validasi/uji: {int(b.n_train_final):,} / {int(b.n_val):,} / {int(b.n_test):,}".replace(",", "."))
    imp = need("rf_feature_importance.csv")
    if imp is not None:
        p("5 fitur paling berpengaruh (feature_importances_):")
        for r in imp.sort_values("rank").head(5).itertuples():
            p(f"    {int(r.rank)}. {r.feature:16s} {r.importance:.4f}" + ("  [identitas SKU]" if bool(r.is_identity_feature) else ""))
        ident = imp[imp["is_identity_feature"]]
        if len(ident):
            p(f"total kepentingan fitur identitas: {f(ident['importance'].sum())} dari 1,0 ({100 * ident['importance'].sum():.2f}%)")
    psku = need("mae_per_sku_validation.csv")
    if psku is not None and "is_best_combo" in psku.columns:
        cb = psku[psku["is_best_combo"]]
        p(f"MAE validasi per SKU (kombinasi terbaik): paling timpang {f(cb['mae_val'].max())} vs paling baik {f(cb['mae_val'].min())} "
          f"(rasio {cb['mae_val'].max() / max(cb['mae_val'].min(), 1e-9):.1f}x)")

    h("CROSTON-SBA (M4)")
    cr = need("croston_summary.csv")
    if cr is not None:
        p(f"MAE validasi Croston          : min {f(cr['mae_val'].min())} | median {f(cr['mae_val'].median())} | maks {f(cr['mae_val'].max())}")
        p(f"MAE uji Croston               : min {f(cr['mae_test'].min())} | median {f(cr['mae_test'].median())} | maks {f(cr['mae_test'].max())}")
        p(f"sigma residual validasi Croston: min {f(cr['sigma_resid_val'].min())} | median {f(cr['sigma_resid_val'].median())} | maks {f(cr['sigma_resid_val'].max())}")
    cmpc = need("mae_comparison.csv")
    if cmpc is not None:
        d = cmpc["selisih_mae_uji_rf_minus_croston"]
        p(f"MAE uji median RF vs Croston  : {f(cmpc['mae_test_rf'].median())} vs {f(cmpc['mae_test_croston'].median())}")
        p(f"SKU di mana MAE RF < Croston  : {int(cmpc['rf_lebih_baik'].sum())} dari {len(cmpc)}")
        p(f"selisih MAE uji (RF - Croston): min {f(d.min())} | median {f(d.median())} | maks {f(d.max())}")
    rl = res / "run_log.txt"
    if rl.exists():
        par = [ln for ln in rl.read_text().splitlines() if ln.startswith(("croston_alpha_ses", "croston_sba_factor", "croston_constructor"))]
        p("parameter Croston aktual dari source pustaka (results/run_log.txt):")
        for ln in par:
            p(f"    {ln}")

    h("SIMULASI PERSEDIAAN & BIAYA (M5)")
    cost = need("cost_summary.csv")
    wil = need("wilcoxon_results.csv")
    pol = cfg["inventory_policy"]
    p(f"kebijakan                     : order-up-to, review R={int(pol['review_period_days'])} hari, lead time L={int(pol['lead_time_days'])} hari, backorder diperbolehkan")
    p(f"skenario biaya (Cs:Ch)        : " + ", ".join(f"{int(cs)}:{int(ch)}" for cs, ch in cfg["cost_scenarios"]))
    if cost is not None:
        for ratio in sorted(cost["ratio_cs_ch"].unique()):
            sub = cost[cost["ratio_cs_ch"] == ratio]
            piv_c = sub.pivot_table(index="sku_id", columns="model", values="cost_total")
            piv_s = sub.pivot_table(index="sku_id", columns="model", values="stockout_units_total")
            piv_h = sub.pivot_table(index="sku_id", columns="model", values="holding_units_total")
            piv_f = sub.pivot_table(index="sku_id", columns="model", values="fill_rate")
            p(f"\nrasio {ratio:.0f}:1")
            for model in ("rf", "croston"):
                c = piv_c[model]
                p(f"  {model:8s} biaya: median {f(c.median(), 1)} | mean {f(c.mean(), 1)} | min {f(c.min(), 1)} | maks {f(c.max(), 1)}")
                p(f"           stockout unit: median {f(piv_s[model].median(), 1)} | holding unit: median {f(piv_h[model].median(), 1)} | fill rate median {f(piv_f[model].median())}")
            row = sub.iloc[0]
            p(f"  RF lebih murah di           : {int((piv_c['rf'] < piv_c['croston']).sum())} dari {len(piv_c)} SKU")
            p(f"  selisih median BERPASANGAN per SKU (RF-Croston): {f(row['selisih_median_berpasangan_per_sku'], 1)}")
            p(f"  selisih MEDIAN AGREGAT (median RF - median Croston): {f(row['selisih_median_agregat'], 1)}")
            if wil is not None:
                w = wil[wil["ratio_cs_ch"] == ratio].iloc[0]
                p(f"  Wilcoxon                    : W={w['statistic']:.0f}, p={w['p_value']:.4f}, signifikan 5%: {bool(w['signifikan_5pct'])}")
                p(f"  ukuran efek                 : rank-biserial r={w['rank_biserial_r']:+.4f} (|r|={w['rank_biserial_abs']:.4f}, efek {w['ukuran_efek']})")
                p(f"  arah                        : RF lebih murah {int(w['rf_lebih_murah_n_sku'])} SKU, Croston lebih murah {int(w['croston_lebih_murah_n_sku'])} SKU")

    if wil is not None:
        h("UKURAN EFEK & KONVENSI (M5)")
        p("konvensi ukuran efek         : r = 1 - 2W/(n(n+1)) dengan W = jumlah peringkat selisih POSITIF (W+); "
          "r positif = RF cenderung LEBIH MURAH, r negatif = RF cenderung lebih mahal (dicek empiris di run v2)")
        for ratio in sorted(wil["ratio_cs_ch"].unique()):
            w = wil[wil["ratio_cs_ch"] == ratio].iloc[0]
            p(f"  rasio {ratio:.0f}:1 Wilcoxon    : W={w['statistic']:.0f}, p={w['p_value']:.4f}, "
              f"signifikan 5%: {bool(w['signifikan_5pct'])} | ukuran efek rank-biserial r={w['rank_biserial_r']:+.4f} "
              f"(|r|={w['rank_biserial_abs']:.4f}, efek {w['ukuran_efek']}) | RF lebih murah "
              f"{int(w['rf_lebih_murah_n_sku'])} SKU, Croston lebih murah {int(w['croston_lebih_murah_n_sku'])} SKU")

    h("RINGKASAN UNTUK HIPOTESIS")
    if wil is not None:
        w0 = wil.sort_values("ratio_cs_ch").iloc[0]
        p(f"H1 (biaya RF < Croston pada rasio dasar {cfg['cost_scenarios'][0][0]:.0f}:{cfg['cost_scenarios'][0][1]:.0f}): "
          f"selisih median berpasangan {f(w0['selisih_median_berpasangan_per_sku'], 1)}, "
          f"selisih median agregat {f(w0['selisih_median_agregat'], 1)}, p={w0['p_value']:.4f} -> "
          f"{'didukung' if (w0['selisih_median_berpasangan_per_sku'] < 0 and w0['signifikan_5pct']) else 'TIDAK didukung'}")
        diffs = ", ".join(f"{x:,.1f}".replace(",", ".") for x in wil["selisih_median_berpasangan_per_sku"])
        ps = ", ".join(f"{x:.4f}" for x in wil["p_value"])
        rs = ", ".join(f"{x:+.3f}" for x in wil["rank_biserial_r"])
        p(f"H2 (selisih berubah signifikan saat rasio dinaikkan): selisih median berpasangan ({diffs}), p ({ps}), r_rb ({rs}) -> "
          f"{'ada skenario signifikan' if bool(wil['signifikan_5pct'].any()) else 'TIDAK terbukti (tidak signifikan di semua rasio)'}")

    h("CATATAN METODE (untuk bagian Metode)")
    if cost is not None:
        p(f"baris trace harian            : {len(cost) * int(cost['n_days'].iloc[0]):,}".replace(",", ".") + f" ({len(cost)} baris biaya x {int(cost['n_days'].iloc[0])} hari)")
    p("sumber data                   : M5 Forecasting Accuracy (Walmart) — URL asal + SHA-256 data mentah ada di "
      "results/raw_sha256.txt (dan dirujuk di results/run_log.txt)")
    p("arsitektur model              : Random Forest GLOBAL (satu model untuk seluruh SKU, fitur identitas kategori/toko/state) vs Croston-SBA per SKU")
    p("lingkungan eksekusi           : Kaggle Notebooks CPU untuk M2-M5; verifikasi ulang di lokal (lihat tests/verify_kaggle_*.py)")
    if sample is not None:
        p(f"ukuran sampel                 : {len(sample)} SKU (spec meminta minimal 30 untuk uji Wilcoxon berpasangan)")
    p("parameter Croston             : dibaca dari source statsforecast saat runtime (lihat bagian CROSTON-SBA di atas)")

    text = "\n".join(out) + "\n"
    (res / "article_facts.txt").write_text(text)
    print(f"\n[disimpan ke {res / 'article_facts.txt'}]")
    if missing:
        print(f"[catatan] berkas yang belum ada: {missing}")


if __name__ == "__main__":
    main()
