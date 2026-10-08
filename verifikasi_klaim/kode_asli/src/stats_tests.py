"""5.9 Uji statistik: Wilcoxon signed-rank berpasangan per skenario rasio biaya + ukuran efek.

Untuk tiap skenario (Cs, Ch) di config.yaml: uji berpasangan per SKU antara biaya total
Random Forest dan Croston-SBA.

    scipy.stats.wilcoxon(cost_rf, cost_croston, alternative="two-sided")

Ukuran efek (spec v2 5.9): rank-biserial correlation dari statistik Wilcoxon

    r = 1 - 2W / (n(n+1))

Konvensi arah — DICEK EMPIRIS terhadap data, bukan diasumsikan: `scipy.stats.wilcoxon` mengembalikan
W = jumlah peringkat selisih POSITIF (W+) setelah selisih nol dibuang (diverifikasi di run v2:
5:1 -> W+ = 220, W- = 600, statistic = 220). Karena itu:

  - r POSITIF  -> W+ kecil -> selisih negatif (RF lebih murah) mendominasi  => RF cenderung LEBIH MURAH
  - r NEGATIF  -> W+ besar -> RF cenderung LEBIH MAHAL
  - |r| 0,10/0,30/0,50 sebagai ambang efek kecil/sedang/besar (disimpan sebagai kolom label, bukan
    interpretasi naratif)

Keluaran: results/tables/wilcoxon_results.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats


def rank_biserial(w: float, n: int) -> float:
    """r = 1 - 2W / (n(n+1)) — ukuran efek untuk Wilcoxon signed-rank."""
    if n <= 0:
        return float("nan")
    return float(1.0 - (2.0 * float(w)) / (n * (n + 1)))


def effect_label(abs_r: float) -> str:
    if not np.isfinite(abs_r):
        return "tidak terdefinisi"
    if abs_r < 0.10:
        return "dapat diabaikan"
    if abs_r < 0.30:
        return "kecil"
    if abs_r < 0.50:
        return "sedang"
    return "besar"


def run(config_path: Path, results_dir: Path | None = None) -> pd.DataFrame:
    cfg = yaml.safe_load(config_path.read_text())
    results = Path(results_dir or cfg["paths"]["results"])
    cost = pd.read_csv(results / "tables" / "cost_summary.csv")

    rows: list[dict] = []
    for (cs, ch), sub in cost.groupby(["cs", "ch"], sort=True):
        piv = sub.pivot_table(index="sku_id", columns="model", values="cost_total")
        a = piv["rf"].to_numpy(dtype=float)
        b = piv["croston"].to_numpy(dtype=float)
        diff = a - b
        n = len(piv)
        test = stats.wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
        w = float(test.statistic)
        r_rb = rank_biserial(w, n)
        rows.append(
            {
                "cs": float(cs),
                "ch": float(ch),
                "ratio_cs_ch": float(cs) / float(ch),
                "n_pairs": int(n),
                "n_zero_diff": int((diff == 0).sum()),
                "median_cost_rf": float(np.median(a)),
                "median_cost_croston": float(np.median(b)),
                # selisih dua jenis, dihitung ulang di sini supaya wilcoxon_results.csv mandiri
                "selisih_median_berpasangan_per_sku": float(np.median(diff)),
                "selisih_median_agregat": float(np.median(a) - np.median(b)),
                "mean_diff_rf_minus_croston": float(diff.mean()),
                "rf_lebih_murah_n_sku": int((diff < 0).sum()),
                "croston_lebih_murah_n_sku": int((diff > 0).sum()),
                "statistic": w,
                "p_value": float(test.pvalue),
                "signifikan_5pct": bool(test.pvalue < 0.05),
                "rank_biserial_r": r_rb,
                "rank_biserial_abs": abs(r_rb),
                "ukuran_efek": effect_label(abs(r_rb)),
            }
        )

    out = pd.DataFrame(rows).sort_values("ratio_cs_ch").reset_index(drop=True)
    out.to_csv(results / "tables" / "wilcoxon_results.csv", index=False)

    for r in out.itertuples():
        arah = "RF lebih murah" if r.selisih_median_berpasangan_per_sku < 0 else "Croston lebih murah"
        print(
            f"  Cs:Ch {r.ratio_cs_ch:.0f}:1 | n={r.n_pairs} | median selisih berpasangan "
            f"{r.selisih_median_berpasangan_per_sku:,.1f} (agregat {r.selisih_median_agregat:,.1f}) -> {arah} | "
            f"W={r.statistic:.0f} p={r.p_value:.4g} {'(signifikan)' if r.signifikan_5pct else '(tidak signifikan)'} | "
            f"r_rb={r.rank_biserial_r:+.3f} (efek {r.ukuran_efek})"
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Wilcoxon signed-rank + ukuran efek per skenario biaya (M5).")
    ap.add_argument("--config", type=Path, default=Path("config.yaml"))
    ap.add_argument("--results", type=Path, default=None)
    a = ap.parse_args()
    run(a.config, a.results)


if __name__ == "__main__":
    main()
