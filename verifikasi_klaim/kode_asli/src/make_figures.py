"""5.10 Analisis sensitivitas: gambar untuk artikel.

Menghasilkan (results/figures/):
- cost_comparison_bar.png    biaya total per rasio biaya: RF vs Croston-SBA (median antar SKU)
- sensitivity_lineplot.png   median biaya total vs rasio Cs:Ch + % SKU di mana RF lebih murah
- contoh_sku_<sku>.png       2-3 SKU contoh: demand aktual + forecast RF + forecast Croston

Pemilihan 3 SKU contoh deterministik: SKU dengan total demand uji paling kecil, paling besar,
dan yang berada di tengah (median) — masing-masing mewakili tingkat kelangkaan yang berbeda.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # tanpa display (headless/Kaggle)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

COLOR = {"rf": "#1C5AEE", "croston": "#EE1C25"}
LABEL = {"rf": "Random Forest", "croston": "Croston-SBA"}
N_CONTOH = 3


def _pick_contoh(fc_rf: pd.DataFrame, n: int = N_CONTOH) -> list[str]:
    total = fc_rf.groupby("sku_id")["demand"].sum().sort_values()
    if len(total) < n:
        return [str(s) for s in total.index.tolist()]
    picks = [total.index[0], total.index[len(total) // 2], total.index[-1]]
    return [str(s) for s in picks[:n]]


def run(config_path: Path, results_dir: Path | None = None) -> list[Path]:
    cfg = yaml.safe_load(config_path.read_text())
    results = Path(results_dir or cfg["paths"]["results"])
    figs = results / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    cost = pd.read_csv(results / "tables" / "cost_summary.csv")
    wil = pd.read_csv(results / "tables" / "wilcoxon_results.csv")
    made: list[Path] = []

    ratios = sorted(cost["ratio_cs_ch"].unique())
    med = cost.groupby(["ratio_cs_ch", "model"])["cost_total"].median().unstack()

    print("median biaya total per SKU (periode uji 180 hari):")
    print(med.rename(columns=LABEL).to_string())

    # --- 1. batang perbandingan biaya ---
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(len(ratios))
    w = 0.38
    for i, model in enumerate(["rf", "croston"]):
        vals = [med.loc[r, model] for r in ratios]
        bars = ax.bar(x + (i - 0.5) * w, vals, w, label=LABEL[model], color=COLOR[model])
        ax.bar_label(bars, fmt="%.0f", padding=2, fontsize=8)
    ax.set_xticks(x, [f"{r:.0f}:1" for r in ratios])
    ax.set_xlabel("Rasio biaya stockout : holding (Cs:Ch)")
    ax.set_ylabel("Median biaya total per SKU")
    ax.set_title("Biaya persediaan per SKU pada periode uji (order-up-to, R=1, L=7)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    p = figs / "cost_comparison_bar.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(p)

    # --- 2. sensitivitas ---
    win_pct = []
    for r in ratios:
        sub = cost[cost["ratio_cs_ch"] == r].pivot_table(index="sku_id", columns="model", values="cost_total")
        win_pct.append(100.0 * float((sub["rf"] < sub["croston"]).mean()))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for model in ["rf", "croston"]:
        ax1.plot(ratios, [med.loc[r, model] for r in ratios], marker="o", color=COLOR[model], label=LABEL[model])
    ax1.set_xlabel("Rasio Cs:Ch")
    ax1.set_ylabel("Median biaya total per SKU")
    ax1.set_title("Median biaya vs rasio biaya")
    ax1.grid(alpha=0.25)
    ax1.legend(frameon=False)
    ax1.set_xticks(ratios, [f"{r:.0f}:1" for r in ratios])

    ax2.plot(ratios, win_pct, marker="s", color="#0E7C4A")
    ax2.axhline(50, ls="--", lw=1, color="#888888")
    ax2.text(ratios[0], 51, "50% (imbang)", fontsize=8, color="#666666")
    ax2.set_ylim(0, 100)
    ax2.set_xlabel("Rasio Cs:Ch")
    ax2.set_ylabel("% SKU di mana RF lebih murah")
    ax2.set_title("Arah keunggulan saat rasio biaya naik")
    ax2.grid(alpha=0.25)
    ax2.set_xticks(ratios, [f"{r:.0f}:1" for r in ratios])
    fig.suptitle("Analisis sensitivitas biaya asimetris", y=1.02)
    fig.tight_layout()
    p = figs / "sensitivity_lineplot.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    made.append(p)

    if len(wil):
        print("\narah kemenangan per skenario:")
        for r in wil.itertuples():
            print(
                f"  Cs:Ch {r.ratio_cs_ch:.0f}:1 -> RF lebih murah di {r.rf_lebih_murah_n_sku}/{r.n_pairs} SKU, "
                f"p={r.p_value:.4g}"
            )

    # --- 3. feature importance RF global (spec v2 Bagian 6) ---
    imp = pd.read_csv(results / "tables" / "rf_feature_importance.csv").sort_values("importance", ascending=False)
    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    colors = ["#0E7C4A" if bool(v) else COLOR["rf"] for v in top["is_identity_feature"]]
    bars = ax.barh(top["feature"], top["importance"], color=colors)
    ax.bar_label(bars, fmt="%.4f", padding=2, fontsize=8)
    ax.set_xlabel("Kepentingan fitur (feature_importances_)")
    ax.set_title("Random Forest global: 15 fitur paling berpengaruh")
    handles = [
        Patch(facecolor=COLOR["rf"], label="deret waktu / kalender"),
        Patch(facecolor="#0E7C4A", label="identitas SKU (kategori/toko/state)"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    p = figs / "rf_feature_importance.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(p)

    # --- 4. contoh SKU ---
    rf_fc = pd.read_csv(results / "tables" / "rf_test_forecast.csv")
    cr_fc = pd.read_csv(results / "tables" / "croston_test_forecast.csv")
    for sku in _pick_contoh(rf_fc):
        r = rf_fc[rf_fc["sku_id"] == sku].sort_values("day")
        c = cr_fc[cr_fc["sku_id"] == sku].sort_values("day")
        fig, ax = plt.subplots(figsize=(8.6, 3.8))
        ax.bar(r["day"], r["demand"], color="#CCCCCC", width=0.9, label="Demand aktual")
        ax.plot(r["day"], r["forecast"], color=COLOR["rf"], lw=1.6, label=f"{LABEL['rf']} (MAE {np.mean(np.abs(r['demand'] - r['forecast'])):.3f})")
        ax.plot(c["day"], c["forecast"], color=COLOR["croston"], lw=1.6, ls="--", label=f"{LABEL['croston']} (MAE {np.mean(np.abs(c['demand'] - c['forecast'])):.3f})")
        hari_demand = int((r["demand"] > 0).sum())
        ax.set_title(f"{sku} — {hari_demand}/180 hari uji ber-demand (pola lumpy)")
        ax.set_xlabel("Hari (1762–1941 = periode uji)")
        ax.set_ylabel("Unit")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        p = figs / f"contoh_sku_{sku}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        made.append(p)

    print("\ngambar yang dibuat:")
    for p in made:
        print(f"  {p}  ({p.stat().st_size / 1e3:.0f} KB)")
    return made


def main() -> None:
    ap = argparse.ArgumentParser(description="Gambar analisis sensitivitas + contoh SKU (M5).")
    ap.add_argument("--config", type=Path, default=Path("config.yaml"))
    ap.add_argument("--results", type=Path, default=None)
    a = ap.parse_args()
    run(a.config, a.results)


if __name__ == "__main__":
    main()
