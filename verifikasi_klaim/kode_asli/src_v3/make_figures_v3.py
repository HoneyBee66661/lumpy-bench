"""make_figures_v3.py — gambar untuk artikel v3 (dari tabel di results_v3, tidak ada angka hardcode).

Gambar yang dibuat (bila datanya ada di results_v3/tables):
  1. kurva biaya vs fill rate per model, satu panel per skenario biaya (inti perbandingan
     service-equalized: dua model dibandingkan pada tingkat layanan yang sama)
  2. sebaran selisih biaya berpasangan per SKU (RF vs pembanding) pada fill target utama
  3. pentingnya fitur RF (dari results_v3/artifacts/importance_*.csv)
  4. histogram selisih relatif biaya per SKU pada fill target utama

Pakai:  .venv/bin/python src_v3/make_figures_v3.py [--origin 4 --fill 0.95]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
WARNA = {
    "rf_global_direct": "#EE1C25",
    "croston_sba": "#1C5AEE",
    "croston_classic": "#7A9BC4",
    "croston_sba_optimized": "#0F3D91",
    "tsb": "#2E9E6B",
    "adida": "#C98A00",
    "imapa": "#8E44AD",
    "naive": "#999999",
    "ma7": "#666666",
    "ma28": "#333333",
    "ses": "#B5651D",
    "lightgbm_tweedie": "#E67E22",
    "rf_recursive": "#C0392B",
}


def warna(nama: str) -> str:
    dasar = nama.split("#")[0]
    return WARNA.get(dasar, "#888888")


def label(nama: str) -> str:
    dasar, _, seed = nama.partition("#")
    return f"{dasar} ({seed})" if seed else dasar


def fig_kurva(costs: pd.DataFrame, out: Path, fill_utama: float) -> None:
    skenario = sorted(costs.cs.unique())
    fig, axes = plt.subplots(1, len(skenario), figsize=(5.6 * len(skenario), 4.4), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, cs in zip(axes, skenario):
        g0 = costs[costs.cs == cs]
        for model, g in g0.groupby("model"):
            kur = g.groupby("fill_rate")["cost_total"].median().sort_index()
            ax.plot(kur.index, kur.values, marker="o", ms=3, lw=1.6, color=warna(model), label=label(model))
        ax.axvline(fill_utama, color="black", ls="--", lw=1, alpha=0.6)
        ax.set_title(f"Cs:Ch = {cs:.0f}:1")
        ax.set_xlabel("fill rate (fraksi demand terlayani)")
        ax.set_ylabel("biaya total median per SKU")
        ax.grid(alpha=0.3)
    axes[-1].legend(fontsize=7, ncol=1, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.suptitle("Kurva biaya vs tingkat layanan (median antar SKU) — perbandingan service-equalized", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  gambar:", out.name)


def fig_pasangan(eq: pd.DataFrame, out: Path, fill: float, model_a: str, model_b: str) -> None:
    g = eq[(np.isclose(eq.fill_target, fill)) & (eq.cs == eq.cs.min())]
    piv = g.pivot_table(index="sku_id", columns="model", values="cost_at_target")
    ka = next((c for c in piv.columns if c.split("#")[0] == model_a), None)
    kb = next((c for c in piv.columns if c.split("#")[0] == model_b), None)
    if ka is None or kb is None:
        print(f"  lewati gambar pasangan: model {model_a}/{model_b} tidak ada")
        return
    d = piv[[ka, kb]].dropna()
    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    warna_titik = np.where(d[ka] < d[kb], "#EE1C25", "#1C5AEE")
    ax.scatter(d[kb], d[ka], s=16, c=warna_titik, alpha=0.8, edgecolors="none")
    lim = [0, float(max(d[ka].max(), d[kb].max())) * 1.05]
    ax.plot(lim, lim, color="black", lw=1, ls="--", label="biaya sama")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(f"biaya {label(kb)} (fill {fill:.0%})")
    ax.set_ylabel(f"biaya {label(ka)} (fill {fill:.0%})")
    n_menang = int((d[ka] < d[kb]).sum())
    ax.set_title(f"Per SKU pada fill {fill:.0%} (Cs:Ch {d.shape[0] and eq.cs.min():.0f}:1)\n"
                 f"{label(ka)} lebih murah di {n_menang}/{len(d)} SKU")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("  gambar:", out.name)


def fig_importance(art: Path, out: Path) -> None:
    berkas = sorted(art.glob("importance_*.csv"))
    if not berkas:
        print("  lewati gambar importance: belum ada artefak")
        return
    imp = pd.read_csv(berkas[-1]).head(12)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.barh(imp["fitur"][::-1], imp["importance"][::-1], color="#1C5AEE")
    ax.set_xlabel("importance (impurity, RF global)")
    ax.set_title(f"Pentingnya fitur — {berkas[-1].stem}")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("  gambar:", out.name)


def fig_sebaran(eq: pd.DataFrame, out: Path, fill: float, model_a: str, model_b: str) -> None:
    g = eq[(np.isclose(eq.fill_target, fill)) & (eq.cs == eq.cs.min())]
    piv = g.pivot_table(index="sku_id", columns="model", values="cost_at_target")
    ka = next((c for c in piv.columns if c.split("#")[0] == model_a), None)
    kb = next((c for c in piv.columns if c.split("#")[0] == model_b), None)
    if ka is None or kb is None:
        return
    d = piv[[ka, kb]].dropna()
    rel = 100 * (d[ka] - d[kb]) / d[kb]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.hist(rel, bins=25, color="#EE1C25", alpha=0.75, edgecolor="white")
    ax.axvline(0, color="black", lw=1)
    ax.axvline(rel.median(), color="#1C5AEE", lw=2, label=f"median {rel.median():+.1f}%")
    ax.set_xlabel(f"selisih biaya relatif {label(ka)} vs {label(kb)} (%)  — negatif = RF lebih murah")
    ax.set_ylabel("jumlah SKU")
    ax.set_title(f"Sebaran selisih biaya per SKU pada fill {fill:.0%} (Cs:Ch {eq.cs.min():.0f}:1)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("  gambar:", out.name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "config_v3.yaml")
    ap.add_argument("--origin", type=int, default=4)
    ap.add_argument("--fill", type=float, default=None)
    ap.add_argument("--model-a", default="rf_global_direct")
    ap.add_argument("--model-b", default="croston_sba")
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text())
    res = ROOT / cfg["paths"]["results"]
    figs = res / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    fill = float(a.fill if a.fill is not None else cfg["policy_v3"]["fill_rate_targets"][1])

    costs_path = res / "tables" / f"costs_o{a.origin}.csv.gz"
    eq_path = res / "tables" / f"service_equalized_o{a.origin}.csv.gz"
    if costs_path.exists():
        fig_kurva(pd.read_csv(costs_path), figs / f"fig_kurva_biaya_fill_o{a.origin}.png", fill)
    if eq_path.exists():
        eq = pd.read_csv(eq_path)
        fig_pasangan(eq, figs / f"fig_biaya_pasangan_o{a.origin}_fill{int(fill * 100)}.png", fill, a.model_a, a.model_b)
        fig_sebaran(eq, figs / f"fig_sebaran_selisih_o{a.origin}_fill{int(fill * 100)}.png", fill, a.model_a, a.model_b)
    fig_importance(res / "artifacts", figs / "fig_importance_rf.png")
    print("selesai ->", figs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
