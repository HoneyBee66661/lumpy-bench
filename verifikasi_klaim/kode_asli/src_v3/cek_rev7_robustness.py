"""cek_rev7_robustness.py — ketahanan hasil pada subset yang TETAP lumpy bila ADI/CV2
dihitung hanya pada irisan periode aktif + periode latih (hari <= 1221).

Menjawab butir 1 ronde 5: pipeline memilih sampel memakai periode aktif (penanda sell_price
tidak kosong), yang untuk sebagian SKU menjangkau jendela uji. Skrip ini mengulang uji pada
subset SKU yang klasifikasinya tidak berubah ketika pembatas periode latih diberlakukan,
tanpa melatih ulang apa pun (semua angka dibaca dari tabel v3 yang sudah diverifikasi).

Keluaran: results_v3/2metode/ROBUSTNESS_REV7.txt (+ .csv untuk sel H1)
Jalankan:  .venv/bin/python src_v3/cek_rev7_robustness.py
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RF, CR = "rf_global_direct#s42", "croston_sba"
CS = 5.0
ORIGINS = (1, 2, 3, 4)

buf = io.StringIO()


def P(*a, **kw):
    print(*a, **kw)
    print(*a, **kw, file=buf)


def page_test(tabel: np.ndarray):
    n, kk = tabel.shape
    per = np.apply_along_axis(lambda r: stats.rankdata(r), 1, tabel)
    T = per.sum(axis=0)
    L = float(np.sum(np.arange(1, kk + 1) * T))
    z = (L - n * kk * (kk + 1) ** 2 / 4) / np.sqrt(n * kk ** 2 * (kk + 1) * (kk ** 2 - 1) / 144)
    return L, float(1 - stats.norm.cdf(z))


def boot_ci_store(df: pd.DataFrame, val: str = "d", clus: str = "store_id",
                  n_boot: int = 5000, seed: int = 42):
    """CI bootstrap yang mengelompokkan SKU per toko (menjawab butir 6 ronde 2)."""
    rng = np.random.default_rng(seed)
    g = {k: v[val].to_numpy(float) for k, v in df.groupby(clus)}
    keys = list(g)
    out = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        out[i] = np.median(np.concatenate([g[keys[j]] for j in pick]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def holm(ps: np.ndarray) -> np.ndarray:
    order = np.argsort(ps)
    out = np.empty_like(ps, dtype=float)
    adj = np.maximum.accumulate((len(ps) - np.arange(len(ps))) * ps[order])
    out[order] = np.minimum(adj, 1.0)
    return out


def baca(o: int) -> pd.DataFrame:
    d = pd.read_csv(ROOT / f"results_v3/tables/service_equalized_o{o}.csv.gz")
    return d[d.model.isin([RF, CR]) & (d.cs == CS)]


# ------------------------------------------------------------------ himpunan subset
ir = pd.read_csv(ROOT / "results_v3/2metode/adi_cv2_irisan_prauji.csv")
sampel = pd.read_csv(ROOT / "results_v3/tables/sku_sample_v3.csv")
s = ir[ir.sku_id.isin(set(sampel.sku_id))]
ketat = set(s.loc[s.lumpy, "sku_id"])
P("=" * 78)
P("KETAHANAN: SUBSET LUMPY PADA IRISAN PERIODE AKTIF + PERIODE LATIH (hari <= 1221)")
P("=" * 78)
P(f"  SKU sampel 300 | tetap lumpy pada irisan: {len(ketat)} | tidak lagi lumpy: "
  f"{int((~s.lumpy).sum())} | tidak terhitung: {300 - len(s)}")

# ------------------------------------------------------------------ H1 pada subset
P("\nH1 (5:1, target 95%) pada subset:")
rows = []
for o in ORIGINS:
    d = baca(o)
    d = d[d.fill_target == 0.95]
    w = d.pivot_table(index="sku_id", columns="model", values="cost_at_target",
                      aggfunc="first").dropna()
    w = w.loc[w.index.intersection(ketat)].copy()
    w["d"] = w[RF] - w[CR]
    w["store_id"] = d.drop_duplicates("sku_id").set_index("sku_id").store_id.reindex(w.index)
    p = float(stats.wilcoxon(w.d, alternative="less").pvalue)   # RF lebih murah = d < 0
    lo, hi = boot_ci_store(w)
    rows.append({"origin": o, "n_sku": len(w), "median_rf": float(w[RF].median()),
                 "median_cr": float(w[CR].median()), "median_diff": float(w.d.median()),
                 "p_one_sided": p, "ci95_toko_lo": lo, "ci95_toko_hi": hi,
                 "n_toko": int(w.store_id.nunique())})
h1 = pd.DataFrame(rows)
P(h1.round(4).to_string(index=False))
h1["p_holm"] = holm(h1.p_one_sided.to_numpy())
P(f"  Holm lintas 4 titik: p mentah terkecil {h1.p_one_sided.min():.4f} -> "
  f"{h1.p_holm.min():.4f} | lolos <5%: {int((h1.p_holm < 0.05).sum())}/4")
P(f"  arah RF lebih murah (median_diff < 0): {int((h1.median_diff < 0).sum())}/4")
h1.to_csv(ROOT / "results_v3/2metode/robustness_h1_subset_rev7.csv", index=False)

# ------------------------------------------------------------------ H2 pada subset
P("\nH2 (selisih 98% vs 90%) pada subset, absolut dan relatif:")
for nama, relatif in (("absolut", False), ("relatif (% biaya Croston)", True)):
    blok = []
    for o in ORIGINS:
        d = baca(o)
        w = d.pivot_table(index=["sku_id", "fill_target"], columns="model",
                          values="cost_at_target", aggfunc="first").dropna()
        sku = [x[0] for x in w.index]
        keep = pd.Series(sku, index=w.index).isin(ketat)
        w = w[keep]
        if relatif:
            w = w[w[CR] > 0]
            sel = (w[RF] - w[CR]).div(w[CR])
        else:
            sel = w[RF] - w[CR]
        t = sel.unstack("fill_target").dropna().sort_index(axis=1)
        if len(t.columns) < 3:
            continue
        arr = t[list(t.columns)].to_numpy(float)
        delta = arr[:, -1] - arr[:, 0]          # 0,98 vs 0,90
        _, p_page = page_test(arr)
        lo, hi = float(np.percentile([np.median(np.random.default_rng(i).choice(delta, len(delta), replace=True)) for i in range(2000)], 2.5)), \
                 float(np.percentile([np.median(np.random.default_rng(i).choice(delta, len(delta), replace=True)) for i in range(2000)], 97.5))
        blok.append({"titik": o, "n": len(delta), "median_d90": float(np.median(arr[:, 0])),
                     "median_d98": float(np.median(arr[:, -1])),
                     "median_delta": float(np.median(delta)), "CI95": f"[{lo:+.4f}, {hi:+.4f}]",
                     "p_kontras": float(stats.wilcoxon(delta, alternative="greater").pvalue),
                     "p_page": p_page, "arah_naik": bool(np.median(delta) > 0)})
    b = pd.DataFrame(blok)
    if len(b):
        P(f"  -- {nama}")
        P(b.round(5).to_string(index=False))
        P(f"     delta naik {int(b.arah_naik.sum())}/{len(b)} | kontras 1 sisi <5% "
          f"{int((b.p_kontras < 0.05).sum())}/{len(b)} | Page 1 sisi <5% "
          f"{int((b.p_page < 0.05).sum())}/{len(b)}")
        b.to_csv(ROOT / f"results_v3/2metode/robustness_h2_subset_rev7_{'rel' if relatif else 'abs'}.csv",
                 index=False)

out = ROOT / "results_v3/2metode/ROBUSTNESS_REV7.txt"
out.write_text(buf.getvalue())
print(f"\ntersimpan: {out}")
