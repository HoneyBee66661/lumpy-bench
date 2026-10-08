
import numpy as np, pandas as pd
from scipy import stats
RF = "rf_global_direct#s42"

def page_test(tabel):
    n, k = tabel.shape
    per = np.apply_along_axis(lambda r: stats.rankdata(r), 1, tabel)
    T = per.sum(axis=0); L = float(np.sum(np.arange(1, k+1) * T))
    z = (L - n*k*(k+1)**2/4) / np.sqrt(n*k**2*(k+1)*(k**2-1)/144)
    return L, float(1 - stats.norm.cdf(z)), float(2*(1-stats.norm.cdf(abs(z))))

def boot_ci(x, n_boot=5000, seed=42):
    rng = np.random.default_rng(seed)
    med = np.median(rng.choice(x, size=(n_boot, len(x)), replace=True), axis=1)
    return float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))

baris = []
for o in (1, 2, 3, 4):
    d = pd.read_csv(f"results_v3/tables/service_equalized_o{o}.csv.gz")
    d = d[(d.model.isin([RF, "croston_sba"])) & (d.cs == 5.0)]
    w = d.pivot_table(index=["sku_id", "fill_target"], columns="model", values="cost_at_target", aggfunc="first")
    w = w[[RF, "croston_sba"]].dropna()
    selisih = w[RF] - w.croston_sba
    rel = selisih / w.croston_sba.replace(0.0, np.nan)
    t_abs = selisih.unstack("fill_target").dropna().sort_index(axis=1)
    t_rel = rel.unstack("fill_target").dropna().sort_index(axis=1)
    umum = sorted(set(t_abs.index) & set(t_rel.index))
    t_abs, t_rel = t_abs.loc[umum], t_rel.loc[umum]
    kol = list(t_abs.columns)
    for nama, t in (("absolut", t_abs), ("relatif", t_rel)):
        arr = t[kol].to_numpy(float)
        arr = arr[np.isfinite(arr).all(axis=1)]
        t = t.loc[t.index[np.isfinite(t[kol].to_numpy(float)).all(axis=1)]]
        d90, d95, d98 = arr[:, 0], arr[:, 1], arr[:, 2]
        delta = d98 - d90
        L1, p1, p2 = page_test(arr)
        lo, hi = boot_ci(delta)
        baris.append({"origin": o, "ukuran": nama, "n": len(delta),
                      "median_90": float(np.median(d90)), "median_95": float(np.median(d95)),
                      "median_98": float(np.median(d98)), "median_delta": float(np.median(delta)),
                      "CI95": f"[{lo:+.5f}, {hi:+.5f}]",
                      "p_kontras": float(stats.wilcoxon(delta, alternative="greater").pvalue),
                      "p_page": p1, "pct_delta_pos": round(100*float((delta > 0).mean()), 1)})
b = pd.DataFrame(baris)
pd.set_option("display.width", 260)
print(b.round(5).to_string(index=False))
print()
for nama in ("absolut", "relatif"):
    s = b[b.ukuran == nama]
    print(f"  {nama:8}: delta positif {int((s.median_delta > 0).sum())}/4 | kontras <5% {int((s.p_kontras<0.05).sum())}/4 | "
          f"Page <5% {int((s.p_page<0.05).sum())}/4 | median_delta range {s.median_delta.min():+.4f}..{s.median_delta.max():+.4f}")
    print(f"           CI memuat 0: {int(((s.median_delta>0) & (s.CI95.str.contains('-'))).sum())} kombinasi (lihat kolom CI95)")
b.to_csv("results_v3/2metode/hasil_h2_absolut_vs_relatif.csv", index=False)
print("\ntersimpan: results_v3/2metode/hasil_h2_absolut_vs_relatif.csv")
