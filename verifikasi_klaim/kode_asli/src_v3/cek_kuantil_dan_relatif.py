
import numpy as np, pandas as pd
from scipy import stats
pd.set_option("display.width", 200)
RF = "rf_global_direct#s42"

print("== B4/B3: kuantil yang menghasilkan tiap target, jarak grid, dan monotonisitas ==")
baris, nonmono = [], 0
tot_kurva = 0
for o in (1, 2, 3, 4):
    sw = pd.read_csv(f"results_v3/tables/policy_sweep_o{o}.csv.gz")
    sw = sw[sw.model.isin([RF, "croston_sba"])]
    for (sku, m), g in sw.groupby(["sku_id", "model"], observed=True):
        g = g.sort_values("tau")
        tot_kurva += 1
        fr = g.fill_rate.to_numpy(float)
        if np.any(np.diff(fr) < -1e-12):
            nonmono += 1
        for target in (0.90, 0.95, 0.98):
            i = int(np.argmin(np.abs(fr - target)))
            dekat = fr[i]
            # titik grid yang mengurung target
            atas = np.where(fr >= target)[0]
            bawah = np.where(fr <= target)[0]
            if len(atas) and len(bawah):
                lo, hi = bawah[-1], atas[0]
                lebar = float(fr[hi] - fr[lo])
            else:
                lebar = np.nan
            baris.append({"origin": o, "model": m, "fill_target": target,
                          "tau_terdekat": float(g.tau.to_numpy()[i]),
                          "fill_terdekat": float(dekat),
                          "selisih_ke_target": float(abs(dekat - target)),
                          "lebar_selang_grid": lebar})
b = pd.DataFrame(baris)
print(f"\nkurva SKU x model diperiksa: {tot_kurva:,} | non-monoton (fill turun minimal sekali): "
      f"{nonmono:,} ({100*nonmono/tot_kurva:.1f}%)")
print("\nmedian tau yang menghasilkan tiap target:")
print(b.groupby(["model", "fill_target"]).tau_terdekat.median().unstack().round(3).to_string())
print("\nmedian |fill - target| pada titik terdekat:")
print(b.groupby(["model", "fill_target"]).selisih_ke_target.median().round(4).unstack().to_string())
print("\nmedian lebar selang grid yang mengurung target:")
print(b.groupby(["model", "fill_target"]).lebar_selang_grid.median().round(4).unstack().to_string())
print("\nberapa persen kasus target jatuh di ujung grid (tidak ada titik di atas target):")
print(b.assign(tanpa_atas=b.lebar_selang_grid.isna()).groupby(["model", "fill_target"]).tanpa_atas.mean().round(3).unstack().to_string())

print("\n== B2: H2 dengan selisih RELATIF (% biaya Croston) ==")
def page_test(tabel):
    n, k = tabel.shape
    per = np.apply_along_axis(lambda r: stats.rankdata(r), 1, tabel)
    T = per.sum(axis=0); L = float(np.sum(np.arange(1, k+1) * T))
    z = (L - n*k*(k+1)**2/4) / np.sqrt(n*k**2*(k+1)*(k**2-1)/144)
    return L, float(1 - stats.norm.cdf(z))

hasil = []
for o in (1, 2, 3, 4):
    d = pd.read_csv(f"results_v3/tables/service_equalized_o{o}.csv.gz")
    d = d[(d.model.isin([RF, "croston_sba"])) & (d.cs == 5.0)]
    w = d.pivot_table(index=["sku_id", "fill_target"], columns="model", values="cost_at_target", aggfunc="first").dropna()
    abs_ = (w[RF] - w.croston_sba).unstack("fill_target")
    rel = ((w[RF] - w.croston_sba) / w.croston_sba).unstack("fill_target")
    for nama, t in (("absolut", abs_), ("relatif_%", rel)):
        kol = sorted(t.columns)
        d90, d98 = t[kol[0]].to_numpy(float), t[kol[-1]].to_numpy(float)
        delta = d98 - d90
        if len(kol) == 3:
            L, p_page = page_test(t[kol].to_numpy(float))
        else:
            L, p_page = np.nan, np.nan
        hasil.append({"origin": o, "ukuran": nama, "n": len(t),
                      "median_d90": float(np.median(d90)), "median_d98": float(np.median(d98)),
                      "median_delta": float(np.median(delta)),
                      "p_kontras_1sisi": float(stats.wilcoxon(delta, alternative="greater").pvalue),
                      "p_page_1sisi": p_page})
h = pd.DataFrame(hasil)
print(h.round(4).to_string(index=False))
for nama in ("absolut", "relatif_%"):
    sub = h[h.ukuran == nama]
    print(f"  {nama:9}: delta positif {int((sub.median_delta > 0).sum())}/{len(sub)} | "
          f"kontras signifikan {int((sub.p_kontras_1sisi < 0.05).sum())}/{len(sub)} | "
          f"Page signifikan {int((sub.p_page_1sisi < 0.05).sum())}/{len(sub)}")
