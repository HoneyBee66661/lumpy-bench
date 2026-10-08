
import numpy as np, pandas as pd
from scipy import stats
RF = "rf_global_direct#s42"
pd.set_option("display.width", 240)

def holm(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p); adj = np.empty(m); run = 0.0
    for i, idx in enumerate(order):
        run = max(run, (m - i) * p[idx]); adj[idx] = min(run, 1.0)
    return adj

print("== kolom berkas kebijakan ==")
sw = pd.read_csv("results_v3/tables/policy_sweep_o1.csv.gz")
print(list(sw.columns))

print("\n== kenapa biaya kosong pada baris tercakup ==")
d = pd.read_csv("results_v3/tables/service_equalized_o1.csv.gz")
d = d[(d.model == RF) & (d.cs == 5.0)]
kosong = d[(d.fill_target == 0.95) & d.tercakup & d.cost_at_target.isna()].sku_id.unique()
print("SKU (titik 1, RF):", len(kosong), "| contoh:", list(kosong[:3]))
# PENGAMAN (ditambahkan untuk verifikasi skala kecil; tidak mengubah angka apa pun):
# blok diagnostik di bawah hanya relevan bila ada SKU dengan biaya kosong.
if len(kosong):
    s = sw[(sw.sku_id == kosong[0]) & (sw.model == RF)].sort_values("tau")
    kol = [c for c in sw.columns if c in ("tau", "fill_rate", "cost", "cost_total", "unmet", "on_hand", "backlog_hari", "harga")]
    print(s[kol].head(11).round(3).to_string(index=False))
    # apakah seluruh sweep SKU ini NaN pada kolom biaya?
    kandidat = [c for c in s.columns if "cost" in c.lower()]
    for c in kandidat:
        print(f"   kolom {c}: NaN {int(s[c].isna().sum())}/{len(s)}")
    print("   fill_rate NaN:", int(s.fill_rate.isna().sum()), "| demand_total kolom ada?", "demand" in s.columns)
else:
    print("   (tidak ada SKU dengan biaya kosong pada jalannya verifikasi ini — blok diagnostik dilewati)")

print("\n== (d) H1 ukuran RELATIF: Wilcoxon satu sisi + Holm lintas titik ==")
rows = []
for o in (1, 2, 3, 4):
    dd = pd.read_csv(f"results_v3/tables/service_equalized_o{o}.csv.gz")
    dd = dd[(dd.model.isin([RF, "croston_sba"])) & (dd.cs == 5.0) & (dd.fill_target == 0.95)]
    w = dd.pivot_table(index="sku_id", columns="model", values="cost_at_target", aggfunc="first").dropna()
    w = w[w.croston_sba != 0]
    rel = (w[RF] - w.croston_sba) / w.croston_sba
    rows.append({"titik": o, "n": len(rel), "median_rel": float(rel.median()),
                 "p_1sisi": float(stats.wilcoxon(rel, alternative="less").pvalue)})
h1 = pd.DataFrame(rows)
h1["p_holm_4titik"] = holm(h1.p_1sisi)
print(h1.round(6).to_string(index=False))
print(f"  signifikan mentah {int((h1.p_1sisi<0.05).sum())}/4 | setelah Holm {int((h1.p_holm_4titik<0.05).sum())}/4 | "
      f"arah RF lebih murah {int((h1.median_rel<0).sum())}/4")
h1.to_csv("results_v3/2metode/hasil_h1_relatif_rev8.csv", index=False)

print("\n== (e) H2 ukuran RELATIF: Page + kontras + Holm lintas titik ==")
def page_test(tabel):
    n, k = tabel.shape
    per = np.apply_along_axis(lambda r: stats.rankdata(r), 1, tabel)
    T = per.sum(axis=0); L = float(np.sum(np.arange(1, k+1) * T))
    z = (L - n*k*(k+1)**2/4) / np.sqrt(n*k**2*(k+1)*(k**2-1)/144)
    return L, float(1 - stats.norm.cdf(z))

rows = []
for o in (1, 2, 3, 4):
    dd = pd.read_csv(f"results_v3/tables/service_equalized_o{o}.csv.gz")
    dd = dd[(dd.model.isin([RF, "croston_sba"])) & (dd.cs == 5.0)]
    w = dd.pivot_table(index=["sku_id", "fill_target"], columns="model", values="cost_at_target", aggfunc="first").dropna()
    w = w[w.croston_sba != 0]
    rel = ((w[RF] - w.croston_sba) / w.croston_sba).unstack("fill_target").dropna().sort_index(axis=1)
    kol = list(rel.columns)
    delta = rel[kol[-1]].to_numpy(float) - rel[kol[0]].to_numpy(float)
    L, p_page = page_test(rel[kol].to_numpy(float))
    rows.append({"titik": o, "n": len(rel), "median_90": float(rel[kol[0]].median()),
                 "median_95": float(rel[kol[1]].median()), "median_98": float(rel[kol[2]].median()),
                 "median_delta": float(np.median(delta)),
                 "p_kontras_1sisi": float(stats.wilcoxon(delta, alternative="greater").pvalue),
                 "p_page_1sisi": p_page})
h2 = pd.DataFrame(rows)
h2["p_kontras_holm"] = holm(h2.p_kontras_1sisi.to_numpy())
h2["p_page_holm"] = holm(h2.p_page_1sisi.to_numpy())
print(h2.round(6).to_string(index=False))
print(f"  arah naik {int((h2.median_delta>0).sum())}/4 | Page mentah {int((h2.p_page_1sisi<0.05).sum())}/4 -> Holm {int((h2.p_page_holm<0.05).sum())}/4 | "
      f"kontras mentah {int((h2.p_kontras_1sisi<0.05).sum())}/4 -> Holm {int((h2.p_kontras_holm<0.05).sum())}/4")
h2.to_csv("results_v3/2metode/hasil_h2_relatif_holm_rev8.csv", index=False)
print("\ntersimpan: hasil_h1_relatif_rev8.csv, hasil_h2_relatif_holm_rev8.csv")
