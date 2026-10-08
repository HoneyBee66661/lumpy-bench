
import numpy as np, pandas as pd
from scipy import stats
RF = "rf_global_direct#s42"
pd.set_option("display.width", 240)

def holm(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p); adj = np.empty(m); run = 0.0
    for i, idx in enumerate(order):
        run = max(run, (m - i) * p[idx]); adj[idx] = min(run, 1.0)
    return adj

def load(o, cs=5.0):
    d = pd.read_csv(f"results_v3/tables/service_equalized_o{o}.csv.gz")
    return d[(d.model.isin([RF, "croston_sba"])) & (d.cs == cs)]

print("=" * 90)
print("(a) CAKUPAN: penanda tercakup vs biaya terisi, per titik dan target")
baris = []
for o in (1, 2, 3, 4):
    d = load(o)
    for t in (0.90, 0.95, 0.98):
        g = d[d.fill_target == t]
        for m in (RF, "croston_sba"):
            gm = g[g.model == m]
            baris.append({"titik": o, "target": t, "model": "RF" if m == RF else "Croston",
                          "tercakup": int(gm.tercakup.sum()), "ada_biaya": int(gm.cost_at_target.notna().sum()),
                          "tercakup_tanpa_biaya": int((gm.tercakup & gm.cost_at_target.isna()).sum())})
a = pd.DataFrame(baris)
print(a.pivot_table(index=["titik", "target"], columns="model", values=["tercakup", "ada_biaya", "tercakup_tanpa_biaya"]).to_string())

print("\n(b) irisan himpunan")
for o in (1, 2, 3, 4):
    d = load(o)
    piv_ada = d.pivot_table(index="sku_id", columns=["fill_target", "model"], values="cost_at_target", aggfunc="first")
    piv_cak = d.pivot_table(index="sku_id", columns=["fill_target", "model"], values="tercakup", aggfunc="first")
    irisan_ada3 = int(piv_ada.notna().all(axis=1).sum())
    irisan_cak3 = int(piv_cak.all(axis=1).sum())
    print(f"  titik {o}: irisan 3 target x 2 model -> tercakup {irisan_cak3} | ada biaya {irisan_ada3} | "
          f"selisih {irisan_cak3 - irisan_ada3}")

print("\n(c) SEBAB biaya kosong pada baris tercakup: periksa kurva kebijakan SKU tersebut")
o, t = 1, 0.95
d = load(o)
sasaran = d[(d.fill_target == t) & (d.model == RF) & d.tercakup & d.cost_at_target.isna()].sku_id.unique()
print(f"  titik 1 target 95% RF: {len(sasaran)} SKU tercakup tanpa biaya; contoh: {list(sasaran[:3])}")
sw = pd.read_csv(f"results_v3/tables/policy_sweep_o{o}.csv.gz")
contoh = sw[(sw.sku_id == sasaran[0]) & (sw.model == RF)]
print("  kurva kebijakan SKU contoh (tau, fill_rate, cost_total, komponen):")
print(contoh[["tau", "fill_rate", "cost_total", "cost_stockout", "cost_holding"]].head(10).round(3).to_string(index=False))
print("  jumlah NaN cost_total pada SKU itu:", int(contoh.cost_total.isna().sum()), "dari", len(contoh))
# cek apakah masalahnya kolom biaya kosong di seluruh sweep untuk SKU itu
print("  kolom di sweep:", [c for c in sw.columns if "cost" in c or "backlog" in c or "on_hand" in c][:8])
