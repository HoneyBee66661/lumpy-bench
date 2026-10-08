
"""ADI/CV2 pada irisan periode aktif x periode pra-uji (hari <= 1221) — versi streaming (RAM kecil)."""
import numpy as np, pandas as pd, pyarrow.dataset as ds, time

BATAS = 1221
parquet = "data/interim/sales_long.parquet"
COLS = ["item_id", "store_id", "day", "demand", "sell_price"]

dataset = ds.dataset(parquet, format="parquet")
akum = {}
t0 = time.time()
n = 0
for batch in dataset.to_batches(columns=COLS, batch_size=1_000_000):
    d = batch.to_pandas()
    d = d[(d.day <= BATAS) & d.sell_price.notna()]
    if not len(d):
        continue
    d["pos"] = d.demand > 0
    d["pos_sum"] = np.where(d.pos, d.demand, 0.0)
    d["pos_sq"] = np.where(d.pos, d.demand ** 2, 0.0)
    g = d.groupby(["item_id", "store_id"], observed=True).agg(
        n_aktif=("demand", "size"), n_nonzero=("pos", "sum"),
        sum_pos=("pos_sum", "sum"), sumsq_pos=("pos_sq", "sum"))
    for k, row in g.iterrows():
        a = akum.get(k)
        if a is None:
            akum[k] = [int(row.n_aktif), int(row.n_nonzero), float(row.sum_pos), float(row.sumsq_pos)]
        else:
            a[0] += int(row.n_aktif); a[1] += int(row.n_nonzero)
            a[2] += float(row.sum_pos); a[3] += float(row.sumsq_pos)
    n += 1
    print(f"  batch {n:3d} | {time.time()-t0:5.0f}s | SKU terkumpul {len(akum):,}", flush=True)

idx = pd.MultiIndex.from_tuples(list(akum.keys()), names=["item_id", "store_id"])
v = np.array(list(akum.values()), dtype=float)
out = pd.DataFrame(v, index=idx, columns=["n_aktif", "n_nonzero", "sum_pos", "sumsq_pos"])
out = out[out.n_nonzero >= 2].copy()
out["mean_pos"] = out.sum_pos / out.n_nonzero
var = (out.sumsq_pos - out.n_nonzero * out.mean_pos**2) / (out.n_nonzero - 1)
out["cv2"] = (np.sqrt(var.clip(lower=0)) / out.mean_pos) ** 2
out["adi"] = out.n_aktif / out.n_nonzero
out["lumpy"] = (out.adi > 1.32) & (out.cv2 > 0.49)
out["sku_id"] = [f"{i}__{s}" for i, s in out.index]
out = out.reset_index(drop=True)
out.to_csv("results_v3/2metode/adi_cv2_irisan_prauji.csv", index=False)
print(f"\nTOTAL SKU dihitung: {len(out):,} | lumpy (irisan hari<= {BATAS}): {int(out.lumpy.sum()):,} "
      f"({100*out.lumpy.mean():.1f}%)")

lama = pd.read_csv("results_v3/tables/lumpy_v3.csv")[["sku_id"]].assign(lumpy_lama=True)
sampel = pd.read_csv("results_v3/tables/sku_sample_v3.csv")
gab = out.merge(lama, on="sku_id", how="left")
gab["lumpy_lama"] = gab.lumpy_lama.fillna(False)
print("populasi lama (periode aktif penuh):", len(lama))
print("  tetap lumpy di irisan          :", int((gab.lumpy & gab.lumpy_lama).sum()))
print("  gugur setelah dipotong         :", int((~gab.lumpy & gab.lumpy_lama).sum()))
print("  baru lumpy di irisan           :", int((gab.lumpy & ~gab.lumpy_lama).sum()))
s = gab[gab.sku_id.isin(set(sampel.sku_id))]
print(f"\ndari 300 SKU sampel: tetap lumpy {int(s.lumpy.sum())} | gugur {int((~s.lumpy).sum())}")
gug = s[~s.lumpy]
if len(gug):
    print("  median ADI gugur:", round(float(gug.adi.median()), 3), "| median CV2 gugur:", round(float(gug.cv2.median()), 3))
    print("  contoh:", list(gug.sku_id.head(5)))
print("\nselesai", f"{time.time()-t0:.0f}s")
