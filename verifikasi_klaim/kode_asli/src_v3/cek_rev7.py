"""cek_rev7.py — menurunkan SEMUA angka yang dibutuhkan revisi ketujuh (20 catatan pembimbing).

Tidak menghitung ulang apa pun dari nol: seluruh angka dibaca dari artefak v3 yang sudah
diverifikasi (results_v3/tables/*) atau dihitung dari prediksi yang tersimpan
(results_v3/interim/pred_*.csv.gz). Keluaran disimpan ke
results_v3/2metode/CATATAN_REV7.txt supaya bisa dikutip langsung oleh draf artikel.

Jalankan:  .venv/bin/python src_v3/cek_rev7.py
"""
from __future__ import annotations

import inspect
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RF = "rf_global_direct#s42"
CR = "croston_sba"
CS_UTAMA = 5.0
TARGETS = (0.90, 0.95, 0.98)
ORIGINS = (1, 2, 3, 4)

buf = io.StringIO()


def P(*a, **kw):
    print(*a, **kw)
    print(*a, **kw, file=buf)


def seksi(t):
    P("\n" + "=" * 78)
    P(t)
    P("=" * 78)


# ---------------------------------------------------------------- 1. CAKUPAN
seksi("1. CAKUPAN SKU & N EFEKTIF (per titik x target x model)")
cov_rows = []
for o in ORIGINS:
    d = pd.read_csv(ROOT / f"results_v3/tables/service_equalized_o{o}.csv.gz")
    d = d[d.model.isin([RF, CR]) & (d.cs == CS_UTAMA)]
    for (m, t), g in d.groupby(["model", "fill_target"], observed=True):
        cov_rows.append({"origin": o, "model": m, "target": float(t),
                         "n_tercakup": int(g.cost_at_target.notna().sum()),
                         "n_fillmax_ge_975": int((g.fill_max >= 0.975).sum())})
cov = pd.DataFrame(cov_rows)
P("\ntabel cakupan (biaya terinterpolasi tersedia), dari 300 SKU per titik:")
P(cov.pivot_table(index=["origin", "target"], columns="model", values="n_tercakup")
    .astype(int).to_string())
P("\nrentang cakupan per target (min-maks lintas titik x model):")
for t in TARGETS:
    s = cov[cov.target == t].n_tercakup
    P(f"  target {t:.2f}: {s.min()} s/d {s.max()} SKU")

# cakupan menurut penanda 'tercakup' (angka yang dulu dikutip draf) vs biaya benar-benar terisi
tc_rows = []
for o in ORIGINS:
    d = pd.read_csv(ROOT / f"results_v3/tables/service_equalized_o{o}.csv.gz")
    d = d[d.model.isin([RF, CR]) & (d.cs == CS_UTAMA)]
    for (m, t), g in d.groupby(["model", "fill_target"], observed=True):
        tc_rows.append({"origin": o, "model": m, "target": float(t),
                        "n_tercakup": int(g.tercakup.sum()),
                        "n_biaya_terisi": int(g.cost_at_target.notna().sum()),
                        "tercakup_tapi_biaya_kosong": int((g.tercakup & g.cost_at_target.isna()).sum())})
tc = pd.DataFrame(tc_rows)
P("\npenanda tercakup vs nilai biaya terisi (penjelas selisih 189 -> 171):")
for t in TARGETS:
    s = tc[tc.target == t]
    P(f"  target {t:.2f}: penanda tercakup {s.n_tercakup.min()} s/d {s.n_tercakup.max()} SKU | "
      f"biaya terisi {s.n_biaya_terisi.min()} s/d {s.n_biaya_terisi.max()} SKU | "
      f"baris tercakup tanpa biaya {s.tercakup_tapi_biaya_kosong.sum()}")


# irisan dua model per target, lalu irisan tiga target x dua model
P("\nirisan cakupan (dua model) per titik:")
ir_rows = []
for o in ORIGINS:
    per_target = {}
    for t in TARGETS:
        sub = cov[(cov.origin == o) & (cov.target == t)]
        per_target[t] = set()
        for m in (RF, CR):
            d = pd.read_csv(ROOT / f"results_v3/tables/service_equalized_o{o}.csv.gz")
            d = d[(d.model == m) & (d.cs == CS_UTAMA) & (d.fill_target == t)]
            per_target[t] |= set(d.loc[d.cost_at_target.notna(), "sku_id"])
        # catatan: di atas = himpunan SKU yang punya biaya pada model itu, belum gabungan
    # himpunan yang biayanya tersedia pada KEDUA model untuk target ini
    per_target = {}
    for t in TARGETS:
        d = pd.read_csv(ROOT / f"results_v3/tables/service_equalized_o{o}.csv.gz")
        d = d[d.model.isin([RF, CR]) & (d.cs == CS_UTAMA) & (d.fill_target == t)]
        w = d.pivot_table(index="sku_id", columns="model", values="cost_at_target",
                          aggfunc="first")
        per_target[t] = set(w.dropna().index)
    iris3 = per_target[0.90] & per_target[0.95] & per_target[0.98]
    # 'biaya lengkap': iris tiga target pada kedua model, hanya SKU dengan biaya terisi
    d = pd.read_csv(ROOT / f"results_v3/tables/service_equalized_o{o}.csv.gz")
    d = d[d.model.isin([RF, CR]) & (d.cs == CS_UTAMA)]
    w = d.pivot_table(index="sku_id", columns=["model", "fill_target"],
                      values="cost_at_target", aggfunc="first")
    lengkap = set(w.dropna().index)
    ir_rows.append({"origin": o, "irisan_90": len(per_target[0.90]),
                    "irisan_95": len(per_target[0.95]), "irisan_98": len(per_target[0.98]),
                    "irisan_3target": len(iris3), "biaya_lengkap_2model_3target": len(lengkap),
                    "iris3_minus_lengkap": len(iris3) - len(lengkap)})
ir = pd.DataFrame(ir_rows)
P(ir.to_string(index=False))

# diagnostik pengguguran pada target 98%
P("\ndiagnostik pengguguran pada target 0,98 (dua model):")
for o in ORIGINS:
    d = pd.read_csv(ROOT / f"results_v3/tables/service_equalized_o{o}.csv.gz")
    d = d[d.model.isin([RF, CR]) & (d.cs == CS_UTAMA) & (d.fill_target == 0.98)]
    gugur = d[d.cost_at_target.isna()]
    uniq = gugur.drop_duplicates("sku_id")
    P(f"  titik {o}: SKU gugur {uniq.sku_id.nunique():3d} | median fill maksimum "
      f"{uniq.fill_max.median():.3f} | di dalam selisih tipis dari target "
      f"(fill_max >= 0,975): {100*float((uniq.fill_max >= 0.975).mean()):.1f}% "
      f"| median ADI {uniq.adi.median():.2f} vs tercakup "
      f"{d[d.cost_at_target.notna()].drop_duplicates('sku_id').adi.median():.2f} "
      f"| median CV2 {uniq.cv2.median():.2f} vs "
      f"{d[d.cost_at_target.notna()].drop_duplicates('sku_id').cv2.median():.2f}")

# uji kesulitan SKU tersisih vs tercakup (target 98%): ADI dan CV2
P("\nuji kesulitan SKU tersisih (target 0,98) — ADI dan CV2, uji satu sisi (tersisih > tercakup):")
for o in ORIGINS:
    d = pd.read_csv(ROOT / f"results_v3/tables/service_equalized_o{o}.csv.gz")
    d = d[d.model.isin([RF, CR]) & (d.cs == CS_UTAMA) & (d.fill_target == 0.98)]
    gug = d[d.cost_at_target.isna()].drop_duplicates("sku_id")
    ada = d[d.cost_at_target.notna()].drop_duplicates("sku_id")
    for kol, label in (("adi", "ADI"), ("cv2", "CV2")):
        u = stats.mannwhitneyu(gug[kol].to_numpy(float), ada[kol].to_numpy(float),
                               alternative="greater")
        P(f"  titik {o} {label}: n tersisih {len(gug)} vs tercakup {len(ada)} | "
          f"median {gug[kol].median():.2f} vs {ada[kol].median():.2f} | "
          f"p satu sisi {u.pvalue:.6f}")


# ------------------------------------------------------------- 2. KUANTIL
seksi("2. KUANTIL YANG MENGHASILKAN TIAP TARGET FILL RATE (titik operasi)")
baris, nonmono, tot_kurva = [], 0, 0
for o in ORIGINS:
    sw = pd.read_csv(ROOT / f"results_v3/tables/policy_sweep_o{o}.csv.gz")
    sw = sw[sw.model.isin([RF, CR])]
    for (sku, m), g in sw.groupby(["sku_id", "model"], observed=True):
        g = g.sort_values("tau")
        tot_kurva += 1
        fr = g.fill_rate.to_numpy(float)
        if np.any(np.diff(fr) < -1e-12):
            nonmono += 1
        for t in TARGETS:
            i = int(np.argmin(np.abs(fr - t)))
            atas = np.where(fr >= t)[0]
            bawah = np.where(fr <= t)[0]
            lebar = float(fr[atas[0]] - fr[bawah[-1]]) if len(atas) and len(bawah) else np.nan
            baris.append({"origin": o, "model": m, "target": t,
                          "tau_terdekat": float(g.tau.to_numpy()[i]),
                          "selisih": float(abs(fr[i] - t)), "lebar": lebar})
k = pd.DataFrame(baris)
P(f"\nkurva SKU x model diperiksa: {tot_kurva:,} | non-monoton: {nonmono:,} "
  f"({100*nonmono/tot_kurva:.1f}%)")
P("\nmedian tau (kuantil kuantil galat validasi) yang menghasilkan tiap target:")
P(k.groupby(["model", "target"]).tau_terdekat.median().unstack().round(3).to_string())
P("\nmedian |fill rate - target| pada titik grid terdekat:")
P(k.groupby(["model", "target"]).selisih.median().round(4).unstack().to_string())
P("\nmedian lebar selang grid yang mengurung target:")
P(k.groupby(["model", "target"]).lebar.median().round(4).unstack().to_string())
P("\npangsa kasus tanpa titik grid di atas target (kurva berhenti di bawah target), %:")
P((100 * k.assign(tanpa_atas=k.lebar.isna()).groupby(["model", "target"])
   .tanpa_atas.mean().unstack()).round(1).to_string())

# ------------------------------------------- 3. H2: absolut vs relatif
seksi("3. H2 — SELISIH ABSOLUT vs RELATIF (% biaya Croston-SBA)")


def page_test(tabel: np.ndarray):
    n, kk = tabel.shape
    per = np.apply_along_axis(lambda r: stats.rankdata(r), 1, tabel)
    T = per.sum(axis=0)
    L = float(np.sum(np.arange(1, kk + 1) * T))
    z = (L - n * kk * (kk + 1) ** 2 / 4) / np.sqrt(n * kk ** 2 * (kk + 1) * (kk ** 2 - 1) / 144)
    return L, float(1 - stats.norm.cdf(z))


def boot_ci(x, n_boot=5000, seed=42):
    rng = np.random.default_rng(seed)
    med = np.median(rng.choice(x, size=(n_boot, len(x)), replace=True), axis=1)
    return float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))


h2 = []
for o in ORIGINS:
    d = pd.read_csv(ROOT / f"results_v3/tables/service_equalized_o{o}.csv.gz")
    d = d[d.model.isin([RF, CR]) & (d.cs == CS_UTAMA)]
    w = d.pivot_table(index=["sku_id", "fill_target"], columns="model",
                      values="cost_at_target", aggfunc="first").dropna()
    nol = w[CR] <= 0
    sel = w[RF] - w[CR]
    rel = sel.div(w[CR].where(~nol))
    P(f"  titik {o}: himpunan seragam {w.index.get_level_values('sku_id').nunique()} SKU "
      f"({len(w)} baris SKU x target) | baris dengan biaya Croston-SBA nol (dikeluarkan dari "
      f"ukuran relatif) {int(nol.sum())}")
    for nama, ser in (("absolut (satuan unit)", sel), ("relatif (% biaya Croston)", rel)):
        t = ser.unstack("fill_target").dropna().sort_index(axis=1)
        kol = list(t.columns)
        arr = t[kol].to_numpy(float)
        d90, d95, d98 = arr[:, 0], arr[:, 1], arr[:, 2]
        delta = d98 - d90
        L, p_page = page_test(arr)
        lo, hi = boot_ci(delta)
        h2.append({"titik": o, "ukuran": nama, "n": len(delta),
                   "median_d90": float(np.median(d90)), "median_d98": float(np.median(d98)),
                   "median_delta": float(np.median(delta)), "CI95": f"[{lo:+.5f}, {hi:+.5f}]",
                   "p_kontras": float(stats.wilcoxon(delta, alternative="greater").pvalue),
                   "p_page": p_page,
                   "arah_naik": bool(np.median(delta) > 0),
                   "%SKU_delta_pos": round(100 * float((delta > 0).mean()), 1),
                   "CI_memuat_0": bool(lo <= 0 <= hi)})
hh = pd.DataFrame(h2)
P(hh.to_string(index=False))
for nama in hh.ukuran.unique():
    s = hh[hh.ukuran == nama]
    P(f"\n  {nama}: delta naik {int(s.arah_naik.sum())}/4 | kontras 1 sisi <5% "
      f"{int((s.p_kontras < 0.05).sum())}/4 | Page 1 sisi <5% {int((s.p_page < 0.05).sum())}/4 | "
      f"CI95 memuat 0: {int(s.CI_memuat_0.sum())}/4")
hh.to_csv(ROOT / "results_v3/2metode/hasil_h2_absolut_vs_relatif.csv", index=False)

# ------------------------------------- 4. INVARIANSI RASIO BIAYA + H1
seksi("4. RASIO BIAYA 5:1 / 10:1 / 20:1 — hanya skala, bukan rezim berbeda")
ub = pd.read_csv(ROOT / "results_v3/2metode/tables/uji_berpasangan_2metode.csv")
P("\nuji berpasangan (Wilcoxon satu sisi, RF lebih murah): 36 baris = 4 titik x 3 rasio x 3 target")
inv = []
for (o, t), g in ub.groupby(["origin", "fill_target"]):
    md = g.median_diff.to_numpy(float)
    pp = g.p_one_sided_a_lebih_murah.to_numpy(float)
    inv.append({"titik": o, "target": t,
                "median_diff_cs5": float(md[0]), "median_diff_cs20": float(md[-1]),
                "selisih_maks_antar_rasio": float(np.abs(md - md[0]).max()),
                "selisih_relatif_maks": float(np.abs(md - md[0]).max() / max(abs(md[0]), 1e-12)),
                "p_identik": len(set(np.round(pp, 12))) == 1,
                "n_sku_identik": len(set(g.n_sku)) == 1})
iv = pd.DataFrame(inv)
P(iv.round(8).to_string(index=False))
P(f"\n  p-value identik lintas tiga rasio biaya pada {int(iv.p_identik.sum())}/"
  f"{len(iv)} sel (titik x target); himpunan SKU identik pada {int(iv.n_sku_identik.sum())}/"
  f"{len(iv)} sel")
P(f"  selisih mutlak median_diff antar rasio: maks {iv.selisih_maks_antar_rasio.max():.3e} "
  f"({100*iv.selisih_relatif_maks.max():.4f}% dari nilai pada 5:1)")
iv.to_csv(ROOT / "results_v3/2metode/invariansi_rasio_biaya_rev7.csv", index=False)

P("\nH1 (keluarga uji artikel): skenario dasar 5:1, target 95%")
h1 = ub[(ub.cs == CS_UTAMA) & (ub.fill_target == 0.95)][
    ["origin", "n_sku", "median_a", "median_b", "median_diff", "hodges_lehmann",
     "p_one_sided_a_lebih_murah", "p_one_sided_holm", "ci_store_id_lo", "ci_store_id_hi"]]
P(h1.round(4).to_string(index=False))
p_raw = h1.p_one_sided_a_lebih_murah.to_numpy()
order = np.argsort(p_raw)
holm = np.maximum.accumulate((len(p_raw) - np.arange(len(p_raw))) * p_raw[order])
P(f"\n  Holm lintas 4 titik: p mentah terkecil {p_raw.min():.4f} -> Holm "
  f"{float(holm.min()):.4f} | lolos <5%: {int((holm < 0.05).sum())}/4")
P("\nsekunder 10:1 dan 20:1 (target 95%), Holm di dalam keluarganya (8 uji):")
sek = ub[(ub.cs != CS_UTAMA) & (ub.fill_target == 0.95)][
    ["cs", "origin", "median_diff", "p_one_sided_a_lebih_murah"]]
P(sek.round(4).to_string(index=False))
ps = sek.p_one_sided_a_lebih_murah.to_numpy()
o2 = np.argsort(ps)
holm2 = np.maximum.accumulate((len(ps) - np.arange(len(ps))) * ps[o2])
P(f"  p mentah terkecil {ps.min():.4f} -> Holm {float(holm2.min()):.4f} | lolos <5%: "
  f"{int((holm2 < 0.05).sum())}/{len(ps)}")
P("\nseluruh 36 uji (H1 semua target x rasio x titik) tanpa dipisah keluarga:")
P(f"  p mentah terkecil {ub.p_one_sided_a_lebih_murah.min():.4f}; "
  f"p Holm terkecil di kolom file {ub.p_one_sided_holm.min():.4f}")

# ------------------------------------------------- 5. EKOR: MAE vs PINBALL
seksi("5. PEMERIKSAAN EKOR — MAE vs pinball pada kuantil yang benar-benar dipakai")


def pinball(err: np.ndarray, tau: float) -> float:
    return float(np.mean(np.where(err >= 0, tau * err, (tau - 1) * err)))


TAB = {}
ok = True
try:
    for o in ORIGINS:
        pd.read_csv(ROOT / f"results_v3/interim/pred_{CR}_o{o}_s0.csv.gz", nrows=5)
        pd.read_csv(ROOT / f"results_v3/interim/pred_rf_global_direct_o{o}_s42.csv.gz", nrows=5)
except Exception as exc:  # pragma: no cover
    ok = False
    P(f"  pred tidak terbaca: {exc}")

if ok:
    import pyarrow.dataset as pds

    sampel = pd.read_csv(ROOT / "results_v3/tables/sku_sample_v3.csv")
    P(f"\nmemuat aktual dari sales_long.parquet untuk {len(sampel):,} SKU sampel ...")
    dsp = pds.dataset(ROOT / "data/interim/sales_long.parquet", format="parquet")
    filt = (pds.field("day") >= 1222) & (pds.field("item_id").isin(list(sampel.item_id.astype(str))))
    aktual = dsp.to_table(columns=["item_id", "store_id", "day", "demand"], filter=filt).to_pandas()
    aktual["sku_id"] = aktual.item_id.astype(str) + "__" + aktual.store_id.astype(str)
    aktual = aktual.sort_values(["sku_id", "day"])
    H_PROTEKSI = 8
    aktual["y"] = (aktual.groupby("sku_id").demand
                   .transform(lambda s: s.rolling(H_PROTEKSI).sum().shift(-H_PROTEKSI)))
    aktual = aktual[["sku_id", "day", "demand", "y"]]
    P(f"  baris aktual: {len(aktual):,} | SKU: {aktual.sku_id.nunique():,} | "
      f"target = jumlah demand t+1..t+{H_PROTEKSI}")
    # kontrol: MAE per SKU yang dihitung di sini harus identik dengan tabel terverifikasi
    ref = pd.read_csv(ROOT / "results_v3/2metode/tables/akurasi_per_sku_2metode.csv.gz")
    ref = ref[(ref.split == "test") & (ref.model.isin(["croston_sba", "rf_global_direct"]))
              & ((ref.model != "rf_global_direct") | (ref.seed == 42))]
    cek_rows = []
    for o in ORIGINS:
        pr = pd.read_csv(ROOT / f"results_v3/interim/pred_{CR}_o{o}_s0.csv.gz")
        pf = pd.read_csv(ROOT / f"results_v3/interim/pred_rf_global_direct_o{o}_s42.csv.gz")
        for model, p in (("croston_sba", pr), ("rf_global_direct", pf)):
            p = p[p.split == "test"][["sku_id", "day", "pred_sum"]]
            mm = p.merge(aktual, on=["sku_id", "day"], how="inner").dropna(subset=["y"])
            mine = mm.groupby("sku_id").apply(
                lambda x: float(np.mean(np.abs(x.y - x.pred_sum))), include_groups=False)
            r = ref[(ref.origin == o) & (ref.model == model)].set_index("sku_id").mae
            j = pd.concat([mine.rename("mine"), r.rename("ref")], axis=1).dropna()
            cek_rows.append({"titik": o, "model": model, "n": len(j),
                             "selisih_maks": float(np.abs(j.mine - j.ref).max())})
    cek = pd.DataFrame(cek_rows)
    P("\nkontrol MAE per SKU (hitung di sini vs tabel terverifikasi akurasi_per_sku_2metode.csv.gz):")
    P(cek.to_string(index=False))
    P(f"  selisih maksimum keseluruhan: {cek.selisih_maks.max():.3e} "
      f"-> {'COCOK' if cek.selisih_maks.max() < 1e-9 else 'TIDAK COCOK'}")
    rows = []
    for o in ORIGINS:
        pr = pd.read_csv(ROOT / f"results_v3/interim/pred_{CR}_o{o}_s0.csv.gz")
        pf = pd.read_csv(ROOT / f"results_v3/interim/pred_rf_global_direct_o{o}_s42.csv.gz")
        pr, pf = pr[pr.split == "test"], pf[pf.split == "test"]
        m = pr.merge(pf, on=["sku_id", "day"], suffixes=("_cr", "_rf"))
        m = m.merge(aktual, on=["sku_id", "day"], how="inner").dropna(subset=["y"])
        m["err_cr"] = m.y - m.pred_sum_cr
        m["err_rf"] = m.y - m.pred_sum_rf
        g = m.groupby("sku_id")
        mae_cr = g.err_cr.apply(lambda x: float(np.mean(np.abs(x))))
        mae_rf = g.err_rf.apply(lambda x: float(np.mean(np.abs(x))))
        dif = mae_rf - mae_cr
        row = {"origin": o, "n": len(dif), "mae_rf": float(mae_rf.median()),
               "mae_cr": float(mae_cr.median()), "selisih_median_mae": float(dif.median()),
               "selisih_pct": 100 * float(dif.median() / mae_cr.median()),
               "p_MAE": float(stats.wilcoxon(dif, alternative="less").pvalue)}
        for tau in (0.85, 0.90, 0.95):
            pb_cr = g.err_cr.apply(lambda x: pinball(x.to_numpy(float), tau))
            pb_rf = g.err_rf.apply(lambda x: pinball(x.to_numpy(float), tau))
            dd = pb_rf - pb_cr
            row[f"pb_rf_{tau}"] = float(pb_rf.median())
            row[f"pb_cr_{tau}"] = float(pb_cr.median())
            row[f"selisih_pb_{tau}"] = float(dd.median())
            row[f"p_pb_{tau}"] = float(stats.wilcoxon(dd, alternative="less").pvalue)
        rows.append(row)
    if rows:
        ek = pd.DataFrame(rows)
        P("\nselisih berpasangan per SKU (RF - Croston-SBA; negatif = RF lebih baik):")
        P(ek.round(5).to_string(index=False))
        ek.to_csv(ROOT / "results_v3/2metode/ekor_mae_pinball_tau_rev7.csv", index=False)

# ---------------------------------------------------- 6. ADI/CV2 IRISAN
seksi("6. KLASIFIKASI ADI/CV2 PADA IRISAN PERIODE LATIH (hari <= 1221)")
f_ir = ROOT / "results_v3/2metode/adi_cv2_irisan_prauji.csv"
if f_ir.exists():
    ir2 = pd.read_csv(f_ir)
    P(f"  SKU dihitung pada irisan: {len(ir2):,} | lumpy: {int(ir2.lumpy.sum()):,} "
      f"({100*ir2.lumpy.mean():.1f}%)")
    lama = pd.read_csv(ROOT / "results_v3/tables/lumpy_v3.csv")[["sku_id"]]
    sampel = pd.read_csv(ROOT / "results_v3/tables/sku_sample_v3.csv")
    gab = (ir2.merge(lama.assign(lumpy_lama=True), on="sku_id", how="left")
           .assign(lumpy_lama=lambda x: x.lumpy_lama.fillna(False)))
    s = gab[gab.sku_id.isin(set(sampel.sku_id))]
    P(f"  dari 300 SKU sampel: tetap lumpy {int(s.lumpy.sum())} | tidak lagi lumpy "
      f"{int((~s.lumpy).sum())} | tidak terhitung {300 - len(s)}")
    gu = s[~s.lumpy]
    P(f"  SKU yang tidak lagi lumpy: median ADI {gu.adi.median():.2f}, "
      f"median CV2 {gu.cv2.median():.2f}")
else:
    P("  (berkas belum ada — jalankan src_v3/cek_irisan_prauji.py)")

# ------------------------------------------------------ 7. ALPHA CROSTON
seksi("7. SUMBER PARAMETER CROSTON-SBA (pustaka statsforecast)")
try:
    import statsforecast
    from statsforecast import models as M

    src = inspect.getsource(getattr(M, "_croston_classic", M.CrostonClassic))
    P(f"  versi statsforecast: {statsforecast.__version__}")
    for ln in src.splitlines():
        if "alpha" in ln or "0.95" in ln or "0.1" in ln:
            P("   ", ln.strip())
    src_sba = inspect.getsource(getattr(M, "_croston_sba", M.CrostonSBA))
    for ln in src_sba.splitlines():
        if "0.95" in ln:
            P("   ", ln.strip())
except Exception as exc:  # pragma: no cover
    P(f"  gagal membaca sumber: {exc}")

# ------------------------------------------------- 8. POPULASI & SAMPLING
seksi("8. POPULASI LUMPY, HARI BERPERMINTAAN, DEFINISI LAYANAN")
lumpy = pd.read_csv(ROOT / "results_v3/tables/lumpy_v3.csv")
P(f"  populasi lumpy (periode aktif): {len(lumpy):,} SKU dari 30.490 deret "
  f"({100*len(lumpy)/30490:.1f}%)")
P(f"  hari berpermintaan (n_nonzero): min {int(lumpy.n_nonzero.min())}, "
  f"median {int(lumpy.n_nonzero.median())}, maks {int(lumpy.n_nonzero.max())}")
sim = (ROOT / "src_v3/simulator_v3.py").read_text()
for i, ln in enumerate(sim.splitlines(), 1):
    if "fill_rate" in ln or "backlog" in ln and "settle" in ln:
        P(f"  simulator_v3.py:{i}: {ln.strip()}")

out = ROOT / "results_v3/2metode/CATATAN_REV7.txt"
out.write_text(buf.getvalue())
print(f"\ntersimpan: {out}")
