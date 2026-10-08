"""stats_v3.py — Fase 4: uji hipotesis & estimasi yang benar untuk desain berpasangan + berkelompok.

Menjawab temuan audit pada statistik v2:
  * Holm dilakukan DI DALAM keluarga uji yang dinyatakan (tiga skenario biaya x pasangan model),
    bukan tiga p-value mentah yang lalu ditafsirkan satu per satu.
  * Estimator besaran efek dilaporkan sebagai HODGES-LEHMAN (median selisih berpasangan) + interval
    kepercayaan CLUSTER BOOTSTRAP (klaster = toko, departemen) karena SKU dari toko/toko-kategori yang
    sama tidak independen. Rata-rata selisih dilaporkan juga, tapi bukan sebagai klaim utama.
  * H2 diuji sebagai INTERAKSI (Friedman lintas skenario + Wilcoxon berpasangan antar skenario dengan
    Holm), bukan sebagai "ada skenario yang signifikan" seperti v2.
  * Metrik di luar MAE: MASE & RMSSE (skala naive H-jumlah pada data latih), ME, pinball loss pada
    kuantil kebijakan, PIS. Semua memakai besaran yang sama dengan yang dipakai kebijakan
    (jumlah demand periode proteksi), bukan metrik 1-langkah yang tidak relevan.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ------------------------------------------------------------------ uji & estimasi dasar
def wilcoxon_paired(a: np.ndarray, b: np.ndarray, alternative: str = "two-sided") -> dict:
    """Wilcoxon signed-rank berpasangan + rank-biserial r.

    Konvensi arah (dinyatakan eksplisit supaya tidak salah baca): d = a - b, dan
        r_rank_biserial = (W_neg - W_pos) / (W_pos + W_neg)
    sehingga r POSITIF berarti `a` LEBIH MURAH/LEBIH KECIL daripada `b` (yaitu RF lebih murah bila
    a = biaya RF). r dihitung dari peringkat langsung, tidak bergantung konvensi statistik scipy.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    d = a[ok] - b[ok]
    n = int(len(d))
    if n == 0 or np.allclose(d, 0):
        return {"n_pairs": n, "statistic": np.nan, "p_value": np.nan, "rank_biserial": np.nan,
                "median_diff": float(np.median(d)) if n else np.nan, "mean_diff": float(d.mean()) if n else np.nan}
    res = stats.wilcoxon(d, alternative=alternative, zero_method="wilcox")
    nz = d[d != 0]
    ranks = stats.rankdata(np.abs(nz))
    w_pos = float(ranks[nz > 0].sum())
    w_neg = float(ranks[nz < 0].sum())
    r_rb = (w_neg - w_pos) / (w_pos + w_neg) if (w_pos + w_neg) else np.nan
    return {"n_pairs": n, "statistic": float(res.statistic), "p_value": float(res.pvalue),
            "rank_biserial": float(r_rb), "w_pos": w_pos, "w_neg": w_neg,
            "median_diff": float(np.median(d)), "mean_diff": float(d.mean())}


def hodges_lehmann(a: np.ndarray, b: np.ndarray, max_n: int = 4000, seed: int = 42) -> float:
    """Estimator Hodges-Lehmann selisih berpasangan = median dari semua rata-rata pasangan (Walsh).

    Untuk n > max_n, Walsh averages dihitung pada subsampel acak deterministik (dicatat), karena
    jumlah rata-rata Walsh tumbuh n(n+1)/2.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    d = a[ok] - b[ok]
    if len(d) == 0:
        return np.nan
    if len(d) > max_n:
        rng = np.random.default_rng(seed)
        d = rng.choice(d, size=max_n, replace=False)
    i, j = np.triu_indices(len(d))
    walsh = (d[i] + d[j]) / 2
    return float(np.median(walsh))


def holm(pvals: list[float]) -> list[float]:
    """Koreksi Holm-Bonferroni; NaN diabaikan tapi tetap punya slot (dikembalikan NaN)."""
    p = np.asarray(pvals, float)
    ok = ~np.isnan(p)
    out = np.full(len(p), np.nan)
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return out.tolist()
    order = idx[np.argsort(p[idx])]
    m = len(order)
    running = 0.0
    for k, i in enumerate(order):
        adj = (m - k) * p[i]
        running = max(running, adj)
        out[i] = min(running, 1.0)
    return out.tolist()


def cluster_bootstrap_ci(
    values: np.ndarray,
    clusters: np.ndarray,
    stat=np.median,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """CI percentile untuk statistik dengan bootstrap pada level KLASTER (toko/departemen).

    Mengembalikan (estimasi, batas bawah, batas atas). Klaster di-resample dengan pengembalian;
    semua SKU dalam klaster yang terpilih ikut masuk — inilah yang membuat CI-nya jujur terhadap
    ketergantungan antar SKU (toko yang sama, kategori yang sama).
    """
    values = np.asarray(values, float)
    clusters = np.asarray(clusters)
    ok = ~np.isnan(values)
    values, clusters = values[ok], clusters[ok]
    uniq = np.unique(clusters)
    if len(uniq) < 2:
        return float(stat(values)), np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    by_cluster = {c: values[clusters == c] for c in uniq}
    for b in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        samp = np.concatenate([by_cluster[c] for c in pick])
        boots[b] = stat(samp)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(stat(values)), float(lo), float(hi)


# ------------------------------------------------------------------ metrik akurasi
def forecast_metrics(actual: pd.Series, pred: pd.Series, train_target: np.ndarray, tau: float) -> dict:
    """MASE/RMSSE (skala naive H-jumlah pada data latih), ME, pinball, dan jumlah absolut.

    train_target: deret target H-jumlah pada data LATIH (untuk skala naive: selisih jendela
    non-tumpang-tindih), sehingga skala metrik tidak memakai data uji.
    """
    a = np.asarray(actual, float)
    p = np.asarray(pred, float)
    ok = ~(np.isnan(a) | np.isnan(p))
    a, p = a[ok], p[ok]
    err = a - p
    n = len(err)
    if n == 0:
        return {"n": 0, "mae": np.nan, "mase": np.nan, "rmsse": np.nan, "me": np.nan, "pinball": np.nan}
    tt = np.asarray(train_target, float)
    tt = tt[~np.isnan(tt)]
    if len(tt) > 3:
        naive = np.abs(np.diff(tt))
        scale_mae = float(np.mean(naive)) if np.mean(naive) > 0 else np.nan
        scale_mse = float(np.mean(np.diff(tt) ** 2)) if np.mean(np.diff(tt) ** 2) > 0 else np.nan
    else:
        scale_mae = scale_mse = np.nan
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err**2))
    # pinball loss pada kuantil kebijakan tau (skor yang sejalan dengan keputusan kuantil)
    pinball = float(np.mean(np.where(err >= 0, tau * err, (tau - 1) * err)))
    return {
        "n": n,
        "mae": mae,
        "mase": mae / scale_mae if scale_mae and not np.isnan(scale_mae) else np.nan,
        "rmsse": float(np.sqrt(mse / scale_mse)) if scale_mse and not np.isnan(scale_mse) else np.nan,
        "me": float(np.mean(err)),
        "pinball": pinball,
    }


# ------------------------------------------------------------------ rangkuman berpasangan
def pairwise_service_equalized(
    eq: pd.DataFrame,
    model_a: str = "rf_global_direct",
    metric: str = "cost_at_target",
    cluster_cols: tuple[str, ...] = ("store_id",),
    n_boot: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """Bandingkan model_a dengan tiap model lain per (cs, fill_target), lalu koreksi Holm.

    eq: keluaran policy_v3.service_equalized yang sudah dilengkapi kolom store_id/dept_id/cat_id
        (SKU yang tidak tercakup pada fill target tertentu otomatis terbuang karena NaN).
    Uji utama satu sisi (a lebih murah dari b); dua sisi dilaporkan sebagai robustness.
    """
    rows = []
    for (cs, ft), g in eq.groupby(["cs", "fill_target"], sort=True):
        piv = g.pivot_table(index="sku_id", columns="model", values=metric, aggfunc="first")
        meta = g.drop_duplicates("sku_id").set_index("sku_id")
        if model_a not in piv.columns:
            continue
        for other in sorted(c for c in piv.columns if c != model_a):
            d = piv[[model_a, other]].dropna()
            if len(d) < 5:
                continue
            satu = wilcoxon_paired(d[model_a].to_numpy(), d[other].to_numpy(), alternative="less")
            dua = wilcoxon_paired(d[model_a].to_numpy(), d[other].to_numpy(), alternative="two-sided")
            selisih = (d[model_a] - d[other]).to_numpy()
            hl = hodges_lehmann(d[model_a].to_numpy(), d[other].to_numpy())
            blk = {}
            for cc in cluster_cols:
                if cc in meta.columns:
                    est, lo, hi = cluster_bootstrap_ci(selisih, meta.loc[d.index, cc].to_numpy(),
                                                       stat=np.median, n_boot=n_boot, seed=seed)
                    blk[f"ci_{cc}"] = (est, lo, hi)
            rows.append(
                {
                    "cs": float(cs),
                    "fill_target": float(ft),
                    "model_a": model_a,
                    "model_b": other,
                    "n_sku": int(len(d)),
                    "median_a": float(d[model_a].median()),
                    "median_b": float(d[other].median()),
                    "median_diff": float(np.median(selisih)),
                    "mean_diff": float(selisih.mean()),
                    "pct_diff_median": float(100 * np.median(selisih) / d[other].median()) if d[other].median() else np.nan,
                    "hodges_lehmann": hl,
                    "p_one_sided_a_lebih_murah": satu["p_value"],
                    "p_two_sided": dua["p_value"],
                    "r_rank_biserial": satu["rank_biserial"],
                    "menang_a": int((selisih < 0).sum()),
                    "menang_b": int((selisih > 0).sum()),
                    **{k: v[0] for k, v in blk.items()},
                    **{k + "_lo": v[1] for k, v in blk.items()},
                    **{k + "_hi": v[2] for k, v in blk.items()},
                }
            )
    out = pd.DataFrame(rows)
    if len(out):
        out["p_one_sided_holm"] = holm(out["p_one_sided_a_lebih_murah"].tolist())
        out["p_two_sided_holm"] = holm(out["p_two_sided"].tolist())
    return out


def friedman_interaksi(wide: pd.DataFrame) -> dict:
    """H2 sebagai interaksi: apakah biaya model berubah antar SKENARIO biaya? (uji Friedman per SKU).

    wide: baris = SKU, kolom = skenario biaya, isi = selisih biaya (RF - baseline) pada fill target
    yang sama. Friedman menguji apakah peringkat selisih berbeda antar skenario.
    """
    w = wide.dropna()
    if w.shape[0] < 5 or w.shape[1] < 3:
        return {"n_sku": int(w.shape[0]), "statistic": np.nan, "p_value": np.nan}
    stat, p = stats.friedmanchisquare(*[w[c].to_numpy() for c in w.columns])
    return {"n_sku": int(w.shape[0]), "n_scenario": int(w.shape[1]), "statistic": float(stat), "p_value": float(p)}


def heterogenitas(
    eq: pd.DataFrame,
    sku_meta: pd.DataFrame,
    model_a: str = "rf_global_direct",
    model_b: str = "croston_sba",
    cs: float = 5.0,
    fill_target: float = 0.95,
    by: tuple[str, ...] = ("cat_id", "adi_kuartil", "cv2_kuartil", "volume_kuartil"),
) -> pd.DataFrame:
    """Apakah keunggulan model bergantung pada karakter SKU? (median selisih + uji per subgrup)."""
    rows = []
    g = eq[(eq.cs == cs) & (eq.fill_target == fill_target)]
    piv = g.pivot_table(index="sku_id", columns="model", values="cost_at_target", aggfunc="first")
    if model_a not in piv.columns or model_b not in piv.columns:
        return pd.DataFrame()
    d = (piv[[model_a, model_b]].dropna())
    d["selisih_relatif"] = 100 * (d[model_a] - d[model_b]) / d[model_b]
    d = d.join(sku_meta.set_index("sku_id"), how="left")
    for col in by:
        if col not in d.columns:
            continue
        for val, sub in d.groupby(col, observed=True):
            r = wilcoxon_paired(sub[model_a].to_numpy(), sub[model_b].to_numpy(), alternative="less")
            rows.append({"dimensi": col, "nilai": str(val), "n_sku": len(sub),
                         "median_relatif_pct": float(sub["selisih_relatif"].median()),
                         "median_abs": float((sub[model_a] - sub[model_b]).median()),
                         "p_one_sided": r["p_value"], "r_rb": r["rank_biserial"]})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm_dalam_dimensi"] = out.groupby("dimensi")["p_one_sided"].transform(lambda s: holm(s.tolist()))
    return out
