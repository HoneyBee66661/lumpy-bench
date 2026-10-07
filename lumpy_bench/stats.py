"""Uji statistik: Wilcoxon berpasangan, koreksi Holm, uji Page, dan bootstrap klaster."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    m = p.size
    urut = np.argsort(p)
    adj = np.empty(m)
    jalan = 0.0
    for i, idx in enumerate(urut):
        jalan = max(jalan, (m - i) * p[idx])
        adj[idx] = min(jalan, 1.0)
    return adj


def page_test(tabel: np.ndarray) -> tuple[float, float]:
    """Uji Page untuk alternatif berurutan (naik). tabel: n baris x k kolom berurutan."""
    n, k = tabel.shape
    peringkat = np.apply_along_axis(stats.rankdata, 1, tabel)
    T = peringkat.sum(axis=0)
    L = float(np.sum(np.arange(1, k + 1) * T))
    rerata = n * k * (k + 1) ** 2 / 4
    var = n * k ** 2 * (k + 1) * (k ** 2 - 1) / 144
    z = (L - rerata) / np.sqrt(var) if var > 0 else 0.0
    return L, float(1 - stats.norm.cdf(z))


def wilcoxon_satu_sisi(selisih: np.ndarray, arah_lebih_kecil: bool = True) -> float:
    d = np.asarray(selisih, float)
    d = d[~np.isnan(d)]
    if d.size < 5:
        return float("nan")
    alt = "less" if arah_lebih_kecil else "greater"
    return float(stats.wilcoxon(d, alternative=alt).pvalue)


def bootstrap_median_klaster(selisih: pd.Series, klaster: pd.Series, n_boot: int = 2000,
                             seed: int = 20261007, alpha: float = 0.05) -> tuple[float, float]:
    """Selang kepercayaan median dengan pengelompokan klaster (mis. toko).

    Bersifat indikatif bila jumlah klaster sedikit.
    """
    df = pd.DataFrame({"d": selisih.to_numpy(float), "k": klaster.to_numpy()}).dropna()
    if df.empty:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    grup = [g["d"].to_numpy(float) for _, g in df.groupby("k")]
    med = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(grup), len(grup))
        gabung = np.concatenate([grup[i] for i in idx])
        med[b] = np.median(gabung)
    return float(np.quantile(med, alpha / 2)), float(np.quantile(med, 1 - alpha / 2))


def ringkas_uji(nama: str, selisih: np.ndarray, klaster: pd.Series | None = None,
                alpha: float = 0.05, seed: int = 20261007) -> dict:
    d = np.asarray(selisih, float)
    d = d[~np.isnan(d)]
    hasil = {
        "uji": nama, "n": int(d.size), "median_selisih": float(np.median(d)) if d.size else np.nan,
        "p_satu_sisi": wilcoxon_satu_sisi(d, arah_lebih_kecil=True),
        "pct_lebih_murah": float(100 * (d < 0).mean()) if d.size else np.nan,
    }
    if klaster is not None and d.size:
        lo, hi = bootstrap_median_klaster(pd.Series(d), klaster.reset_index(drop=True),
                                          seed=seed, alpha=alpha)
        hasil["ci95_bawah"], hasil["ci95_atas"] = lo, hi
    return hasil
