"""forecast_v3.py — Fase 2: forecast JUMLAH demand periode proteksi (H = R+L) tanpa look-ahead.

Dua keluarga model, sengaja dipisah karena sifat informasinya berbeda:

  A. Model RASIO (keluarga Croston/ES/MA/naive — per SKU, murni univariat)
     rate_t diperbarui walk-forward (hanya memakai demand <= t), lalu
         pred_sum_t = rate_t * H
     Model: naive, ma7, ma28, ses (alpha dituning di validasi), croston_classic, croston_sba,
     croston_sba_optimized, tsb, adida, imapa  (semua dari statsforecast kecuali na/ma/ses/optimized).

  B. Model ML GLOBAL (pooled lintas SKU, direct multi-horizon)
     target = jumlah demand t+1..t+H (lihat features_v3), fitur <= t. Dilatih SEKALI per origin pada
     data latih+validasi, lalu memprediksi tiap hari periode uji. Tidak ada penjumlahan forecast
     1-langkah seperti v2 (penyebab look-ahead yang diaudit).
     Model: rf_global_direct (gelombang 1), lightgbm_tweedie & rf_recursive (gelombang 2).

Kesalahan forecast per SKU (untuk safety stock kuantil empiris, Fase 2):
    err_t = actual_sum(t) - pred_sum(t), dihitung pada jendela VALIDASI (bukan uji).
    SS_sku(tau) = kuantil_tau(err_sku); bila observasi SKU < min_obs -> pooled per kategori/global.

Semua fungsi di sini deterministik pada seed dan tidak membaca data uji untuk membentuk prediksi.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src_v3"))
from features_v3 import GLOBAL_FEATURE_COLS, TARGET_COL, build_features_v3, encode_pooled, features_and_target  # noqa: E402

RATE_MODELS = [
    "naive",
    "ma7",
    "ma28",
    "ses",
    "croston_classic",
    "croston_sba",
    "croston_sba_optimized",
    "tsb",
    "adida",
    "imapa",
]
ML_MODELS = ["rf_global_direct"]
ML_MODELS_GELOMBANG_2 = ["lightgbm_tweedie", "rf_recursive"]


# --------------------------------------------------------------------------- keluarga A: model rasio
def _ses_forecast(y: np.ndarray, alpha: float) -> float:
    fitted = float(y[0])
    for v in y[1:]:
        fitted = alpha * float(v) + (1 - alpha) * fitted
    return alpha * float(y[-1]) + (1 - alpha) * fitted


def rate_forecast(model: str, hist: np.ndarray, alpha_ses: float = 0.1, alpha_tsb: float = 0.2, beta_tsb: float = 0.2) -> float:
    """Rate satu hari berikutnya dari riwayat demand <= t (hist[-1] = demand hari t)."""
    y = np.asarray(hist, dtype=float)
    if len(y) == 0:
        return 0.0
    if model == "naive":
        return float(y[-1])
    if model == "ma7":
        return float(y[-7:].mean())
    if model == "ma28":
        return float(y[-28:].mean())
    if model == "ses":
        return float(_ses_forecast(y, alpha_ses))
    if model in {"croston_classic", "croston_sba", "croston_sba_optimized"}:
        from statsforecast.models import CrostonClassic, CrostonOptimized, CrostonSBA

        if model == "croston_classic":
            m = CrostonClassic()
        elif model == "croston_sba":
            m = CrostonSBA()
        else:
            m = CrostonOptimized()
        return float(m.forecast(y=y, h=1)["mean"][0])
    if model == "tsb":
        from statsforecast.models import TSB

        m = TSB(alpha_d=alpha_tsb, alpha_p=beta_tsb)
        return float(m.forecast(y=y, h=1)["mean"][0])
    if model == "adida":
        from statsforecast.models import ADIDA

        return float(ADIDA().forecast(y=y, h=1)["mean"][0])
    if model == "imapa":
        from statsforecast.models import IMAPA

        return float(IMAPA().forecast(y=y, h=1)["mean"][0])
    raise ValueError(f"model rasio tidak dikenal: {model}")


def rate_forecasts_series(model: str, demand: pd.Series, days: np.ndarray, alpha_ses: float = 0.1) -> np.ndarray:
    """Walk-forward: untuk tiap hari di `days`, hitung rate memakai demand hari < hari itu."""
    y = demand.to_numpy(float)
    idx = demand.index.to_numpy()
    out = np.empty(len(days), dtype=float)
    for i, t in enumerate(days):
        pos = int(np.searchsorted(idx, t))          # jumlah observasi dengan day < t
        out[i] = rate_forecast(model, y[:pos], alpha_ses=alpha_ses) if pos > 0 else 0.0
    return out


def tune_alpha_ses(demand: pd.Series, val_days: np.ndarray, grid: tuple = (0.05, 0.1, 0.2, 0.3, 0.5)) -> float:
    """Pilih alpha SES dengan MSE terkecil pada jendela validasi (bukan data uji)."""
    y = demand.to_numpy(float)
    idx = demand.index.to_numpy()
    best, best_err = 0.1, np.inf
    for alpha in grid:
        err = 0.0
        n = 0
        for t in val_days:
            pos = int(np.searchsorted(idx, t))
            if pos == 0:
                continue
            pred = _ses_forecast(y[:pos], alpha)
            err += (y[pos - 1] - pred) ** 2 if pos - 1 < len(y) else 0.0
            n += 1
        err = err / max(n, 1)
        if err < best_err:
            best, best_err = alpha, err
    return float(best)


# ------------------------------------------------------------- keluarga B: model ML global (direct)
def train_pooled_direct(frames: list[pd.DataFrame], hari_latih_max: int, horizon: int, params: dict, seed: int):
    """Latih SATU model global pada target direct, memakai baris dengan hari <= hari_latih_max."""
    pooled, mapping = encode_pooled(frames)
    mask = (pooled["day"] <= hari_latih_max) & pooled[GLOBAL_FEATURE_COLS].notna().all(axis=1) & pooled[TARGET_COL].notna()
    train = pooled.loc[mask]
    model = RandomForestRegressor(
        n_estimators=int(params["n_estimators"]),
        max_depth=(None if params.get("max_depth") is None else int(params["max_depth"])),
        min_samples_leaf=int(params["min_samples_leaf"]),
        random_state=seed,
        n_jobs=-1,
    ).fit(train[GLOBAL_FEATURE_COLS], train[TARGET_COL])
    return model, mapping, pooled, int(len(train))


def predict_pooled_direct(model, pooled: pd.DataFrame, days: np.ndarray) -> pd.DataFrame:
    """Prediksi target proteksi untuk hari-hari tertentu dari fitur <= t (tanpa look-ahead)."""
    sub = pooled.loc[pooled["day"].isin(days) & pooled[GLOBAL_FEATURE_COLS].notna().all(axis=1)]
    pred = model.predict(sub[GLOBAL_FEATURE_COLS])
    return pd.DataFrame({"sku_id": sub["sku_id"].to_numpy(), "day": sub["day"].to_numpy(), "pred_sum": pred})


def train_pooled_1step(frames: list[pd.DataFrame], hari_latih_max: int, params: dict, seed: int):
    """RF pada target 1 LANGKAH (demand[t+1]) — dipakai oleh baseline `rf_recursive`.

    Target dibuat dari demand yang digeser; fitur tetap hanya <= t. Model inilah yang dipakai v2,
    di sini dibawa sebagai pembanding "pendekatan lama" dengan fitur yang sudah bebas look-ahead.
    """
    pooled, mapping = encode_pooled(frames)
    # target 1 langkah: demand hari t+1 (bergeser per SKU)
    pooled = pooled.sort_values(["sku_id", "day"]).reset_index(drop=True)
    pooled["target_1"] = pooled.groupby("sku_id", sort=False)["demand"].shift(-1)
    mask = ((pooled["day"] <= hari_latih_max) & pooled[GLOBAL_FEATURE_COLS].notna().all(axis=1)
            & pooled["target_1"].notna())
    train = pooled.loc[mask]
    model = RandomForestRegressor(
        n_estimators=int(params["n_estimators"]),
        max_depth=(None if params.get("max_depth") is None else int(params["max_depth"])),
        min_samples_leaf=int(params["min_samples_leaf"]),
        random_state=seed,
        n_jobs=-1,
    ).fit(train[GLOBAL_FEATURE_COLS], train["target_1"])
    return model, mapping, pooled, int(len(train))


def rf_recursive_predict(
    model,
    series: dict[str, pd.DataFrame],
    days: np.ndarray,
    horizon: int,
    ident_by_sku: dict[str, dict],
) -> pd.DataFrame:
    """Jumlah demand t+1..t+H secara REKURSIF, dibatch lintas SKU (cepat, satu panggilan predict/step).

    Untuk tiap hari t: prediksi langkah 1..H. Pada tiap langkah, fitur SEMUA SKU dibangun sekaligus
    dari riwayat (aktual sampai t, lalu prediksi sebelumnya pada rekursi hari itu) sehingga model
    dipanggil H kali per hari, bukan H x n_SKU kali. Fitur <= t (ditambah kalender esok hari yang
    memang diketahui di muka); tidak ada informasi demand masa depan yang masuk.
    """
    skus = list(series.keys())
    d = {s: series[s].sort_values("day").reset_index(drop=True) for s in skus}
    n = len(skus)
    dem = {s: d[s]["demand"].astype(float).to_numpy() for s in skus}
    pos_of = {s: {int(v): i for i, v in enumerate(d[s]["day"].to_numpy())} for s in skus}
    dow = {s: pd.to_datetime(d[s]["date"]).dt.dayofweek.to_numpy() for s in skus}
    bulan = {s: pd.to_datetime(d[s]["date"]).dt.month.to_numpy() for s in skus}
    snap = {s: d[s]["snap"].to_numpy() for s in skus}
    is_ev = {s: (d[s]["event_name_1"].notna() | d[s]["event_name_2"].notna()).to_numpy() for s in skus}

    # buffer numpy pra-alokasi per SKU (panjang riwayat + H) -> tidak ada konversi list->array tiap langkah
    buf = {}
    for s in skus:
        arr = np.zeros(len(dem[s]) + horizon, dtype=float)
        arr[: len(dem[s])] = dem[s]
        buf[s] = arr

    # inferensi rekursif memakai batch kecil: matikan paralelisme thread supaya overhead joblib tidak
    # mendominasi (terukur: predict 8-300 baris jauh lebih cepat dengan n_jobs=1)
    n_jobs_semula = getattr(model, "n_jobs", None)
    if n_jobs_semula is not None:
        model.n_jobs = 1

    # indeks kolom fitur: mengisi matriks fitur langsung jauh lebih cepat daripada membangun dict
    # per SKU per langkah (terukur pada 8 SKU: 295 s -> 138 s saat dict dihapus, optimasi ini lanjutan)
    COL = {c: i for i, c in enumerate(GLOBAL_FEATURE_COLS)}
    (C_L1, C_L7, C_L14, C_L28, C_RM7, C_RM28, C_RS28, C_DOW, C_MONTH, C_WKND, C_EVT,
     C_SNAP, C_EVTH, C_SNAPH, C_WKNDH) = (
        COL["lag_1"], COL["lag_7"], COL["lag_14"], COL["lag_28"], COL["rolling_mean_7"],
        COL["rolling_mean_28"], COL["rolling_std_28"], COL["dow"], COL["month"], COL["is_weekend"],
        COL["is_event_hari_ini"], COL["snap_hari_ini"], COL["event_h"], COL["snap_h"], COL["weekend_h"],
    )

    out_rows = []
    for t in days:
        t = int(t)
        pos = {s: pos_of[s].get(t) for s in skus}
        if any(p is None for p in pos.values()):
            continue
        ptr = {s: pos[s] + 1 for s in skus}       # posisi tulis prediksi berikutnya
        total = np.zeros(n)
        gagal = False
        for k in range(1, horizon + 1):
            hari = t + k
            feat = np.full((n, len(GLOBAL_FEATURE_COLS)), np.nan, dtype=float)
            for r, s in enumerate(skus):
                i = pos_of[s].get(hari)
                if i is None:
                    gagal = True
                    break
                p = ptr[s]
                b = buf[s][:p]
                panjang = len(b)
                if panjang >= 28:
                    w28 = b[-28:]
                    feat[r, C_RM28] = w28.mean()
                    feat[r, C_RS28] = w28.std(ddof=1)
                    feat[r, C_L28] = b[-28]
                if panjang >= 7:
                    feat[r, C_RM7] = b[-7:].mean()
                    feat[r, C_L7] = b[-7]
                if panjang >= 14:
                    feat[r, C_L14] = b[-14]
                feat[r, C_L1] = b[-1]
                feat[r, C_DOW] = dow[s][i]
                feat[r, C_MONTH] = bulan[s][i]
                feat[r, C_WKND] = int(dow[s][i] >= 5)
                feat[r, C_SNAP] = snap[s][i]
                feat[r, C_EVT] = int(is_ev[s][i])
                feat[r, C_EVTH] = 0.0
                feat[r, C_SNAPH] = 0.0
                feat[r, C_WKNDH] = 0.0
                for c, v in ident_by_sku[s].items():
                    feat[r, COL[c]] = v
            if gagal or np.isnan(feat).any():
                gagal = True
                break
            pred = np.clip(model.predict(feat), 0.0, None)
            total += pred
            for r, s in enumerate(skus):
                buf[s][ptr[s]] = pred[r]
                ptr[s] += 1
        for r, s in enumerate(skus):
            out_rows.append({"sku_id": s, "day": t, "pred_sum": np.nan if gagal else float(total[r])})

    if n_jobs_semula is not None:
        model.n_jobs = n_jobs_semula
    return pd.DataFrame(out_rows)


def protection_actual(daily: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Jumlah demand aktual t+1..t+H per hari (dipakai untuk error validasi & evaluasi)."""
    d = daily.sort_values("day").reset_index(drop=True)
    s = d["demand"].astype(float)
    fwd = s.rolling(horizon, min_periods=horizon).sum().shift(-horizon)
    return pd.DataFrame({"day": d["day"], "actual_sum": fwd})
