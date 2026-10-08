"""M3 (v2) — Random Forest GLOBAL: SATU model untuk seluruh SKU (spec v2 5.4-5.5).

Perbedaan dari versi v1: dulu 27 kombinasi x 40 SKU = 1.080 model per-SKU; sekarang setiap
kombinasi grid dilatih SEKALI di dataframe gabungan 40 SKU, jadi 27 fit + 1 fit model final.

Alur (semua berbasis urutan waktu, tidak ada k-fold acak):
  latih inti     hari 1..1581   -> tuning 27 kombinasi grid, MAE gabungan di jendela validasi
  validasi       hari 1582..1761 -> dipakai untuk MAE (gabungan + per SKU) dan residual per SKU
  latih ulang    hari 1..1761   -> model final (1 kombinasi terbaik), feature importance
  uji            hari 1762..1941 -> forecast walk-forward satu langkah, model tunggal untuk semua SKU

Keluaran (results/tables/):
  rf_tuning.csv                  MAE validasi GABUNGAN tiap kombinasi grid (checkpoint M3)
  mae_per_sku_validation.csv     MAE validasi per SKU untuk SETIAP kombinasi (deteksi model timpang)
  rf_best_params.csv             1 baris: kombinasi terbaik + ringkasan MAE/sigma
  rf_feature_importance.csv      feature_importances_ model final (bahan Pembahasan)
  rf_identity_mapping.csv        arti tiap integer code fitur identitas
  rf_val_residuals.csv           residual validasi per SKU per hari (sumber sigma simulasi 5.7)
  rf_test_forecast.csv           forecast periode uji per SKU per hari (dipakai M4/M5)
  rf_test_mae_per_sku.csv        MAE validasi/uji + sigma per SKU (dipakai M4 untuk mae_comparison)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import (  # noqa: E402
    GLOBAL_FEATURE_COLS,
    IDENTITY_COLS,
    IDENTITY_SOURCE,
    build_features,
    encode_identity,
)
from runlog import write as write_run_log  # noqa: E402

SERIES_COLS = ["item_id", "store_id", "day", "date", "demand", "snap", "event_name_1", "event_name_2"]
TUNING_COLS = [
    "n_estimators",
    "max_depth",
    "min_samples_leaf",
    "mae_val_pooled",
    "mae_val_median_per_sku",
    "n_train",
    "n_val",
    "fit_seconds",
    "is_best",
]
PER_SKU_COLS = [
    "n_estimators",
    "max_depth",
    "min_samples_leaf",
    "sku_id",
    "n_val",
    "mae_val",
    "is_best_combo",
]


def split_bounds(total_days: int, test_days: int, validation_days: int) -> tuple[int, int]:
    """(akhir validasi, total hari) -> latih inti 1..(akhir-valid), validasi, uji."""
    val_end = total_days - test_days
    return val_end - validation_days, val_end


def load_series(parquet: Path, sku_ids: list[str]) -> dict[str, pd.DataFrame]:
    """Ambil seluruh riwayat hari untuk SKU terpilih, satu scan dataset (bukan 40 scan)."""
    items = sorted({s.split("__")[0] for s in sku_ids})
    filt = ds.field("item_id").isin(items)
    tbl = ds.dataset(parquet, format="parquet").to_table(columns=SERIES_COLS, filter=filt)
    df = tbl.to_pandas()
    df["sku_id"] = df["item_id"].astype(str) + "__" + df["store_id"].astype(str)
    out = {sku: g.sort_values("day").reset_index(drop=True) for sku, g in df.groupby("sku_id")}
    missing = [s for s in sku_ids if s not in out]
    if missing:
        raise SystemExit(f"SKU berikut tidak ada di parquet: {missing}")
    return {s: out[s] for s in sku_ids}


def _make_model(n_estimators: int, max_depth, min_samples_leaf: int, seed: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=int(n_estimators),
        max_depth=(None if max_depth is None or (isinstance(max_depth, float) and np.isnan(max_depth)) else int(max_depth)),
        min_samples_leaf=int(min_samples_leaf),
        random_state=seed,
        n_jobs=-1,
    )


def pooled_frame(series: dict[str, pd.DataFrame], sku_ids: list[str]) -> tuple[pd.DataFrame, dict]:
    """Gabungkan fitur seluruh SKU jadi SATU dataframe + pemetaan fitur identitas."""
    frames = [build_features(series[sku], sku_id=sku) for sku in sku_ids]
    pooled = pd.concat(frames, ignore_index=True)
    pooled, mapping = encode_identity(pooled)
    return pooled, mapping


def identity_mapping_table(mapping: dict[str, dict[str, int]]) -> pd.DataFrame:
    rows = [
        {"kolom_fitur": code_col, "atribut_sumber": IDENTITY_SOURCE[code_col], "nilai": value, "code": code}
        for code_col, values in mapping.items()
        for value, code in values.items()
    ]
    return pd.DataFrame(rows).sort_values(["kolom_fitur", "code"]).reset_index(drop=True)


def best_of(tuning: pd.DataFrame) -> pd.Series:
    """Pilih kombinasi dengan MAE validasi gabungan terkecil; seri diputus deterministik."""
    t = tuning.copy()
    t["depth_key"] = t["max_depth"].fillna(0).astype(int)  # None (tanpa batas) = 0
    t = t.sort_values(["mae_val_pooled", "n_estimators", "depth_key", "min_samples_leaf"], kind="mergesort")
    return t.iloc[0]


def per_sku_mae(model, feat: pd.DataFrame, mask: pd.Series, sku_col: str = "sku_id") -> pd.DataFrame:
    """MAE per SKU pada baris terpilih (mask), memakai model yang sudah dilatih."""
    sub = feat.loc[mask, [sku_col, "target"]].copy()
    sub["pred"] = model.predict(feat.loc[mask, GLOBAL_FEATURE_COLS])
    sub["abs_err"] = (sub["target"] - sub["pred"]).abs()
    return (
        sub.groupby(sku_col)
        .agg(n_val=("abs_err", "size"), mae_val=("abs_err", "mean"))
        .reset_index()
    )


def run(config_path: Path, parquet: Path, sample_csv: Path) -> pd.DataFrame:
    cfg = yaml.safe_load(config_path.read_text())
    seed = int(cfg["seed"])
    test_days = int(cfg["split"]["test_days"])
    val_days = int(cfg["split"]["validation_days"])
    grids = cfg["rf"]
    if str(grids.get("model_type", "global")).lower() != "global":
        raise SystemExit("config rf.model_type harus 'global' untuk spec v2")
    results = Path(cfg["paths"]["results"])
    (results / "tables").mkdir(parents=True, exist_ok=True)

    sample = pd.read_csv(sample_csv)
    sku_ids = sample["sku_id"].tolist()
    print(f"SKU contoh: {len(sku_ids)} (dari {sample_csv}) — MODEL GLOBAL, satu model untuk semua SKU")

    series = load_series(parquet, sku_ids)
    total_days = int(max(s["day"].max() for s in series.values()))
    train_end, val_end = split_bounds(total_days, test_days, val_days)
    print(f"total hari {total_days}; latih 1..{train_end}, validasi {train_end + 1}..{val_end}, uji {val_end + 1}..{total_days}")

    combos = [
        (n, d, m)
        for n in grids["n_estimators_grid"]
        for d in grids["max_depth_grid"]
        for m in grids["min_samples_leaf_grid"]
    ]
    print(
        f"grid: {len(grids['n_estimators_grid'])} x {len(grids['max_depth_grid'])} x "
        f"{len(grids['min_samples_leaf_grid'])} = {len(combos)} kombinasi × 1 model global "
        f"(v1: {len(combos)} × {len(sku_ids)} = {len(combos) * len(sku_ids)} fit)"
    )

    t_start = time.time()
    print("\nmenyusun dataframe gabungan (fitur per SKU + fitur identitas)...")
    pooled, mapping = pooled_frame(series, sku_ids)
    mapping_df = identity_mapping_table(mapping)
    mapping_df.to_csv(results / "tables" / "rf_identity_mapping.csv", index=False)
    complete = pooled[GLOBAL_FEATURE_COLS].notna().all(axis=1)
    n_warmup = int((~complete).sum())
    print(
        f"  baris gabungan: {len(pooled):,} (dibuang warm-up {n_warmup:,} baris) | "
        f"fitur: {len(GLOBAL_FEATURE_COLS)} ({len(IDENTITY_COLS)} identitas)"
    )

    tr_mask = complete & (pooled["day"] <= train_end)
    va_mask = complete & (pooled["day"] > train_end) & (pooled["day"] <= val_end)
    tv_mask = complete & (pooled["day"] <= val_end)
    te_mask = complete & (pooled["day"] > val_end)
    print(
        f"  latih inti {int(tr_mask.sum()):,} baris | validasi {int(va_mask.sum()):,} | "
        f"latih+validasi {int(tv_mask.sum()):,} | uji {int(te_mask.sum()):,}"
    )

    X_tr, y_tr = pooled.loc[tr_mask, GLOBAL_FEATURE_COLS], pooled.loc[tr_mask, "target"]
    X_va, y_va = pooled.loc[va_mask, GLOBAL_FEATURE_COLS], pooled.loc[va_mask, "target"]

    tuning_rows: list[dict] = []
    per_sku_rows: list[dict] = []
    fitted: dict[tuple, RandomForestRegressor] = {}
    for i, (n, d, m) in enumerate(combos, start=1):
        t0 = time.time()
        model = _make_model(n, d, m, seed).fit(X_tr, y_tr)
        pred = model.predict(X_va)
        mae_pooled = float(mean_absolute_error(y_va, pred))
        sku_mae = per_sku_mae(model, pooled, va_mask)
        secs = round(time.time() - t0, 3)
        fitted[(n, d, m)] = model
        tuning_rows.append(
            {
                "n_estimators": int(n),
                "max_depth": (None if d is None else int(d)),
                "min_samples_leaf": int(m),
                "mae_val_pooled": mae_pooled,
                "mae_val_median_per_sku": float(sku_mae["mae_val"].median()),
                "n_train": int(tr_mask.sum()),
                "n_val": int(va_mask.sum()),
                "fit_seconds": secs,
            }
        )
        per_sku_rows += [
            {
                "n_estimators": int(n),
                "max_depth": (None if d is None else int(d)),
                "min_samples_leaf": int(m),
                "sku_id": r.sku_id,
                "n_val": int(r.n_val),
                "mae_val": float(r.mae_val),
            }
            for r in sku_mae.itertuples()
        ]
        print(
            f"  [{i:2d}/{len(combos)}] n={int(n):3d} depth={str(d):>4s} leaf={int(m):2d} | "
            f"MAE val gabungan {mae_pooled:.4f} | median per SKU {sku_mae['mae_val'].median():.4f} | {secs:.0f}s"
        )

    tuning = pd.DataFrame(tuning_rows)[[c for c in TUNING_COLS if c != "is_best"]]
    chosen = best_of(tuning)
    best_cfg = (
        int(chosen["n_estimators"]),
        None if pd.isna(chosen["max_depth"]) else int(chosen["max_depth"]),
        int(chosen["min_samples_leaf"]),
    )
    tuning["is_best"] = (
        (tuning["n_estimators"] == best_cfg[0])
        & (tuning["max_depth"].fillna(-1).astype(int) == (-1 if best_cfg[1] is None else best_cfg[1]))
        & (tuning["min_samples_leaf"] == best_cfg[2])
    )
    tuning = tuning[TUNING_COLS]
    tuning.to_csv(results / "tables" / "rf_tuning.csv", index=False)

    per_sku = pd.DataFrame(per_sku_rows)
    per_sku["is_best_combo"] = (
        (per_sku["n_estimators"] == best_cfg[0])
        & (per_sku["max_depth"].fillna(-1).astype(int) == (-1 if best_cfg[1] is None else best_cfg[1]))
        & (per_sku["min_samples_leaf"] == best_cfg[2])
    )
    per_sku = per_sku[PER_SKU_COLS].sort_values(["is_best_combo", "mae_val"], ascending=[False, True])
    per_sku.to_csv(results / "tables" / "mae_per_sku_validation.csv", index=False)
    print(
        f"\nkombinasi terbaik: n_estimators={best_cfg[0]}, max_depth={best_cfg[1]}, "
        f"min_samples_leaf={best_cfg[2]} -> MAE validasi gabungan {chosen['mae_val_pooled']:.4f}"
    )

    # residual validasi (model terbaik dilatih di latih inti saja) -> sigma per SKU untuk simulasi
    model_val = fitted[best_cfg]
    va = pooled.loc[va_mask, ["sku_id", "day", "date", "target"]].copy()
    va["forecast"] = model_val.predict(X_va)
    va["residual"] = va["target"] - va["forecast"]
    resid = va[["sku_id", "day", "date", "target", "forecast", "residual"]].rename(columns={"target": "demand"})
    resid.to_csv(results / "tables" / "rf_val_residuals.csv", index=False)

    # model final: latih ulang di latih+validasi
    t0 = time.time()
    X_tv, y_tv = pooled.loc[tv_mask, GLOBAL_FEATURE_COLS], pooled.loc[tv_mask, "target"]
    model_final = _make_model(*best_cfg, seed).fit(X_tv, y_tv)
    fit_seconds = round(time.time() - t0, 3)

    imp = (
        pd.DataFrame({"feature": GLOBAL_FEATURE_COLS, "importance": model_final.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    imp.insert(0, "rank", imp.index + 1)
    imp["is_identity_feature"] = imp["feature"].isin(IDENTITY_COLS)
    imp.to_csv(results / "tables" / "rf_feature_importance.csv", index=False)
    print("\n5 fitur paling berpengaruh:")
    for r in imp.head(5).itertuples():
        print(f"  {r.rank}. {r.feature:18s} {r.importance:.4f}")

    # walk-forward periode uji: fitur hari t dari actual <= t-1, model tunggal untuk semua SKU
    te = pooled.loc[te_mask, ["sku_id", "day", "date", "target"]].copy()
    te["forecast"] = model_final.predict(pooled.loc[te_mask, GLOBAL_FEATURE_COLS])
    test_fc = te[["sku_id", "day", "date", "target", "forecast"]].rename(columns={"target": "demand"})
    test_fc.to_csv(results / "tables" / "rf_test_forecast.csv", index=False)

    test_sku = (
        te.assign(abs_err=lambda d: (d["target"] - d["forecast"]).abs())
        .groupby("sku_id")
        .agg(n_test=("abs_err", "size"), mae_test=("abs_err", "mean"))
        .reset_index()
    )
    val_sku = (
        va.assign(abs_err=lambda d: (d["residual"]).abs())
        .groupby("sku_id")
        .agg(n_val=("abs_err", "size"), mae_val=("abs_err", "mean"), sigma_resid_val=("residual", "std"))
        .reset_index()
    )
    per_sku_test = test_sku.merge(val_sku, on="sku_id", how="left")[["sku_id", "mae_val", "mae_test", "sigma_resid_val", "n_val", "n_test"]]
    per_sku_test.to_csv(results / "tables" / "rf_test_mae_per_sku.csv", index=False)

    mae_test_pooled = float(mean_absolute_error(te["target"], te["forecast"]))
    best = pd.DataFrame(
        [
            {
                "model_type": "global",
                "n_estimators": best_cfg[0],
                "max_depth": best_cfg[1],
                "min_samples_leaf": best_cfg[2],
                "n_sku": len(sku_ids),
                "n_features": len(GLOBAL_FEATURE_COLS),
                "mae_val_pooled": float(chosen["mae_val_pooled"]),
                "mae_val_median_per_sku": float(per_sku_test["mae_val"].median()),
                "mae_test_pooled": mae_test_pooled,
                "mae_test_median_per_sku": float(per_sku_test["mae_test"].median()),
                "mae_test_min_per_sku": float(per_sku_test["mae_test"].min()),
                "mae_test_max_per_sku": float(per_sku_test["mae_test"].max()),
                "sigma_resid_val_median_per_sku": float(per_sku_test["sigma_resid_val"].median()),
                "sigma_resid_val_min_per_sku": float(per_sku_test["sigma_resid_val"].min()),
                "sigma_resid_val_max_per_sku": float(per_sku_test["sigma_resid_val"].max()),
                "n_train_core": int(tr_mask.sum()),
                "n_train_final": int(tv_mask.sum()),
                "n_val": int(va_mask.sum()),
                "n_test": int(te_mask.sum()),
                "fit_seconds_final": fit_seconds,
            }
        ]
    )
    best.to_csv(results / "tables" / "rf_best_params.csv", index=False)

    dur = time.time() - t_start
    print(f"\nM3 (global) selesai dalam {dur:.1f} s")
    print(f"MAE validasi gabungan {best_cfg and float(chosen['mae_val_pooled']):.4f} | MAE uji gabungan {mae_test_pooled:.4f}")
    print(
        f"MAE uji per SKU: min {per_sku_test['mae_test'].min():.4f} | median {per_sku_test['mae_test'].median():.4f} | "
        f"maks {per_sku_test['mae_test'].max():.4f}"
    )

    extra = "\n".join(
        [
            f"model_type    : GLOBAL (satu model untuk {len(sku_ids)} SKU, spec v2 5.5)",
            f"skema split   : latih 1..{train_end}, validasi {train_end + 1}..{val_end}, uji {val_end + 1}..{total_days}",
            f"kombinasi     : {len(combos)} kombinasi grid × 1 model global = {len(combos)} fit (v1: {len(combos) * len(sku_ids)} fit)",
            f"fitur         : {len(GLOBAL_FEATURE_COLS)} = 12 deret waktu/kalender + {len(IDENTITY_COLS)} identitas {IDENTITY_COLS}",
            f"baris latih   : inti {int(tr_mask.sum()):,} | final (latih+validasi) {int(tv_mask.sum()):,} | uji {int(te_mask.sum()):,}",
            f"best params   : n_estimators={best_cfg[0]}, max_depth={best_cfg[1]}, min_samples_leaf={best_cfg[2]}",
            f"sklearn       : {__import__('sklearn').__version__}",
            f"mae_val_pooled : {float(chosen['mae_val_pooled']):.6f}",
            f"mae_test_pooled: {mae_test_pooled:.6f}",
            f"mae_test_median_per_sku: {float(per_sku_test['mae_test'].median()):.6f}",
            f"fitur teratas : " + ", ".join(f"{r.feature} ({r.importance:.4f})" for r in imp.head(5).itertuples()),
            f"durasi        : {dur:.1f} s",
        ]
    )
    write_run_log(results / "run_log.txt", "M3 model_rf GLOBAL (spec v2 5.4-5.5)", seed, t_start, extra=extra)
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description="RF GLOBAL: tuning waktu + walk-forward (M3 v2).")
    ap.add_argument("--config", type=Path, default=Path("config.yaml"))
    ap.add_argument("--parquet", type=Path, default=None)
    ap.add_argument("--sample", type=Path, default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text())
    parquet = a.parquet or (Path(cfg["paths"]["data_interim"]) / "sales_long.parquet")
    sample = a.sample or (Path(cfg["paths"]["results"]) / "tables" / "sku_sample.csv")
    run(a.config, parquet, sample)


if __name__ == "__main__":
    main()
