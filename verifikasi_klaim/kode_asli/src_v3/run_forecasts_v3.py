"""run_forecasts_v3.py — Fase 2/5: hitung forecast periode proteksi (val + uji) untuk semua model.

Dipanggil lokal (subset kecil, untuk uji) MAUPUN di Kaggle (300 SKU, 4 origin, 3 seed). Keluaran satu
berkas per (model, origin, seed):

    <out>/pred_<model>_o<origin>_s<seed>.csv.gz     kolom: sku_id, day, split (val|test), pred_sum

Val dan test dihitung dari satu model per origin (per seed): model dilatih pada data <= train_end
untuk memprediksi jendela validasi (dipakai membentuk safety stock kuantil empiris), dan dilatih ulang
pada data <= val_end untuk memprediksi jendela uji. Tidak ada informasi uji yang masuk ke model mana pun.

Rate model (naive/MA/SES/Croston/TSB/ADIDA/IMAPA) deterministik -> seed 0.
Model ML: rf_global_direct, rf_recursive (gelombang 1-2), lightgbm_tweedie (gelombang 2).
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src_v3"))
from features_v3 import GLOBAL_FEATURE_COLS, TARGET_COL, build_features_v3, encode_pooled  # noqa: E402
from forecast_v3 import (  # noqa: E402
    predict_pooled_direct,
    rate_forecasts_series,
    rf_recursive_predict,
    train_pooled_1step,
    train_pooled_direct,
    tune_alpha_ses,
)

COLS = ["item_id", "store_id", "day", "date", "demand", "snap", "event_name_1", "event_name_2", "sell_price"]
RATE_MODELS = ["naive", "ma7", "ma28", "ses", "croston_classic", "croston_sba", "croston_sba_optimized", "tsb", "adida", "imapa"]
ML_MODELS = ["rf_global_direct"]
ML_MODELS_GELOMBANG_2 = ["lightgbm_tweedie", "rf_recursive"]


def load_series(parquet: Path, sku_ids: list[str]) -> dict[str, pd.DataFrame]:
    items = sorted({s.split("__")[0] for s in sku_ids})
    stores = sorted({s.split("__")[1] for s in sku_ids})
    tbl = ds.dataset(parquet, format="parquet").to_table(
        columns=COLS, filter=(ds.field("item_id").isin(items) & ds.field("store_id").isin(stores))
    )
    df = tbl.to_pandas()
    df["sku_id"] = df["item_id"].astype(str) + "__" + df["store_id"].astype(str)
    df = df[df["sku_id"].isin(set(sku_ids))]
    out = {s: g.sort_values("day").reset_index(drop=True) for s, g in df.groupby("sku_id", sort=False)}
    missing = [s for s in sku_ids if s not in out]
    if missing:
        raise SystemExit(f"{len(missing)} SKU tidak ditemukan di parquet, mis. {missing[:3]}")
    return {s: out[s] for s in sku_ids}


def origin_windows(cfg: dict) -> list[dict]:
    total = int(cfg.get("total_days", 1941))
    test_days = int(cfg["origin"]["test_days"])
    val_days = int(cfg["split"]["validation_days"])
    n_origin = int(cfg["origin"]["n_origin"])
    out = []
    for i in range(1, n_origin + 1):
        val_end = total - test_days * (n_origin - i + 1)
        train_end = val_end - val_days
        out.append(
            {
                "origin": i,
                "train_end": train_end,
                "val_end": val_end,
                "days_val": np.arange(train_end + 1, val_end + 1),
                "days_test": np.arange(val_end + 1, min(val_end + 1 + test_days, total + 1)),
            }
        )
    return out


def _rate_model_preds(model: str, series: dict[str, pd.DataFrame], days_val, days_test, H: int, cfg: dict) -> pd.DataFrame:
    from joblib import Parallel, delayed

    def satu(sku: str) -> pd.DataFrame:
        d = series[sku].set_index("day")["demand"]
        alpha = tune_alpha_ses(d, days_val) if model == "ses" else 0.1
        rv = rate_forecasts_series(model, d, days_val, alpha_ses=alpha)
        rt = rate_forecasts_series(model, d, days_test, alpha_ses=alpha)
        return pd.concat(
            [
                pd.DataFrame({"sku_id": sku, "day": days_val, "split": "val", "pred_sum": rv * H}),
                pd.DataFrame({"sku_id": sku, "day": days_test, "split": "test", "pred_sum": rt * H}),
            ],
            ignore_index=True,
        )

    n_jobs = int(cfg.get("n_jobs", -1))
    frames = Parallel(n_jobs=n_jobs, prefer="processes")(delayed(satu)(s) for s in series)
    return pd.concat(frames, ignore_index=True)


def _ml_preds(model: str, series, frames, pooled, origin, cfg, seed, H, mapping, origin_max: int) -> pd.DataFrame:
    params = cfg["rf_v3"]
    days_val, days_test = origin["days_val"], origin["days_test"]
    parts = []
    if model == "rf_global_direct":
        for tag, hari, dd in (("val", origin["train_end"], days_val), ("test", origin["val_end"], days_test)):
            m, _, _, n = train_pooled_direct(frames, hari, H, params, seed)
            p = predict_pooled_direct(m, pooled, dd)
            p["split"] = tag
            parts.append(p)
            print(f"      rf_global_direct {tag}: latih {n:,} baris (<= {hari}) -> {len(p):,} prediksi")
            if tag == "test":
                # Fase 6: artefak model + feature importance ikut disimpan (gap yang ditemukan sendiri
                # di v2: tidak ada artefak model tersimpan). Model lengkap hanya untuk satu kombinasi
                # representatif (origin terakhir, seed pertama) supaya paket tidak membengkak.
                art = Path(cfg["paths"]["results"]) / "artifacts"
                art.mkdir(parents=True, exist_ok=True)
                imp = (pd.DataFrame({"fitur": GLOBAL_FEATURE_COLS, "importance": m.feature_importances_})
                       .sort_values("importance", ascending=False))
                imp.to_csv(art / f"importance_{model}_o{origin['origin']}_s{seed}.csv", index=False)
                print(f"      importance tersimpan: importance_{model}_o{origin['origin']}_s{seed}.csv")
                if seed == 42 and origin["origin"] == origin_max:
                    try:
                        import joblib

                        pth = art / f"model_{model}_o{origin['origin']}_s{seed}.joblib"
                        joblib.dump(m, pth, compress=3)
                        print(f"      artefak model: {pth} ({pth.stat().st_size / 1e6:.1f} MB)")
                    except Exception as exc:  # noqa: BLE001
                        print(f"      artefak model gagal disimpan: {exc}")
    elif model == "rf_recursive":
        ident_cols = [c for c in pooled.columns if c.endswith("_code")]
        ident_sku = pooled.drop_duplicates("sku_id").set_index("sku_id")[ident_cols]
        ident_by_sku = {s: {c: int(ident_sku.loc[s, c]) for c in ident_cols} for s in series}
        for tag, hari, dd in (("val", origin["train_end"], days_val), ("test", origin["val_end"], days_test)):
            m, _, _, n = train_pooled_1step(frames, hari, params, seed)
            print(f"      rf_recursive {tag}: latih {n:,} baris (<= {hari})")
            p = rf_recursive_predict(m, series, dd, H, ident_by_sku)
            p["split"] = tag
            parts.append(p[["sku_id", "day", "split", "pred_sum"]])
    elif model == "lightgbm_tweedie":
        import lightgbm as lgb

        p_cfg = cfg["lightgbm_v3"]
        for tag, hari, dd in (("val", origin["train_end"], days_val), ("test", origin["val_end"], days_test)):
            # CATATAN: jangan import GLOBAL_FEATURE_COLS/TARGET_COL di dalam fungsi ini. Import lokal
            # membuat nama itu LOKAL untuk seluruh fungsi, sehingga cabang rf_global_direct di atas
            # gagal dengan UnboundLocalError (bug nyata yang lolos uji lokal karena cabang artefak
            # ditambahkan belakangan). Keduanya diimpor di tingkat modul.
            mask = ((pooled["day"] <= hari) & pooled[GLOBAL_FEATURE_COLS].notna().all(axis=1)
                    & pooled[TARGET_COL].notna())
            tr = pooled.loc[mask]
            booster = lgb.LGBMRegressor(
                objective="tweedie", tweedie_variance_power=float(p_cfg["tweedie_variance_power"]),
                n_estimators=int(p_cfg["n_estimators"]), learning_rate=float(p_cfg["learning_rate"]),
                num_leaves=int(p_cfg["num_leaves"]), min_child_samples=int(p_cfg["min_child_samples"]),
                random_state=seed, n_jobs=-1, verbose=-1,
            ).fit(tr[GLOBAL_FEATURE_COLS], tr[TARGET_COL])
            sub = pooled.loc[pooled["day"].isin(dd) & pooled[GLOBAL_FEATURE_COLS].notna().all(axis=1)]
            p = pd.DataFrame({"sku_id": sub["sku_id"].to_numpy(), "day": sub["day"].to_numpy(),
                              "pred_sum": booster.predict(sub[GLOBAL_FEATURE_COLS])})
            p["split"] = tag
            parts.append(p)
            print(f"      lightgbm_tweedie {tag}: latih {len(tr):,} baris (<= {hari}) -> {len(p):,} prediksi")
    else:
        raise ValueError(f"model ML tidak dikenal: {model}")
    return pd.concat(parts, ignore_index=True)[["sku_id", "day", "split", "pred_sum"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "config_v3.yaml")
    ap.add_argument("--n-sku", type=int, default=300)
    ap.add_argument("--origins", default="1,2,3,4")
    ap.add_argument("--models", default=",".join(RATE_MODELS + ML_MODELS))
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--horizon", type=int, default=None, help="H periode proteksi (default dari config)")
    ap.add_argument("--resume", action="store_true", help="lewati berkas yang sudah ada")
    a = ap.parse_args()

    cfg = yaml.safe_load(a.config.read_text())
    H0 = int(cfg["policy_v3"]["protection_days"])
    H = int(a.horizon) if a.horizon else H0
    tag_h = "" if H == H0 else f"_h{H}"          # varian H diberi tanda di nama berkas
    res = ROOT / cfg["paths"]["results"]
    out = a.out or (res / "interim")
    out.mkdir(parents=True, exist_ok=True)

    sample = pd.read_csv(res / "tables" / "sku_sample_v3.csv").head(a.n_sku)
    sku_ids = sample["sku_id"].tolist()
    parquet = Path(cfg["paths"]["data_interim"]) / "sales_long.parquet"
    print(f"[forecast v3] {len(sku_ids)} SKU | parquet {parquet}")
    t0 = time.time()
    series = load_series(parquet, sku_ids)
    print(f"  riwayat dimuat dalam {time.time() - t0:.0f}s")

    models = [m.strip() for m in a.models.split(",") if m.strip()]
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    origins = [o for o in origin_windows(cfg) if o["origin"] in {int(x) for x in a.origins.split(",")}]

    need_ml = [m for m in models if m in ML_MODELS + ML_MODELS_GELOMBANG_2]
    frames = pooled = mapping = None
    if need_ml:
        t0 = time.time()
        frames = [build_features_v3(series[s], s, horizon=H) for s in sku_ids]
        pooled, mapping = encode_pooled(frames)
        print(f"  fitur dibangun: {len(pooled):,} baris x {len(pooled.columns)} kolom dalam {time.time() - t0:.0f}s")

    log_path = out / "forecast_log.txt"
    for origin in origins:
        print(f"\n=== origin {origin['origin']}: latih <= {origin['train_end']} | val {origin['days_val'][0]}..{origin['days_val'][-1]} "
              f"| uji {origin['days_test'][0]}..{origin['days_test'][-1]}")
        for model in models:
            is_rate = model in RATE_MODELS
            for seed in ([0] if is_rate else seeds):
                target = out / f"pred_{model}{tag_h}_o{origin['origin']}_s{seed}.csv.gz"
                if a.resume and target.exists():
                    print(f"  [lewati] {target.name}")
                    continue
                t0 = time.time()
                if is_rate:
                    pred = _rate_model_preds(model, series, origin["days_val"], origin["days_test"], H, cfg)
                else:
                    pred = _ml_preds(model, series, frames, pooled, origin, cfg, seed, H, mapping,
                                     origin_max=max(o["origin"] for o in origins))
                pred = pred.sort_values(["sku_id", "split", "day"])
                pred.to_csv(target, index=False, compression="gzip")
                dur = time.time() - t0
                n_nan = int(pred["pred_sum"].isna().sum())
                with log_path.open("a") as fh:
                    fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {model} o{origin['origin']} s{seed} "
                             f"rows={len(pred)} nan={n_nan} {dur:.1f}s -> {target.name}\n")
                print(f"  {model:24s} o{origin['origin']} s{seed} -> {len(pred):,} baris ({n_nan} NaN) {dur:.1f}s")
    print("\n[selesai] berkas di", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
