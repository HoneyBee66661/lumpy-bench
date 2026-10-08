"""run_all_v3.py — orkestrator lokal v3 (Fase 4-5): kebijakan, biaya, statistik, fakta, manifest.

Bagian berat (latih model, walk-forward 300 SKU) dijalankan di Kaggle oleh `run_forecasts_v3.py`.
Skrip ini memakai keluaran itu (`results_v3/interim/pred_*.csv.gz`) untuk:

  stage policy : error validasi -> safety stock kuantil empiris -> sapuan tau -> kurva biaya vs fill
                 rate -> tabel service-equalized
  stage stats  : Wilcoxon berpasangan (satu sisi, arah pra-registrasi) + Holm, Hodges-Lehmann +
                 CI cluster bootstrap (toko/departemen), Friedman lintas skenario (H2), heterogenitas,
                 metrik MASE/RMSSE/ME/pinball
  stage sens   : Fase 5 — sensitivitas L, R, konvensi biaya, pembulatan order, warm-up, tanpa SKU
                 regime-shift (memakai prediksi yang sama, hanya kebijakan/simulasi yang berubah)
  stage facts  : results_v3/article_facts_v3.txt (semua angka kunci + konvensi + jejak berkas)
  stage pack   : manifest SHA-256 seluruh berkas results_v3 + ringkasan untuk laporan

Pakai:  .venv/bin/python src_v3/run_all_v3.py --stages policy,stats,facts
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src_v3"))
from forecast_v3 import protection_actual  # noqa: E402
from policy_v3 import (  # noqa: E402
    add_costs,
    billing_window,
    run_sweep,
    safety_stock_table,
    sample_trace,
    service_equalized,
)
from simulator_v3 import Policy  # noqa: E402
from stats_v3 import (  # noqa: E402
    forecast_metrics,
    friedman_interaksi,
    heterogenitas,
    pairwise_service_equalized,
)

COLS = ["item_id", "store_id", "day", "date", "demand", "snap", "event_name_1", "event_name_2", "sell_price"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_daily(parquet: Path, sku_ids: list[str], cache: dict | None = None) -> dict[str, pd.DataFrame]:
    if cache is not None and "daily" in cache:
        return cache["daily"]
    items = sorted({s.split("__")[0] for s in sku_ids})
    stores = sorted({s.split("__")[1] for s in sku_ids})
    tbl = ds.dataset(parquet, format="parquet").to_table(
        columns=COLS, filter=(ds.field("item_id").isin(items) & ds.field("store_id").isin(stores))
    )
    df = tbl.to_pandas()
    df["sku_id"] = df["item_id"].astype(str) + "__" + df["store_id"].astype(str)
    df = df[df["sku_id"].isin(set(sku_ids))]
    daily = {s: g.sort_values("day").reset_index(drop=True) for s, g in df.groupby("sku_id", sort=False)}
    if cache is not None:
        cache["daily"] = daily
    return daily


def origin_windows(cfg: dict, only: list[int] | None = None) -> list[dict]:
    total = int(cfg.get("total_days", 1941))
    test_days = int(cfg["origin"]["test_days"])
    val_days = int(cfg["split"]["validation_days"])
    n_origin = int(cfg["origin"]["n_origin"])
    out = []
    for i in range(1, n_origin + 1):
        if only and i not in only:
            continue
        val_end = total - test_days * (n_origin - i + 1)
        out.append({"origin": i, "train_end": val_end - val_days, "val_end": val_end,
                    "days_val": np.arange(val_end - val_days + 1, val_end + 1),
                    "days_test": np.arange(val_end + 1, min(val_end + 1 + test_days, total + 1))})
    return out


def policy_of(cfg: dict, **over) -> Policy:
    p = cfg["policy_v3"]
    kw = dict(
        review_period_days=int(p["review_period_days"]), lead_time_days=int(p["lead_time_days"]),
        stockout_cost=1.0, holding_cost=1.0, cost_convention=p["cost_convention_utama"],
        initial_rule=p["initial_rule"], rounding=p["order_rounding"], moq=float(p["moq"]),
        warmup_days=int(p["warmup_tidak_dihitung"]),
    )
    kw.update(over)
    return Policy(**kw)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


PRED_RE = re.compile(r"^pred_(?P<model>.+?)(?P<h>_h\d+)?_o(?P<origin>\d+)_s(?P<seed>\d+)\.csv\.gz$")


def model_whitelist() -> set[str]:
    """Nama model yang sah = sumber tunggal di run_forecasts_v3 (hindari drift nama berkas)."""
    from run_forecasts_v3 import ML_MODELS, ML_MODELS_GELOMBANG_2, RATE_MODELS

    return set(RATE_MODELS + ML_MODELS + ML_MODELS_GELOMBANG_2)


def parse_pred_name(nama: str) -> dict:
    """Nama berkas prediksi -> (model, horizon, origin, seed).

    Parsing harus regex, bukan split('_o'): nama model bisa memuat '_o' (croston_sba_optimized) —
    bug nyata yang ketangkap saat uji pipa (split menghasilkan origin 'ptimized').
    Nama model juga diperiksa terhadap daftar model yang sah supaya salah nama gagal keras.
    """
    m = PRED_RE.match(nama)
    if not m:
        raise ValueError(f"nama berkas prediksi tidak dikenali: {nama}")
    if m.group("model") not in model_whitelist():
        raise ValueError(f"nama model tidak dikenal di berkas: {nama} (model '{m.group('model')}')")
    return {"model": m.group("model"),
            "horizon": int(m.group("h")[2:]) if m.group("h") else None,
            "origin": int(m.group("origin")), "seed": int(m.group("seed"))}


def key_of(pred_file: Path) -> str:
    """Kunci model untuk tabel: model polos untuk rate model (seed 0), `model#s<seed>` untuk ML."""
    d = parse_pred_name(pred_file.name)
    return d["model"] if d["seed"] == 0 else f"{d['model']}#s{d['seed']}"


def resolve_model_key(kolom: list[str], diminta: str) -> str | None:
    """Cari nama kolom model yang cocok dengan `diminta` (mis. rf_global_direct -> rf_global_direct#s42)."""
    if diminta in kolom:
        return diminta
    kandidat = sorted(c for c in kolom if c.split("#")[0] == diminta)
    return kandidat[0] if kandidat else None


# --------------------------------------------------------------------------- stage: policy
def stage_policy(cfg: dict, args) -> None:
    res = ROOT / cfg["paths"]["results"]
    interim = res / "interim"
    tables = res / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    sample = pd.read_csv(tables / "sku_sample_v3.csv").head(args.n_sku)
    sku_ids = sample["sku_id"].tolist()
    H = int(cfg["policy_v3"]["protection_days"])
    q_levels = [float(t) for t in cfg["policy_v3"]["quantile_levels"]]
    scenarios = [(float(cs), float(ch)) for cs, ch in cfg["cost_scenarios"]]
    parquet = Path(cfg["paths"]["data_interim"]) / "sales_long.parquet"
    log(f"memuat riwayat {len(sku_ids)} SKU dari {parquet}")
    daily = load_daily(parquet, sku_ids)
    actual = {s: protection_actual(daily[s], H).set_index("day")["actual_sum"] for s in sku_ids}
    demand_by_sku = {s: daily[s].set_index("day")["demand"] for s in sku_ids}

    meta = sample[["sku_id", "cat_id", "dept_id", "state_id", "adi", "cv2"]].copy()
    meta["store_id"] = meta["sku_id"].str.split("__").str[1]
    meta["dept_id"] = meta["sku_id"].str.split("__").str[0].str.rsplit("_", n=1).str[0]
    meta["volume"] = [float(demand_by_sku[s].sum()) for s in sku_ids]
    for col, nama in (("adi", "adi_kuartil"), ("cv2", "cv2_kuartil"), ("volume", "volume_kuartil")):
        meta[nama] = pd.qcut(meta[col], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    meta.to_csv(tables / "sample_meta_v3.csv", index=False)

    for origin in origin_windows(cfg, args.origins):
        o = origin["origin"]
        preds = {}
        for f in sorted(interim.glob(f"pred_*_o{o}_s*.csv.gz")):
            if parse_pred_name(f.name)["horizon"] is not None:
                continue                       # varian H lain dipakai di stage sensitivitas
            preds[key_of(f)] = pd.read_csv(f)
        if not preds:
            log(f"origin {o}: tidak ada berkas prediksi — lewati (jalankan kernel Kaggle dulu)")
            continue
        log(f"origin {o}: {len(preds)} model/seed | uji {origin['days_test'][0]}..{origin['days_test'][-1]}")

        ss_tables_all: dict[str, dict] = {}
        err_tables = {}
        for key, d in preds.items():
            v = d[d.split == "val"].set_index(["sku_id", "day"])["pred_sum"]
            rows = []
            for s in sku_ids:
                p = v.loc[s] if s in v.index.get_level_values(0) else None
                if p is None:
                    continue
                e = (actual[s].reindex(p.index) - p).dropna()
                rows.append(pd.DataFrame({"sku_id": s, "err": e.to_numpy(),
                                          "cat_id": meta.set_index("sku_id").loc[s, "cat_id"]}))
            err = pd.concat(rows, ignore_index=True)
            err.to_csv(interim / f"err_val_{key}_o{o}.csv.gz", index=False, compression="gzip")
            ss_all, diag = safety_stock_table(err, q_levels, group_col="cat_id")
            for tau, tbl in ss_all.items():
                tbl.insert(0, "origin", o)
                tbl.insert(1, "model", key)
            diag.insert(0, "origin", o)
            diag.insert(1, "model", key)
            err_tables[key] = diag
            ss_tables_all[key] = ss_all

        diag_all = pd.concat(err_tables.values(), ignore_index=True)
        diag_path = tables / f"ss_diagnostik_o{o}.csv"
        diag_all.to_csv(diag_path, index=False)
        log(f"  SS kuantil empiris: {diag_path.name}")

        pol = policy_of(cfg)
        sweeps = []
        for key, d in preds.items():
            sw = run_sweep(d[d.split == "test"], ss_tables_all[key], demand_by_sku, origin["days_test"], pol, key)
            sweeps.append(sw)
            log(f"    sapuan {key:28s} {len(sw):6,d} baris")
        sweep = pd.concat(sweeps, ignore_index=True)
        sweep.to_csv(tables / f"policy_sweep_o{o}.csv.gz", index=False, compression="gzip")

        costs = add_costs(sweep, scenarios, cfg["policy_v3"]["cost_convention_utama"])
        costs = costs.merge(meta[["sku_id", "store_id", "dept_id", "cat_id", "adi", "cv2", "adi_kuartil", "cv2_kuartil", "volume_kuartil"]],
                            on="sku_id", how="left")
        costs.to_csv(tables / f"costs_o{o}.csv.gz", index=False, compression="gzip")
        eq = service_equalized(costs, [float(f) for f in cfg["policy_v3"]["fill_rate_targets"]])
        eq = eq.merge(meta[["sku_id", "store_id", "dept_id", "cat_id", "adi", "cv2", "adi_kuartil", "cv2_kuartil", "volume_kuartil"]],
                      on="sku_id", how="left")
        eq.to_csv(tables / f"service_equalized_o{o}.csv.gz", index=False, compression="gzip")
        log(f"  service-equalized o{o}: {len(eq):,} baris (fill target {cfg['policy_v3']['fill_rate_targets']})")


# --------------------------------------------------------------------------- stage: stats
def stage_stats(cfg: dict, args) -> None:
    res = ROOT / cfg["paths"]["results"]
    tables = res / "tables"
    rows_pw, rows_fr, rows_het, rows_met = [], [], [], []
    for f in sorted(tables.glob("service_equalized_o*.csv.gz")):
        o = int(f.name.split("_o")[-1].replace(".csv.gz", ""))
        eq = pd.read_csv(f)
        kolom = list(eq.model.unique())
        ma = resolve_model_key(kolom, args.model_a)
        mb = resolve_model_key(kolom, args.model_b)
        if ma is None or mb is None:
            log(f"  statistik o{o}: model {args.model_a}/{args.model_b} tidak ada di tabel — lewati")
            continue
        if ma != args.model_a or mb != args.model_b:
            log(f"  statistik o{o}: kunci model dipetakan {args.model_a}->{ma}, {args.model_b}->{mb}")
        pw = pairwise_service_equalized(eq, model_a=ma, cluster_cols=("store_id", "dept_id"),
                                        n_boot=args.n_boot)
        if len(pw):
            pw.insert(0, "origin", o)
            rows_pw.append(pw)
            # selisih model_a vs model_b per skenario -> uji interaksi (H2)
            sel = {}
            for cs in sorted(eq.cs.unique()):
                g = eq[(eq.cs == cs) & (eq.fill_target == cfg["policy_v3"]["fill_rate_targets"][1])]
                piv = g.pivot_table(index="sku_id", columns="model", values="cost_at_target")
                if ma in piv.columns and mb in piv.columns:
                    sel[f"cs{cs:g}"] = (piv[ma] - piv[mb])
            if sel:
                fr = friedman_interaksi(pd.DataFrame(sel))
                fr.update({"origin": o, "model_a": ma, "model_b": mb})
                rows_fr.append(fr)
            for cs in sorted(eq.cs.unique()):
                het = heterogenitas(eq, pd.read_csv(tables / "sample_meta_v3.csv"), model_a=ma,
                                    model_b=mb, cs=float(cs),
                                    fill_target=float(cfg["policy_v3"]["fill_rate_targets"][1]))
                if len(het):
                    het.insert(0, "origin", o)
                    het.insert(1, "cs", float(cs))
                    rows_het.append(het)
        log(f"  statistik o{o}: {len(pw)} perbandingan berpasangan")
    if rows_pw:
        pd.concat(rows_pw, ignore_index=True).to_csv(tables / "stats_pairwise_v3.csv", index=False)
    if rows_fr:
        pd.DataFrame(rows_fr).to_csv(tables / "stats_friedman_v3.csv", index=False)
    if rows_het:
        pd.concat(rows_het, ignore_index=True).to_csv(tables / "stats_heterogenitas_v3.csv", index=False)

    # metrik akurasi (MASE/RMSSE/ME/pinball) dari prediksi val+uji vs aktual
    H = int(cfg["policy_v3"]["protection_days"])
    sample = pd.read_csv(tables / "sku_sample_v3.csv").head(args.n_sku)
    sku_ids = sample["sku_id"].tolist()
    parquet = Path(cfg["paths"]["data_interim"]) / "sales_long.parquet"
    daily = load_daily(parquet, sku_ids)
    actual_sum = {s: protection_actual(daily[s], H).set_index("day")["actual_sum"] for s in sku_ids}
    val_end_of = {o["origin"]: o["val_end"] for o in origin_windows(cfg)}
    for f in sorted((res / "interim").glob("pred_*.csv.gz")):
        info = parse_pred_name(f.name)
        if info["horizon"] is not None:
            continue
        model, o, seed = info["model"], info["origin"], info["seed"]
        d = pd.read_csv(f)
        for (s, split), g in d.groupby(["sku_id", "split"]):
            if s not in actual_sum:
                continue
            a = actual_sum[s].reindex(g.day).to_numpy()
            # skala naive MASE/RMSSE memakai target pada data LATIH saja (<= val_end origin ini)
            train_target = actual_sum[s].loc[: val_end_of[o]].dropna().to_numpy()
            m = forecast_metrics(a, g.pred_sum.to_numpy(), train_target, tau=0.95)
            rows_met.append({"model": model, "seed": seed, "origin": o, "split": split, "sku_id": s, **m})
    if rows_met:
        met = pd.DataFrame(rows_met)
        met.to_csv(tables / "metrics_akurasi_v3.csv.gz", index=False, compression="gzip")
        piv = (met.groupby(["model", "origin", "split"])
               .agg(mae=("mae", "median"), mase=("mase", "median"), rmsse=("rmsse", "median"),
                    me=("me", "median"), pinball=("pinball", "median"), n_sku=("sku_id", "nunique"))
               .reset_index())
        piv.to_csv(tables / "metrics_rekap_v3.csv", index=False)
        log(f"  metrik akurasi: {len(met):,} baris SKU-model; rekap {piv.shape[0]} baris")


# --------------------------------------------------------------------------- stage: sens
def stage_sens(cfg: dict, args) -> None:
    res = ROOT / cfg["paths"]["results"]
    tables = res / "tables"
    interim = res / "interim"
    sample = pd.read_csv(tables / "sku_sample_v3.csv").head(args.n_sku)
    sku_ids = sample["sku_id"].tolist()
    meta_map = sample.set_index("sku_id")["cat_id"].to_dict()
    H0 = int(cfg["policy_v3"]["protection_days"])
    parquet = Path(cfg["paths"]["data_interim"]) / "sales_long.parquet"
    daily = load_daily(parquet, sku_ids)

    def protection_actual_series(d, h: int) -> dict:
        return {s: protection_actual(d[s], h).set_index("day")["actual_sum"] for s in sku_ids}

    demand_by_sku = {s: daily[s].set_index("day")["demand"] for s in sku_ids}
    scenarios = [(float(cs), float(ch)) for cs, ch in cfg["cost_scenarios"]]
    base_R = int(cfg["policy_v3"]["review_period_days"])
    base_L = int(cfg["policy_v3"]["lead_time_days"])
    variants = [
        ("L3", dict(lead_time_days=3)),
        ("L7", dict(lead_time_days=7)),
        ("L14", dict(lead_time_days=14)),
        ("R7", dict(review_period_days=7)),
        ("konvensi_per_unit_day", dict(cost_convention="per_unit_day")),
        ("awal_nol", dict(initial_rule="zero")),
        ("integer_ceil", dict(rounding="integer_ceil")),
        ("moq_16", dict(rounding="moq", moq=16.0)),
        ("warmup_56", dict(warmup_days=56)),
        ("tanpa_batas_penagihan", dict(bill_until_day=None)),
    ]
    rows = []
    for origin in origin_windows(cfg, args.origins):
        o = origin["origin"]
        preds = {}
        for f in sorted(interim.glob(f"pred_*_o{o}_s*.csv.gz")):
            if parse_pred_name(f.name)["horizon"] is not None:
                continue
            preds[key_of(f)] = pd.read_csv(f)
        if not preds:
            continue
        # SS dari error validasi (dipakai semua varian pada tau utama 0.95 dan tau terpilih)
        ss_by_model = {}
        for key, d in preds.items():
            p = Path(interim / f"err_val_{key}_o{o}.csv.gz")
            if not p.exists():
                continue
            err = pd.read_csv(p)
            ss_all, _ = safety_stock_table(err, [0.90, 0.95], group_col="cat_id")
            ss_by_model[key] = ss_all
        for nama, over in variants:
            H_var = int(over.get("review_period_days", base_R)) + int(over.get("lead_time_days", base_L))
            pol = policy_of(cfg, **over)
            if H_var != H0:
                # H berbeda -> perlu forecast dengan H baru. Dipakai berkas `pred_*_h{H}_o{o}_s*.csv.gz`
                # bila kernel sensitivitas sudah dijalankan; kalau tidak ada, dilaporkan sebagai batasan
                # (tidak dihitung dari prediksi H=8 karena itu mengarang angka).
                berkas_h = sorted(interim.glob(f"pred_*_h{H_var}_o{o}_s*.csv.gz"))
                if not berkas_h:
                    rows.append({"origin": o, "varian": nama,
                                 "status": f"tidak dihitung (H={H_var}: berkas pred_*_h{H_var}_o{o} tidak ada)"})
                    continue
                for fh in berkas_h:
                    dh = pd.read_csv(fh)
                    key_h = fh.name[len("pred_"):].split("_o")[0]
                    pval = dh[dh.split == "val"]
                    if pval.empty:
                        rows.append({"origin": o, "varian": nama,
                                     "status": f"{key_h}: tidak ada split validasi"})
                        continue
                    H_act = protection_actual_series(daily, H_var)
                    err_rows = []
                    for s in sku_ids:
                        p = pval[pval.sku_id == s]
                        if p.empty:
                            continue
                        e = (H_act[s].reindex(p.day) - p.set_index("day")["pred_sum"]).dropna()
                        err_rows.append(pd.DataFrame({"sku_id": s, "err": e.to_numpy(), "cat_id": meta_map[s]}))
                    if not err_rows:
                        continue
                    ss_all, _ = safety_stock_table(pd.concat(err_rows, ignore_index=True),
                                                   [0.90, 0.95], group_col="cat_id")
                    for tau in sorted(ss_all):
                        sw = run_sweep(dh[dh.split == "test"], {tau: ss_all[tau]}, demand_by_sku,
                                       origin["days_test"], pol, key_h)
                        sw["tau"] = tau
                        sw["varian"] = nama
                        sw["origin"] = o
                        rows.append(sw)
                continue
            for key, d in preds.items():
                if key not in ss_by_model:
                    continue
                for tau in sorted(ss_by_model[key]):
                    sw = run_sweep(d[d.split == "test"], {tau: ss_by_model[key][tau]}, demand_by_sku,
                                   origin["days_test"], pol, key)
                    sw["tau"] = tau
                    sw["varian"] = nama
                    sw["origin"] = o
                    rows.append(sw)
        del preds
        log(f"  sensitivitas o{o} selesai")
    if rows:
        out = pd.concat([pd.DataFrame([r]) if isinstance(r, dict) else r for r in rows], ignore_index=True)
        out.to_csv(tables / "sensitivitas_v3.csv.gz", index=False, compression="gzip")
        log(f"  sensitivitas: {len(out):,} baris -> sensitivitas_v3.csv.gz")


# --------------------------------------------------------------------------- stage: facts
def stage_facts(cfg: dict, args) -> None:
    res = ROOT / cfg["paths"]["results"]
    tables = res / "tables"
    lines = []
    add = lines.append
    add("FAKTA ANGKA v3 — RF vs CROSTON-SBA, LUMPY DEMAND, BIAYA ASIMETRIS")
    add(f"dibuat: {time.strftime('%Y-%m-%d %H:%M:%S')} | pipeline: v3 (simulator net + forecast direct multi-horizon)")
    add(f"config: config_v3.yaml (seed {cfg['seed']}, seed RF dijalankan {cfg['seeds_rf_dijalankan']})")
    add("")
    add("KONVENSI ARAH (WAJIB DIBACA SEBELUM MENGUTIP ANGKA)")
    add("  * r_rank_biserial = (W_neg - W_pos)/(W_pos + W_neg) dari d = biaya_model_a - biaya_model_b.")
    add("    r POSITIF berarti model A (RF) LEBIH MURAH. Uji utama satu sisi (a lebih murah).")
    add("  * biaya konvensi utama: per_unit_once (unit tidak terlayani dihitung sekali).")
    add("    Konvensi alternatif per_unit_day tersedia sebagai sensitivitas.")
    add("  * hari yang dibebankan = hari yang jendela proteksinya (t+1..t+H) lengkap di dalam data;")
    add("    H = R + L = 8; warm-up 28 hari di awal periode uji tidak dibebankan.")
    add("")
    for nama in ("sku_sample_v3.csv", "lumpy_v3.csv"):
        p = tables / nama
        if p.exists():
            df = pd.read_csv(p)
            add(f"SAMPEL/KLASIFIKASI ({nama}): {len(df):,} baris")
    p = tables / "sample_meta_v3.csv"
    if p.exists():
        sm = pd.read_csv(p)
        add(f"  kategori: {sm.cat_id.value_counts().to_dict()}")
        add(f"  state: {sm.state_id.value_counts().to_dict()}")
        add(f"  ADI median {sm.adi.median():.2f} (min {sm.adi.min():.2f}, maks {sm.adi.max():.2f}) | "
            f"CV2 median {sm.cv2.median():.2f}")
    add("")
    p = tables / "metrics_rekap_v3.csv"
    if p.exists():
        mr = pd.read_csv(p)
        add("AKURASI (median antar SKU, split=test) — metrik jumlah periode proteksi:")
        t = mr[mr.split == "test"].sort_values(["origin", "mae"])
        for _, r in t.iterrows():
            add(f"  o{int(r.origin)} {r.model:26s} MAE {r.mae:8.4f} | MASE {r.mase:6.3f} | RMSSE {r.rmsse:6.3f} "
                f"| ME {r.me:8.4f} | pinball {r.pinball:8.4f} | n_sku {int(r.n_sku)}")
        add("")
    p = tables / "stats_pairwise_v3.csv"
    if p.exists():
        pw = pd.read_csv(p)
        add("STATISTIK BERPASANGAN (Holm dalam keluarga uji per origin):")
        kol = ["origin", "cs", "fill_target", "model_b", "n_sku", "median_a", "median_b", "median_diff",
               "pct_diff_median", "hodges_lehmann", "ci_store_id_lo", "ci_store_id_hi",
               "p_one_sided_a_lebih_murah", "p_one_sided_holm", "r_rank_biserial", "menang_a", "menang_b"]
        for _, r in pw[[c for c in kol if c in pw.columns]].iterrows():
            add(f"  o{int(r.origin)} cs{int(r.cs)} fill{int(r.fill_target*100)}% vs {r.model_b:24s} "
                f"median {r.median_diff:9.2f} ({r.pct_diff_median:+6.2f}%) HL {r.hodges_lehmann:9.2f} "
                f"p {r.p_one_sided_a_lebih_murah:.4f} p_Holm {r.p_one_sided_holm:.4f} r {r.r_rank_biserial:+.3f} "
                f"menang {int(r.menang_a)}/{int(r.n_sku)}")
        add("")
    p = tables / "stats_friedman_v3.csv"
    if p.exists():
        fr = pd.read_csv(p)
        add("H2 (interaksi skenario biaya, Friedman per SKU):")
        for _, r in fr.iterrows():
            add(f"  o{int(r.origin)} {r.model_a} vs {r.model_b}: chi2 {r.statistic:.3f} p {r.p_value:.4f} n_sku {int(r.n_sku)}")
        add("")
    for nama in ("stats_heterogenitas_v3.csv",):
        p = tables / nama
        if p.exists():
            ht = pd.read_csv(p)
            add(f"HETEROGENITAS ({nama}) pada fill 95%:")
            for _, r in ht.iterrows():
                add(f"  o{int(r.origin)} cs{int(r.cs)} {r.dimensi}={r.nilai:12s} n {int(r.n_sku):3d} "
                    f"selisih relatif median {r.median_relatif_pct:+7.2f}% p {r.p_one_sided:.4f} "
                    f"p_Holm {r.p_holm_dalam_dimensi:.4f}")
            add("")
    p = tables / "sensitivitas_v3.csv.gz"
    if p.exists():
        sv = pd.read_csv(p)
        add("SENSITIVITAS (variasi kebijakan, biaya, konvensi):")
        if "varian" in sv.columns and "cost_total" not in sv.columns:
            add("  varian yang butuh forecast H baru tidak dihitung (dilaporkan sebagai batasan)")
        else:
            g = (sv.groupby(["varian", "model"])
                 .agg(fill=("fill_rate", "median"), unmet=("unmet", "median"), on_hand=("on_hand", "median"),
                      hari_atas=("hari_on_hand_di_atas_level", "median"), n=("sku_id", "size")).reset_index())
            for _, r in g.iterrows():
                add(f"  {r.varian:24s} {r.model:26s} fill {r.fill:.4f} unmet {r.unmet:8.2f} on_hand {r.on_hand:10.2f} "
                    f"hari_atas_level {r.hari_atas:6.1f}")
        add("")
    p = tables / "ss_diagnostik_o4.csv"
    if p.exists():
        sd = pd.read_csv(p)
        add("SAFETY STOCK (kuantil empiris error validasi, o4):")
        for _, r in sd.iterrows():
            add(f"  {r.model:26s} tau {r.tau:.3f} ss_median {r.ss_median:9.3f} "
                f"(min {r.ss_min:9.3f}, maks {r.ss_max:9.3f}) pooled {int(r.n_sku_pooled)}")
        add("")
    files = sorted([p for p in (res).rglob("*") if p.is_file()])
    man = res / "manifest_v3.txt"
    with man.open("w") as fh:
        fh.write(f"# manifest v3 — {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        fh.write(f"# {len(files)} berkas di {res}\n")
        for p in files:
            if p.name == "manifest_v3.txt":
                continue
            fh.write(f"{sha256(p)}  {p.relative_to(res)}  {p.stat().st_size}\n")
    add(f"MANIFEST: {man.relative_to(ROOT)} ({len(files)} berkas)")
    (res / "article_facts_v3.txt").write_text("\n".join(lines) + "\n")
    log(f"  fakta -> {(res / 'article_facts_v3.txt').relative_to(ROOT)} ({len(lines)} baris)")
    if args.git_commit:
        try:
            out = subprocess.run(["git", "add", "-A"], cwd=ROOT, capture_output=True, text=True)
            out = subprocess.run(["git", "commit", "-q", "-m", args.git_commit], cwd=ROOT, capture_output=True, text=True)
            log(f"  git: {(out.stdout or out.stderr or 'ok').strip()[:200]}")
        except Exception as exc:  # noqa: BLE001
            log(f"  git gagal: {exc}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "config_v3.yaml")
    ap.add_argument("--stages", default="policy,stats,sens,facts")
    ap.add_argument("--n-sku", type=int, default=300)
    ap.add_argument("--origins", type=int, nargs="*", default=None)
    ap.add_argument("--model-a", default="rf_global_direct")
    ap.add_argument("--model-b", default="croston_sba")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--no-git", dest="git_commit", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    for st in stages:
        fn = {"policy": stage_policy, "stats": stage_stats, "sens": stage_sens, "facts": stage_facts}.get(st)
        if fn is None:
            raise SystemExit(f"stage tidak dikenal: {st}")
        log(f"=== stage {st}")
        fn(cfg, args)
    log("selesai")
    return 0


if __name__ == "__main__":
    sys.exit(main())
