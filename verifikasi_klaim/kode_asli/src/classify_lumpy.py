"""5.3 Klasifikasi & sampling SKU lumpy (Syntetos dkk., 2005).

ADI   = jumlah hari dalam periode / jumlah hari dengan demand > 0
CV^2  = (std/mean)^2 dari demand pada hari dengan demand > 0 SAJA

Dihitung HANYA dari hari 1..(total_hari - test_days) — data uji tidak boleh dilihat
(spec Bagian 5.3, guardrail Bagian 9: dilarang tuning di data uji).

Catatan implementasi (host ini RAM 2 GB): statistik dihitung **streaming per batch
pyarrow** dengan akumulator berukuran tetap, bukan dengan memuat 59 juta baris ke pandas.
Rumusnya sama persis, dan kesetaraannya diuji di `tests/test_classify_lumpy.py` terhadap
`compute_adi_cv2()` (implementasi langsung, dipertahankan sebagai referensi/pembanding).

Akumulasi memakai bilangan bulat eksak (n_days, n_nonzero, sum_nz, sumsq_nz) supaya hasil
TIDAK bergantung urutan atau pembagian batch, dan semua agregasi dikerjakan di Arrow
(`Table.group_by`) — pandas hanya pernah menyentuh tabel ringkas <= 305 ribu baris.

Output:
  data/interim/lumpy_all.csv                 (semua SKU yang lolos filter)
  results/tables/all_sku_stats.csv           (ADI/CV^2 seluruh SKU)
  results/tables/sku_sample.csv              (n_sku terpilih, seed dari config)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import yaml

MIN_NONZERO_DAYS = 3  # SKU tanpa riwayat demand tidak bisa dinilai: dilaporkan, bukan dipaksa lolos
STAT_COLS = ["item_id", "store_id", "n_days", "n_nonzero", "mean_pos", "std_pos", "adi", "cv2"]
KEY = ["item_id", "store_id"]
ACC_COLS = ["n_days", "n_nonzero", "sum_nz", "sumsq_nz"]


def load_config(path: Path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def compute_adi_cv2(df: pd.DataFrame) -> pd.DataFrame:
    """Implementasi langsung (referensi) atas df yang sudah dibatasi ke hari latih+validasi.

    df: kolom item_id, store_id, demand. Dipakai tests/ sebagai pembanding versi streaming.
    """
    g = df.groupby(KEY, observed=True)
    n_days = g["demand"].size().rename("n_days")

    pos = df[df["demand"] > 0]
    gp = pos.groupby(KEY, observed=True)["demand"]
    n_nonzero = gp.size().rename("n_nonzero")
    mean_pos = gp.mean().rename("mean_pos")
    std_pos = gp.std(ddof=1).rename("std_pos")

    out = pd.concat([n_days, n_nonzero, mean_pos, std_pos], axis=1).reset_index()
    out["n_nonzero"] = out["n_nonzero"].fillna(0).astype("int32")
    out["adi"] = np.where(
        out["n_nonzero"] > 0, out["n_days"] / out["n_nonzero"].replace(0, np.nan), np.nan
    )
    out["cv2"] = np.where(
        (out["n_nonzero"] > 1) & (out["mean_pos"] > 0),
        (out["std_pos"] / out["mean_pos"]) ** 2,
        np.nan,
    )
    return out[STAT_COLS]


def finalize_stats(acc: pd.DataFrame) -> pd.DataFrame:
    """Akumulator integer -> ADI/CV^2 float, dengan rumus identik `compute_adi_cv2`.

    var(ddof=1) = (n*sumsq - sum^2) / (n*(n-1));  cv2 = var / mean^2;  adi = n_days / n
    """
    stats = acc.reset_index()
    for col in ACC_COLS:
        stats[col] = stats[col].fillna(0).astype("int64")

    n = stats["n_nonzero"]
    mean = stats["sum_nz"] / n.replace(0, np.nan)
    var = (n * stats["sumsq_nz"] - stats["sum_nz"] ** 2) / (n * (n - 1)).replace(0, np.nan)
    stats["mean_pos"] = mean.astype("float64")
    stats["std_pos"] = np.sqrt(var.clip(lower=0))
    stats["adi"] = np.where(n > 0, stats["n_days"] / n.replace(0, np.nan), np.nan)
    stats["cv2"] = np.where(
        (n > 1) & (stats["mean_pos"] > 0),
        (var / mean**2).clip(lower=0),
        np.nan,
    )
    return stats[STAT_COLS]


def total_days(src: Path) -> int:
    """Hari maksimum di parquet, diambil dari statistik metadata (tanpa membaca data)."""
    pf = pq.ParquetFile(src)
    col = pf.schema_arrow.names.index("day")
    md = pf.metadata
    return int(max(md.row_group(i).column(col).statistics.max for i in range(md.num_row_groups)))


def _accumulate(acc: pd.DataFrame | None, batch: pa.RecordBatch) -> pd.DataFrame:
    """Tambahkan satu batch ke akumulator. Memori terikat tabel ringkas, bukan ukuran batch."""
    tbl = pa.Table.from_batches([batch])
    demand = tbl.column("demand")

    tot = (
        tbl.group_by(KEY, use_threads=False)
        .aggregate([("demand", "count")])
        .rename_columns(KEY + ["n_days"])
        .to_pandas()
    )

    pos = tbl.filter(pc.greater(demand, 0))
    if pos.num_rows:
        d = pc.cast(pos.column("demand"), pa.int64())
        pos = pos.append_column("demand_sq", pc.multiply(d, d))
        agg = (
            pos.group_by(KEY, use_threads=False)
            .aggregate([("demand", "count"), ("demand", "sum"), ("demand_sq", "sum")])
            .rename_columns(KEY + ["n_nonzero", "sum_nz", "sumsq_nz"])
            .to_pandas()
        )
        part = tot.merge(agg, on=KEY, how="left")
    else:
        part = tot
        part["n_nonzero"] = 0
        part["sum_nz"] = 0
        part["sumsq_nz"] = 0

    for col in ACC_COLS:
        part[col] = part[col].fillna(0).astype("int64")
    part = part.set_index(KEY)

    if acc is None:
        return part
    return pd.concat([acc, part]).groupby(level=KEY).sum()


def stream_sku_stats(
    src: Path,
    day_max: int,
    batch_rows: int = 500_000,
    progress_every: int = 20,
) -> pd.DataFrame:
    """Statistik per (item_id, store_id) secara streaming: memori terikat, bukan O(n_baris).

    Akumulator berisi maksimal satu baris per SKU (<= 305 ribu baris untuk M5) dan TIDAK
    ada frame per batch yang ditahan sampai akhir — itu penyebab OOM pada versi sebelumnya.
    """
    dset = ds.dataset(src, format="parquet")
    scanner = dset.scanner(
        columns=["item_id", "store_id", "demand"],
        filter=ds.field("day") <= day_max,
        batch_size=batch_rows,
    )
    acc: pd.DataFrame | None = None
    n_rows = 0
    t0 = time.time()
    for i, batch in enumerate(scanner.to_batches(), start=1):
        acc = _accumulate(acc, batch)
        n_rows += batch.num_rows
        if i == 1 or i % progress_every == 0:
            print(
                f"  batch {i:4d}: {batch.num_rows:,} baris (total {n_rows:,})"
                f" | SKU {len(acc):,} | {time.time() - t0:.0f}s"
            )

    if acc is None:
        print(f"STOP: tidak ada baris dengan day <= {day_max} di {src}", file=sys.stderr)
        sys.exit(2)
    print(f"  selesai: {n_rows:,} baris, {len(acc):,} SKU, {time.time() - t0:.0f}s")
    return finalize_stats(acc)


def run(config_path: Path, batch_rows: int = 500_000) -> None:
    cfg = load_config(config_path)
    seed = int(cfg["seed"])
    test_days = int(cfg["split"]["test_days"])
    adi_min = float(cfg["lumpy_threshold"]["adi_min"])
    cv2_min = float(cfg["lumpy_threshold"]["cv2_min"])
    n_sku = int(cfg["sampling"]["n_sku"])

    interim = Path(cfg["paths"]["data_interim"])
    results = Path(cfg["paths"]["results"])
    (results / "tables").mkdir(parents=True, exist_ok=True)

    src = interim / "sales_long.parquet"
    if not src.exists():
        print(f"STOP: {src} tidak ada — jalankan src/data_prep.py dulu.", file=sys.stderr)
        sys.exit(2)

    total = total_days(src)
    train_val_end = total - test_days
    print(f"total hari = {total}; ADI/CV2 dihitung dari hari 1..{train_val_end} (streaming)")
    stats = stream_sku_stats(src, train_val_end, batch_rows)
    stats.to_csv(results / "tables" / "all_sku_stats.csv", index=False)

    eligible = stats[
        (stats["n_nonzero"] >= MIN_NONZERO_DAYS)
        & (stats["adi"] > adi_min)
        & (stats["cv2"] > cv2_min)
    ].copy()
    print(f"SKU total                      : {len(stats):,}")
    print(f"SKU lolos ADI>{adi_min} & CV2>{cv2_min}: {len(eligible):,}")

    n_take = min(n_sku, len(eligible))
    if n_take < n_sku:
        print(f"PERINGATAN: hanya {n_take} SKU memenuhi kriteria, diminta {n_sku}.")
    if n_take == 0:
        print("STOP: tidak ada SKU lumpy — periksa threshold/data.", file=sys.stderr)
        sys.exit(2)

    sample = eligible.sample(n=n_take, random_state=seed).sort_values(KEY).reset_index(drop=True)
    sample["sku_id"] = sample["item_id"].astype(str) + "__" + sample["store_id"].astype(str)

    eligible.to_csv(interim / "lumpy_all.csv", index=False)
    sample.to_csv(results / "tables" / "sku_sample.csv", index=False)
    print(f"OK -> {results / 'tables' / 'sku_sample.csv'} ({len(sample)} SKU, seed={seed})")
    print(sample[["sku_id", "adi", "cv2"]].head(10).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description="Filter + sampling SKU lumpy.")
    ap.add_argument("--config", type=Path, default=Path("config.yaml"))
    ap.add_argument("--batch-rows", type=int, default=500_000)
    a = ap.parse_args()
    run(a.config, a.batch_rows)


if __name__ == "__main__":
    main()
