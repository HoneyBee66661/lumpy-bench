"""5.1 Load & reshape M5 wide -> long, leakage-safe terhadap memori.

sales_train_evaluation.csv = 30.490 baris x 1941 kolom d_* -> bila di-melt
sekaligus jadi ~59 juta baris. Di host ini (14 GB sisa disk, RAM terbatas)
melt dilakukan per-chunk dan ditulis append ke satu file Parquet.

Output: data/interim/sales_long.parquet
  kolom: item_id, store_id, state_id, day (1..1941), date, wday, month,
         wm_yr_wk, event_name_1, event_type_1, event_name_2, event_type_2,
         snap, demand, sell_price (nullable)

Catatan integritas: TIDAK ada baris yang dibuat-buat. Bila file mentah tidak
lengkap, script berhenti dengan pesan jelas (spec Bagian 9).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SALES_FILE = "sales_train_evaluation.csv"
CALENDAR_FILE = "calendar.csv"
PRICES_FILE = "sell_prices.csv"


def _die(msg: str) -> None:
    print(f"STOP: {msg}", file=sys.stderr)
    sys.exit(2)


def load_calendar(raw: Path) -> pd.DataFrame:
    cal = pd.read_csv(raw / CALENDAR_FILE)
    cal["day"] = cal["d"].str.replace("d_", "", regex=False).astype("int32")
    snap_cols = [c for c in cal.columns if c.startswith("snap_")]
    # snap di-flatten: dipakai nanti di features.py sesuai state_id SKU
    cal["snap"] = 0
    for c in snap_cols:
        cal.loc[cal[c] == 1, "snap"] = 1
    keep = [
        "day", "date", "wday", "month", "wm_yr_wk",
        "event_name_1", "event_type_1", "event_name_2", "event_type_2",
    ] + snap_cols + ["snap"]
    return cal[keep].copy()


def load_prices(raw: Path) -> pd.DataFrame:
    pr = pd.read_csv(
        raw / PRICES_FILE,
        dtype={"store_id": "category", "item_id": "category", "wm_yr_wk": "int32"},
    )
    pr["sell_price"] = pr["sell_price"].astype("float32")
    return pr


def run(raw: Path, interim: Path, chunk_rows: int = 1500) -> Path:
    for f in (SALES_FILE, CALENDAR_FILE, PRICES_FILE):
        if not (raw / f).exists():
            _die(f"{raw / f} tidak ada. Unduh M5 lewat Kaggle dulu (spec Bagian 2). "
                 "Jangan pakai data sintetis.")

    interim.mkdir(parents=True, exist_ok=True)
    out = interim / "sales_long.parquet"
    if out.exists():
        out.unlink()

    cal = load_calendar(raw)
    pr = load_prices(raw)

    header = pd.read_csv(raw / SALES_FILE, nrows=0)
    d_cols = [c for c in header.columns if c.startswith("d_")]
    if not d_cols:
        _die(f"kolom d_* tidak ditemukan di {SALES_FILE}")

    id_dtypes = {
        "id": "string", "item_id": "category", "dept_id": "category",
        "cat_id": "category", "store_id": "category", "state_id": "category",
    }
    demand_dtypes = {c: "int16" for c in d_cols}

    writer: pq.ParquetWriter | None = None
    n_rows = 0
    day_order_verified: bool | None = None
    t0 = time.time()
    reader = pd.read_csv(
        raw / SALES_FILE, chunksize=chunk_rows,
        dtype={**id_dtypes, **demand_dtypes},
    )
    snap_cols = [c for c in cal.columns if c.startswith("snap_")]

    for i, chunk in enumerate(reader, start=1):
        long = chunk.melt(
            id_vars=["item_id", "store_id", "state_id"],
            value_vars=d_cols,
            var_name="d",
            value_name="demand",
        )
        # Urutan pandas.melt (diukur di pandas 3.0.6) = kolom-mayor: d_1 untuk SEMUA
        # baris chunk, lalu d_2, dst. Jadi day bisa dibuat lewat np.repeat tanpa
        # mem-parse 1,16 juta string per chunk — tapi asumsinya diverifikasi dulu
        # di chunk pertama terhadap kolom `d` hasil melt (jangan percaya ordering
        # implisit: kalau pandas berubah, script berhenti, bukan diam-diam salah).
        expected = np.repeat(np.arange(1, len(d_cols) + 1, dtype="int32"), len(chunk))
        if day_order_verified is None:
            actual = long["d"].str.replace("d_", "", regex=False).astype("int32").to_numpy()
            if not np.array_equal(expected, actual):
                _die("urutan hasil melt tidak kolom-mayor seperti asumsi — "
                     f"pandas {pd.__version__} mengubah perilaku; jangan lanjut.")
            day_order_verified = True
        long["day"] = expected
        long = long.drop(columns=["d"])

        long = long.merge(cal, on="day", how="left", validate="many_to_one")
        # sell_price bergantung pada (store_id, item_id, wm_yr_wk).
        # JANGAN merge ke frame pr penuh (6.841.121 baris): hash table pandas-nya
        # menahan puncak memori ~1,3 GB/chunk di host 2 GB ini (diukur 2026-09-29,
        # tidak berubah walau chunk 100 vs 600 SKU). Filter ke SKU chunk dulu
        # (~70 baris/SKU) -> puncak turun drastis, hasil merge identik.
        pr_chunk = pr[pr["item_id"].isin(chunk["item_id"].unique())]
        long = long.merge(
            pr_chunk, on=["store_id", "item_id", "wm_yr_wk"], how="left",
            validate="many_to_one",
        )
        # snap_* dari calendar umum; simpan hanya flag milik state SKU tersebut
        # (perbandingan kategorikal langsung — astype("string") per chunk
        # mengalokasi ~1 juta objek string dan ikut menaikkan puncak memori)
        for c in snap_cols:
            st = c.replace("snap_", "")
            long["snap"] = long["snap"].where(
                long["state_id"] != st, long[c].fillna(0).astype("int8")
            )
        long = long.drop(columns=snap_cols)

        long["date"] = pd.to_datetime(long["date"]).dt.date
        long["wday"] = long["wday"].astype("int8")
        long["month"] = long["month"].astype("int8")
        long["wm_yr_wk"] = long["wm_yr_wk"].astype("int32")
        long["snap"] = long["snap"].fillna(0).astype("int8")
        long["event_name_1"] = long["event_name_1"].astype("string")
        long["event_name_2"] = long["event_name_2"].astype("string")
        long["event_type_1"] = long["event_type_1"].astype("string")
        long["event_type_2"] = long["event_type_2"].astype("string")

        table = pa.Table.from_pandas(long, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out, table.schema, compression="snappy")
        writer.write_table(table)
        n_rows += len(long)
        print(f"  chunk {i:3d}: +{len(long):,} baris (total {n_rows:,}) "
              f"[{time.time() - t0:,.0f}s]")

    if writer is not None:
        writer.close()

    print(f"OK -> {out}  ({n_rows:,} baris, {time.time() - t0:,.0f}s)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="M5 wide -> long (chunked melt).")
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--interim", type=Path, default=Path("data/interim"))
    ap.add_argument("--chunk-rows", type=int, default=1500)
    ap.add_argument("--config", type=Path, default=Path("config.yaml"))
    a = ap.parse_args()

    t0 = time.time()
    out = run(a.raw, a.interim, a.chunk_rows)

    # Spec Bagian 6: results/run_log.txt wajib memuat versi library, seed, durasi.
    try:
        import yaml

        cfg = yaml.safe_load(open(a.config))
        results = Path(cfg["paths"]["results"])
        seed = int(cfg["seed"])
    except Exception as exc:  # noqa: BLE001
        print(f"peringatan: config tidak terbaca ({exc}); log ditulis ke ./results", file=sys.stderr)
        results, seed = Path("results"), 42

    import runlog

    rows = pq.ParquetFile(out).metadata.num_rows
    runlog.write(
        results / "run_log.txt",
        "M1 data_prep (spec 5.1 wide->long)",
        seed,
        t0,
        extra=(
            f"raw_dir     : {a.raw}\n"
            f"interim     : {out}\n"
            f"rows        : {rows:,}\n"
            f"chunk_rows  : {a.chunk_rows}"
        ),
    )
    print(f"run_log     : {results / 'run_log.txt'}")


if __name__ == "__main__":
    main()
