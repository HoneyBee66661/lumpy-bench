"""sample_v3.py — Fase 3: klasifikasi lumpy + sampling 300 SKU (stratified) untuk v3.

Perbedaan pokok dari M2 v2 (yang dikritik audit):
  1. ADI/CV2 dihitung pada PERIODE AKTIF SKU, bukan 1.761 hari seragam. Periode aktif = rentang
     hari pertama sampai hari terakhir SKU itu punya harga jual di `sell_price` (kolom hasil merge
     sell_prices di parquet M1). Nol struktural sebelum peluncuran / setelah penghentian tidak lagi
     menggembungkan ADI.
  2. Sampling STRATIFIED (kuadran ADI x kuadran CV2 x kategori x state), bukan sampel acak sederhana,
     dengan catatan alokasi per stratum (docs: results_v3/tables/strata_v3.csv).

Keluaran:
  results_v3/tables/all_sku_stats_v3.csv   statistik per SKU (30.490 baris)
  results_v3/tables/sku_sample_v3.csv      n_sku terpilih + kolom stratum
  results_v3/tables/strata_v3.csv          jumlah SKU per stratum + jumlah terpilih
  results_v3/tables/lumpy_v3.csv           seluruh SKU yang lolos kriteria (bukan hanya sampel)

Jalankan:
  .venv/bin/python src_v3/sample_v3.py --config config_v3.yaml
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
KEY = ["item_id", "store_id"]


def _acc() -> dict:
    """Akumulator fixed-width per SKU (tidak menyimpan frame per batch)."""
    return {
        "n_hari_aktif": 0,       # hari dalam periode aktif
        "n_nonzero": 0,          # hari dengan demand > 0 (dalam periode aktif)
        "sum_pos": 0.0,
        "sumsq_pos": 0.0,
        "hari_pertama": 10**9,
        "hari_terakhir": -1,
        "hari_pertama_jual": 10**9,   # hari pertama demand > 0
        "hari_terakhir_jual": -1,
        "demand_total": 0.0,
    }


def stream_stats(parquet: Path, batch_rows: int = 500_000, verbose: bool = True) -> pd.DataFrame:
    """Hitung statistik ADI/CV2 per SKU pada periode aktifnya (streaming, memori terbatas)."""
    dataset = ds.dataset(parquet, format="parquet")
    cols = ["item_id", "store_id", "day", "demand", "sell_price"]
    n_batch = 0
    acc: dict[tuple, dict] = {}
    t0 = time.time()
    for batch in dataset.to_batches(columns=cols, batch_size=batch_rows):
        df = batch.to_pandas()
        # periode aktif: sell_price tidak kosong
        aktif = df[df["sell_price"].notna()]
        for (item, store), g in aktif.groupby(["item_id", "store_id"], sort=False, observed=True):
            k = (item, store)
            a = acc.setdefault(k, _acc())
            a["n_hari_aktif"] += int(len(g))
            a["hari_pertama"] = min(a["hari_pertama"], int(g["day"].min()))
            a["hari_terakhir"] = max(a["hari_terakhir"], int(g["day"].max()))
            a["demand_total"] += float(g["demand"].sum())
            pos = g.loc[g["demand"] > 0, "demand"]
            if len(pos):
                a["n_nonzero"] += int(len(pos))
                a["sum_pos"] += float(pos.sum())
                a["sumsq_pos"] += float((pos**2).sum())
                a["hari_pertama_jual"] = min(a["hari_pertama_jual"], int(g.loc[g["demand"] > 0, "day"].min()))
                a["hari_terakhir_jual"] = max(a["hari_terakhir_jual"], int(g.loc[g["demand"] > 0, "day"].max()))
        n_batch += 1
        if verbose and n_batch % 20 == 0:
            print(f"    batch {n_batch:4d} | {len(acc):,} SKU | {time.time() - t0:.0f}s", flush=True)

    rows = []
    for (item, store), a in acc.items():
        n_aktif = a["n_hari_aktif"]
        n_pos = a["n_nonzero"]
        if n_pos >= 2:
            mean_pos = a["sum_pos"] / n_pos
            var_pos = max((a["sumsq_pos"] - n_pos * mean_pos**2) / (n_pos - 1), 0.0)
            std_pos = float(np.sqrt(var_pos))
            cv2 = float((std_pos / mean_pos) ** 2) if mean_pos > 0 else np.nan
        else:
            mean_pos = a["sum_pos"] / n_pos if n_pos else np.nan
            std_pos = np.nan
            cv2 = np.nan
        adi = float(n_aktif / n_pos) if n_pos else np.inf
        rows.append(
            {
                "item_id": item,
                "store_id": store,
                "sku_id": f"{item}__{store}",
                "n_hari_aktif": int(n_aktif),
                "hari_pertama_aktif": int(a["hari_pertama"]),
                "hari_terakhir_aktif": int(a["hari_terakhir"]),
                "hari_pertama_jual": (None if a["hari_pertama_jual"] == 10**9 else int(a["hari_pertama_jual"])),
                "hari_terakhir_jual": (None if a["hari_terakhir_jual"] == -1 else int(a["hari_terakhir_jual"])),
                "n_nonzero": int(n_pos),
                "mean_pos": mean_pos,
                "std_pos": std_pos,
                "adi": adi,
                "cv2": cv2,
                "demand_total": float(a["demand_total"]),
            }
        )
    out = pd.DataFrame(rows)
    out["cat_id"] = out["item_id"].str.split("_").str[0]
    out["dept_id"] = out["item_id"].str.split("_").str[:2].str.join("_")
    out["state_id"] = out["store_id"].str.split("_").str[0]
    return out.sort_values(KEY).reset_index(drop=True)


def _kuadran(s: pd.Series) -> pd.Series:
    """Kuadran 1..4 berdasarkan kuantil (deterministik, label q1 = terendah)."""
    q = pd.qcut(s.rank(method="first"), 4, labels=[1, 2, 3, 4])
    return q.astype(int)


def sample_stratified(
    stats: pd.DataFrame,
    n_sku: int,
    seed: int,
    adi_min: float,
    cv2_min: float,
    min_nonzero: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Filter lumpy pada periode aktif + alokasi stratified proporsional dengan pembulatan sisa."""
    eligible = stats[
        (stats["n_nonzero"] >= min_nonzero)
        & (stats["adi"] > adi_min)
        & (stats["cv2"] > cv2_min)
    ].copy()
    eligible["adi_kuadran"] = _kuadran(eligible["adi"])
    eligible["cv2_kuadran"] = _kuadran(eligible["cv2"])
    eligible["stratum"] = (
        eligible["adi_kuadran"].astype(str) + "-" + eligible["cv2_kuadran"].astype(str)
        + "-" + eligible["cat_id"].astype(str) + "-" + eligible["state_id"].astype(str)
    )

    strata = eligible.groupby("stratum", observed=True).size().rename("n_tersedia").to_frame()
    rng = np.random.default_rng(seed)
    target = min(n_sku, len(eligible))

    # alokasi proporsional + sisa terbesar
    prop = strata["n_tersedia"] / strata["n_tersedia"].sum() * target
    alok = np.floor(prop).astype(int)
    sisa = target - int(alok.sum())
    if sisa > 0:
        urutan = (prop - alok).sort_values(ascending=False).index[:sisa]
        alok.loc[urutan] += 1
    alok = np.minimum(alok, strata["n_tersedia"])          # tidak boleh melebihi yang tersedia
    strata["n_dipilih"] = alok.astype(int)

    picks = []
    for stratum, n_take in strata["n_dipilih"].items():
        if n_take <= 0:
            continue
        sub = eligible[eligible["stratum"] == stratum]
        idx = rng.choice(sub.index.to_numpy(), size=int(n_take), replace=False)
        picks.append(eligible.loc[idx])
    sample = pd.concat(picks, ignore_index=True) if picks else eligible.head(0)
    sample = sample.sort_values(KEY).reset_index(drop=True)
    return sample, strata.reset_index(), eligible


def main() -> int:
    ap = argparse.ArgumentParser(description="Klasifikasi lumpy (periode aktif) + sampling stratified v3.")
    ap.add_argument("--config", type=Path, default=ROOT / "config_v3.yaml")
    ap.add_argument("--parquet", type=Path, default=None)
    ap.add_argument("--batch-rows", type=int, default=500_000)
    a = ap.parse_args()

    cfg = yaml.safe_load(a.config.read_text())
    parquet = a.parquet or (Path(cfg["paths"]["data_interim"]) / "sales_long.parquet")
    res = ROOT / cfg["paths"]["results"]
    (res / "tables").mkdir(parents=True, exist_ok=True)

    print(f"streaming statistik per SKU (periode aktif) dari {parquet}")
    stats = stream_stats(parquet, a.batch_rows)
    stats.to_csv(res / "tables" / "all_sku_stats_v3.csv", index=False)
    print(f"  {len(stats):,} SKU | median hari aktif {stats.n_hari_aktif.median():.0f} "
          f"| median ADI {stats.adi.median():.3f} | median CV2 {stats.cv2.median():.3f}")

    s = cfg["sampling_v3"]
    sample, strata, eligible = sample_stratified(
        stats,
        n_sku=int(s["n_sku"]),
        seed=int(cfg["seed"]),
        adi_min=float(cfg["lumpy_threshold"]["adi_min"]),
        cv2_min=float(cfg["lumpy_threshold"]["cv2_min"]),
        min_nonzero=int(s["aktif_min_nonzero"]),
    )
    eligible.to_csv(res / "tables" / "lumpy_v3.csv", index=False)
    strata.to_csv(res / "tables" / "strata_v3.csv", index=False)
    sample.to_csv(res / "tables" / "sku_sample_v3.csv", index=False)

    print(f"SKU lolos kriteria (periode aktif) : {len(eligible):,} dari {len(stats):,}")
    print(f"SKU ter-sampling                   : {len(sample)} (target {s['n_sku']}, strata {len(strata)})")
    print(f"  ADI sampel    : min {sample.adi.min():.2f} | median {sample.adi.median():.2f} | maks {sample.adi.max():.2f}")
    print(f"  CV2 sampel    : min {sample.cv2.min():.2f} | median {sample.cv2.median():.2f} | maks {sample.cv2.max():.2f}")
    print(f"  komposisi     : kategori {sample.cat_id.value_counts().to_dict()} | state {sample.state_id.value_counts().to_dict()}")
    print(f"  hari aktif    : median {sample.n_hari_aktif.median():.0f} | n_nonzero median {sample.n_nonzero.median():.0f}")
    print("\n5 baris contoh:")
    print(sample[["sku_id", "n_hari_aktif", "hari_pertama_aktif", "hari_terakhir_aktif", "n_nonzero", "adi", "cv2"]].head().to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
