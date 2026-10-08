"""M6 (v2) — Kumpulkan seluruh isi results/, pastikan tidak ada yang tercecer.

Spec v2 Bagian 6 & 9: M6 hanya menyatukan/memeriksa artefak, TIDAK menulis narasi apa pun.
Skrip ini:
  1. memeriksa keberadaan + jumlah baris + kolom tiap artefak wajib,
  2. menghitung SHA-256 tiap berkas (supaya angka yang dipakai di chat bisa diverifikasi),
  3. menulis results/manifest.txt (daftar berkas) dan results/raw_sha256.txt kalau ada data mentah,
  4. berhenti dengan exit code 1 kalau ada artefak wajib yang hilang — bukan diam-diam lolos.

Pemakaian: .venv/bin/python src/collect_results.py
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

WAJIB_TABEL = [
    "sku_sample.csv",
    "mae_comparison.csv",
    "mae_per_sku_validation.csv",
    "rf_best_params.csv",
    "rf_feature_importance.csv",
    "rf_tuning.csv",
    "rf_identity_mapping.csv",
    "rf_test_mae_per_sku.csv",
    "rf_val_residuals.csv",
    "rf_test_forecast.csv",
    "croston_summary.csv",
    "croston_val_residuals.csv",
    "croston_test_forecast.csv",
    "cost_summary.csv",
    "wilcoxon_results.csv",
]
WAJIB_GAMBAR = [
    "cost_comparison_bar.png",
    "sensitivity_lineplot.png",
    "rf_feature_importance.png",
]
# kolom yang wajib ada supaya penulisan di chat tidak salah kutip (spec v2 5.8 & 5.9)
WAJIB_KOLOM = {
    "cost_summary.csv": [
        "selisih_median_berpasangan_per_sku",
        "selisih_median_agregat",
        "cost_total",
        "fill_rate",
        "ratio_cs_ch",
    ],
    "wilcoxon_results.csv": [
        "rank_biserial_r",
        "rank_biserial_abs",
        "ukuran_efek",
        "p_value",
        "statistic",
    ],
    "rf_feature_importance.csv": ["rank", "feature", "importance", "is_identity_feature"],
    "rf_best_params.csv": ["model_type", "n_estimators", "max_depth", "min_samples_leaf", "mae_val_pooled", "mae_test_pooled"],
}


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Periksa & rekap artefak results/ (M6).")
    ap.add_argument("--config", type=Path, default=Path("config.yaml"))
    ap.add_argument("--results", type=Path, default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text())
    res = Path(a.results or cfg["paths"]["results"])
    tabs = res / "tables"
    figs = res / "figures"

    lines: list[str] = []
    problems: list[str] = []

    def p(line: str = "") -> None:
        print(line)
        lines.append(line)

    p(f"MANIFEST results/ — dibuat {datetime.now(timezone.utc).isoformat(timespec='seconds')} (UTC)")
    p("")

    p("== tabel ==")
    for name in WAJIB_TABEL:
        f = tabs / name
        if not f.exists():
            problems.append(f"hilang: tables/{name}")
            p(f"  [HILANG] {name}")
            continue
        if name.endswith(".csv"):
            df = pd.read_csv(f)
            cols_ok = WAJIB_KOLOM.get(name)
            missing = [c for c in (cols_ok or []) if c not in df.columns]
            if missing:
                problems.append(f"kolom hilang di {name}: {missing}")
            p(f"  {name:34s} {len(df):>9,} baris | {df.shape[1]:>2} kolom | {f.stat().st_size / 1e3:>8.1f} KB"
              + (f" | KOLOM HILANG {missing}" if missing else ""))

    p("")
    p("== gambar ==")
    for name in WAJIB_GAMBAR:
        f = figs / name
        if not f.exists():
            problems.append(f"hilang: figures/{name}")
            p(f"  [HILANG] {name}")
        else:
            p(f"  {name:34s} {f.stat().st_size / 1e3:>8.1f} KB")

    contoh = sorted(figs.glob("contoh_sku_*.png"))
    p(f"  contoh SKU                        {len(contoh)} berkas")
    if len(contoh) < 2:
        problems.append(f"gambar contoh_sku_* hanya {len(contoh)} (spec minta 2-3)")

    p("")
    p("== berkas teks ==")
    for name in ("run_log.txt", "article_facts.txt"):
        f = res / name
        if not f.exists():
            (problems if name == "run_log.txt" else lines).append(f"catatan: {name} tidak ada")
            p(f"  [{'HILANG' if name == 'run_log.txt' else 'catatan'}] {name}")
        else:
            p(f"  {name:34s} {f.stat().st_size / 1e3:>8.1f} KB | {len(f.read_text().splitlines()):>4} baris")

    trace = tabs / "inventory_daily_trace.csv.gz"
    p(f"  {'inventory_daily_trace.csv.gz':34s} "
      + (f"{trace.stat().st_size / 1e3:>8.1f} KB (ter-gzip)" if trace.exists() else "[HILANG]"))
    if not trace.exists():
        problems.append("hilang: tables/inventory_daily_trace.csv.gz")

    p("")
    p("== SHA-256 berkas keluaran (untuk verifikasi angka di chat) ==")
    all_files = sorted(f for f in res.rglob("*") if f.is_file() and f.name not in {"manifest.txt"})
    for f in all_files:
        p(f"  {sha256(f)}  {f.relative_to(res)}")
    p("")
    p("catatan: setiap berkas di results/ ikut dihitung, termasuk article_facts.txt, run_log.txt,")
    p("         dan tables/all_sku_stats.csv — manifest ini dibuat SETELAH semua tahap selesai")

    (res / "manifest.txt").write_text("\n".join(lines) + "\n")
    print(f"\n[disimpan ke {res / 'manifest.txt'}]")

    if problems:
        print("\nMASALAH:")
        for x in problems:
            print(f"  - {x}")
        return 1
    print("\nsemua artefak wajib ada dan kolomnya lengkap")
    return 0


if __name__ == "__main__":
    sys.exit(main())
