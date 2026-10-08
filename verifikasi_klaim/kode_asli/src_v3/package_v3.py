"""package_v3.py — susun paket akhir v3 (dua zip: paket utama + prediksi mentah) + README + SHA-256.

Paket 1 `results_v3_paket.zip`   : kode (src_v3/, tests/), konfigurasi, tabel, gambar, laporan, fakta,
                                   run log, manifest, kernel Kaggle (sumber + metadata) — bahan tulis
                                   Tugas 2 (Metode) & Tugas 3 (Hasil/Pembahasan/Kesimpulan).
Paket 2 `results_v3_forecasts.zip`: berkas prediksi mentah hasil kernel Kaggle (pred_*.csv.gz) +
                                   forecast_log.txt + error validasi, supaya angka bisa ditelusuri
                                   sampai prediksi per SKU per hari.

Tidak ada berkas besar yang wajib (data/raw, parquet) yang ikut: paket menyebutkan asal-usulnya.

Pakai:  .venv/bin/python src_v3/package_v3.py [--out-dir /home/ubuntu]
"""
from __future__ import annotations

import argparse
import hashlib
import time
import zipfile
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def kumpulkan(res: Path) -> tuple[list[tuple[Path, str]], list[tuple[Path, str]]]:
    """(berkas paket utama, berkas paket prediksi) sebagai (path, nama-di-zip)."""
    utama: list[tuple[Path, str]] = []
    for pola, prefix in (
        ("src_v3/*.py", "src_v3/"),
        ("tests/*.py", "tests/"),
        ("config_v3.yaml", ""),
        ("kaggle/build_kernel.py", "kaggle/"),
    ):
        for p in sorted(ROOT.glob(pola)):
            utama.append((p, prefix + p.name))
    for d in ("kaggle/v3", "kaggle/v3ml", "kaggle/v3h"):
        for p in sorted((ROOT / d).glob("*")):
            if p.is_file() and p.suffix in {".py", ".json", ".txt", ".yaml"}:
                utama.append((p, f"{d}/{p.name}"))
    for p in sorted(res.glob("*")):
        if p.is_file():
            utama.append((p, f"results_v3/{p.name}"))
    for sub in ("tables", "figures", "artifacts", "evidence"):
        d = res / sub
        if d.exists():
            for p in sorted(d.glob("*")):
                if p.is_file():
                    utama.append((p, f"results_v3/{sub}/{p.name}"))
    pred: list[tuple[Path, str]] = []
    inter = res / "interim"
    if inter.exists():
        for p in sorted(inter.glob("*")):
            if p.is_file():
                pred.append((p, f"results_v3/interim/{p.name}"))
    return utama, pred


def baca_ringkas(res: Path, cfg: dict) -> list[str]:
    """Ringkasan angka siap baca untuk README paket."""
    L: list[str] = []
    pw = res / "tables" / "stats_pairwise_v3.csv"
    if pw.exists():
        d = pd.read_csv(pw)
        sel = d[(d.origin == 4) & (d.cs == 5.0) & (d.fill_target == 0.95)]
        if len(sel):
            L.append("Biaya service-equalized (origin 4, Cs:Ch 5:1, fill 95%), median antar SKU:")
            for _, r in sel.sort_values("median_diff").iterrows():
                L.append(f"  RF vs {r.model_b:24s}: selisih {r.median_diff:10.2f} ({r.pct_diff_median:+6.2f}%), "
                         f"HL {r.hodges_lehmann:10.2f}, p {r.p_one_sided_a_lebih_murah:.4f}, "
                         f"p Holm {r.p_one_sided_holm:.4f}, r {r.r_rank_biserial:+.3f}, "
                         f"menang {int(r.menang_a)}/{int(r.n_sku)}")
    mr = res / "tables" / "metrics_rekap_v3.csv"
    if mr.exists():
        m = pd.read_csv(mr)
        m = m[(m.split == "test") & (m.origin == 4)].sort_values("mae")
        L.append("Akurasi (median antar SKU, origin 4, uji), MAE pada besaran periode proteksi:")
        for _, r in m.iterrows():
            L.append(f"  {r.model:26s} MAE {r.mae:8.4f} | MASE {r.mase:6.3f} | RMSSE {r.rmsse:6.3f}")
    return L


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "config_v3.yaml")
    ap.add_argument("--out-dir", type=Path, default=Path.home())
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text())
    res = ROOT / cfg["paths"]["results"]
    if not (res / "article_facts_v3.txt").exists():
        print(f"STOP: {(res / 'article_facts_v3.txt')} belum ada — jalankan stage facts dulu.")
        return 2

    utama, pred = kumpulkan(res)
    ringkas = baca_ringkas(res, cfg)

    readme = [
        "# Paket v3 — RF vs Croston-SBA untuk lumpy demand (asymmetric cost)",
        "",
        f"dibangun {time.strftime('%Y-%m-%d %H:%M:%S')} dari repo `{ROOT}`",
        "",
        "## Isi paket ini (results_v3_paket.zip)",
        "",
        "- `src_v3/` — pipeline v3: simulator net (`simulator_v3.py`), fitur direct multi-horizon",
        "  (`features_v3.py`), sampling (`sample_v3.py`), forecast (`forecast_v3.py`), kebijakan &",
        "  service-equalized (`policy_v3.py`), statistik (`stats_v3.py`), orkestrator (`run_all_v3.py`),",
        "  gambar, run log, laporan, paket.",
        "- `tests/` — 3 tes (simulator+invarian, fitur, statistik) + 2 verifikasi independen",
        "  (`verify_v3_forecasts.py`, `verify_v3_policy.py`).",
        "- `results_v3/tables/` — semua tabel angka (CSV/CSV.GZ). `figures/` gambar. `artifacts/`",
        "  feature importance + model RF (.joblib). `evidence/` skrip bukti untuk audit.",
        "- `results_v3/article_facts_v3.txt` — semua angka kunci siap dikutip, dengan konvensi arah.",
        "- `results_v3/LAPORAN_PERUBAHAN_v3.md` — perubahan dari v2, claim ledger, vonis H1/H2, keterbatasan.",
        "- `results_v3/run_log_v3.txt` + `manifest_v3.txt` — provenance dan SHA-256 tiap berkas.",
        "- `config_v3.yaml` — semua parameter; `kaggle/v3*/` — sumber & metadata kernel Kaggle.",
        "",
        "## Paket kedua (results_v3_forecasts.zip)",
        "",
        "Berkas prediksi mentah per model/origin/seed (`pred_*.csv.gz`, kolom sku_id, day, split, pred_sum)",
        "beserta error validasi (`err_val_*.csv.gz`) dan log waktu produksinya. Ini yang membuat setiap",
        "angka di tabel bisa ditelusuri sampai prediksi per SKU per hari.",
        "",
        "## Cara memverifikasi ulang (urut)",
        "",
        "```",
        "uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt",
        ".venv/bin/python tests/test_simulator_v3.py      # invarian + regresi legacy v2 (selisih < 1e-8)",
        ".venv/bin/python tests/test_features_v3.py       # rumus fitur + uji mutasi (tanpa kebocoran)",
        ".venv/bin/python tests/test_stats_v3.py          # rumus statistik + sifat Holm/CI",
        ".venv/bin/python tests/verify_v3_forecasts.py    # hitung ulang prediksi dari parquet + artefak",
        ".venv/bin/python tests/verify_v3_policy.py       # simulator ditulis ulang dari nol vs tabel biaya",
        "```",
        "",
        "Data mentah tidak ikut (ukuran besar): M5 Forecasting Accuracy (Kaggle) — 5 berkas dengan SHA-256",
        "di `results/raw_sha256.txt` repo; parquet turunan dipakai lewat dataset privat",
        "`honeybee66661/m5-sales-long`.",
        "",
        "## Angka ringkas (dibaca dari tabel, bukan diketik)",
        "",
    ] + [*ringkas] + [
        "",
        "## Catatan konvensi (jangan dibalik saat menulis)",
        "",
        "- `r_rank_biserial` dihitung dari `d = biaya_RF − biaya_pembanding`; **r POSITIF = RF LEBIH MURAH**.",
        "- Uji utama **satu sisi** (arah pra-registrasi: RF lebih murah), koreksi **Holm** dalam keluarga",
        "  uji per origin; dua sisi dilaporkan sebagai robustness.",
        "- Biaya konvensi utama `per_unit_once`; `per_unit_day` ada di tabel sensitivitas.",
        "- Hari yang dibebankan = jendela proteksi lengkap di dalam data; warm-up 28 hari tidak dibebankan.",
    ]
    readme_path = res / "README_paket_v3.md"
    readme_path.write_text("\n".join(readme) + "\n")
    utama.append((readme_path, "README_paket_v3.md"))

    a.out_dir.mkdir(parents=True, exist_ok=True)
    keluaran = []
    for nama_zip, daftar in (("results_v3_paket.zip", utama), ("results_v3_forecasts.zip", pred)):
        zp = a.out_dir / nama_zip
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            for p, nama in daftar:
                z.write(p, f"v3/{nama}")
        kb = zp.stat().st_size / 1e6
        print(f"{nama_zip:28s} {len(daftar):4d} berkas {kb:8.2f} MB  sha256 {sha256(zp)[:16]}...")
        keluaran.append((zp, len(daftar), kb))
    # daftar isi + SHA untuk lampiran
    isi = res / "paket_v3_isi.txt"
    with isi.open("w") as fh:
        fh.write(f"# isi paket v3 — {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        for nama_zip, daftar in (("results_v3_paket.zip", utama), ("results_v3_forecasts.zip", pred)):
            fh.write(f"\n## {nama_zip} ({len(daftar)} berkas)\n")
            for p, nama in daftar:
                fh.write(f"{sha256(p)}  {nama}  {p.stat().st_size}\n")
    print(f"daftar isi -> {isi.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
