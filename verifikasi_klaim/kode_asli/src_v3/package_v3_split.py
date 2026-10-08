"""package_v3_split.py — pecah paket v3 menjadi beberapa zip < 45 MB (batas dokumen Telegram).

Kenapa: paket tunggal hasil `package_v3.py` berukuran ~98 MB (tabel biaya per SKU x tau + artefak
model 19 MB), sementara Bot API Telegram menolak dokumen > 50 MB. Alih-alih memaksa, paket dipecah
menurut isi supaya setiap kiriman kecil, jelas, dan tetap punya SHA-256 sendiri.

Keluaran (di --out-dir): v3_1_inti.zip, v3_2_tabel_utama.zip, v3_3_tabel_lain.zip,
v3_4_prediksi_rasio.zip, v3_5_prediksi_ml.zip, v3_6_artefak_model.zip + daftar SHA ke stdout.

Pakai:  .venv/bin/python src_v3/package_v3_split.py --out-dir /home/ubuntu
"""
from __future__ import annotations

import argparse
import hashlib
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def buat(zp: Path, daftar: list[tuple[Path, str]], readme: str) -> None:
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        untuk_readme = [f"{sha256(p)}  {nama}  {p.stat().st_size}" for p, nama in daftar]
        z.writestr("ISI_DAN_SHA256.txt", readme + "\n\n" + "\n".join(untuk_readme) + "\n")
        for p, nama in daftar:
            z.write(p, nama)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path.home())
    a = ap.parse_args()
    res = ROOT / "results_v3"
    T, F, I, A = res / "tables", res / "figures", res / "interim", res / "artifacts"
    a.out_dir.mkdir(parents=True, exist_ok=True)

    def berkas(pola_glob: str, folder: Path, prefix: str) -> list[tuple[Path, str]]:
        return [(p, f"{prefix}{p.name}") for p in sorted(folder.glob(pola_glob)) if p.is_file()]

    kode = []
    for pola, prefix in (("src_v3/*.py", "src_v3/"), ("tests/*.py", "tests/"), ("kaggle/build_kernel.py", "kaggle/")):
        kode += [(p, prefix + p.name) for p in sorted(ROOT.glob(pola))]
    for d in ("kaggle/v3", "kaggle/v3ml", "kaggle/v3h"):
        kode += [(p, f"{d}/{p.name}") for p in sorted((ROOT / d).glob("*"))
                 if p.is_file() and p.suffix in {".py", ".json", ".txt", ".yaml"}]
    kode += [(ROOT / "config_v3.yaml", "config_v3.yaml"), (ROOT / "requirements.txt", "requirements.txt")] \
        if (ROOT / "requirements.txt").exists() else [(ROOT / "config_v3.yaml", "config_v3.yaml")]
    kode += berkas("*.md", res, "results_v3/") + berkas("*.txt", res, "results_v3/")
    kode += berkas("run_registry.json", res, "results_v3/") + berkas("forecast_manifest_*.csv", res, "results_v3/")
    kode += berkas("stats_*.csv", T, "results_v3/tables/") + berkas("metrics_*.csv*", T, "results_v3/tables/")
    kode += berkas("sample_meta_v3.csv", T, "results_v3/tables/") + berkas("sku_sample_v3.csv", T, "results_v3/tables/")
    kode += berkas("lumpy_v3.csv", T, "results_v3/tables/") + berkas("ss_diagnostik_*.csv", T, "results_v3/tables/")
    kode += berkas("*.png", F, "results_v3/figures/")
    kode += berkas("importance_*.csv", A, "results_v3/artifacts/") + berkas("identity_mapping_v3.csv", A, "results_v3/artifacts/")

    tabel_utama = (berkas("costs_o4.csv.gz", T, "results_v3/tables/")
                   + berkas("policy_sweep_o4.csv.gz", T, "results_v3/tables/")
                   + berkas("service_equalized_o4.csv.gz", T, "results_v3/tables/")
                   + berkas("sensitivitas_v3.csv.gz", T, "results_v3/tables/"))
    tabel_lain = []
    for o in (1, 2, 3):
        tabel_lain += (berkas(f"costs_o{o}.csv.gz", T, "results_v3/tables/")
                       + berkas(f"policy_sweep_o{o}.csv.gz", T, "results_v3/tables/")
                       + berkas(f"service_equalized_o{o}.csv.gz", T, "results_v3/tables/")
                       + berkas(f"ss_diagnostik_o{o}.csv", T, "results_v3/tables/"))
    def grup_pred(prefix: str) -> list[tuple[Path, str]]:
        out = []
        for p in sorted(I.glob("pred_*.csv.gz")):
            nama = p.name
            kecil = ("rf_global_direct" in nama) or ("lightgbm" in nama) or ("rf_recursive" in nama)
            if prefix == "rasio" and kecil:
                continue
            if prefix == "ml" and not kecil:
                continue
            out.append((p, f"results_v3/interim/{nama}"))
        return out

    def per_origin(daftar: list[tuple[Path, str]], origins: tuple[str, ...]) -> list[tuple[Path, str]]:
        return [(p, n) for p, n in daftar if any(f"_o{o}_" in Path(n).name for o in origins)]

    rasio, ml = grup_pred("rasio"), grup_pred("ml")
    pred_rasio_12, pred_rasio_34 = per_origin(rasio, ("1", "2")), per_origin(rasio, ("3", "4"))
    pred_ml_12, pred_ml_34 = per_origin(ml, ("1", "2")), per_origin(ml, ("3", "4"))
    err_val = berkas("err_val_*.csv.gz", I, "results_v3/interim/")
    err_12 = [(p, n) for p, n in err_val if any(f"_o{o}." in Path(n).name or f"_o{o}.csv" in Path(n).name for o in ("1", "2"))]
    err_34 = [(p, n) for p, n in err_val if (p, n) not in err_12]
    artefak = berkas("model_*.joblib", A, "results_v3/artifacts/")

    paket = [
        ("v3_1_inti.zip", kode, "Kode, konfigurasi, laporan, fakta, gambar, statistik ringkas, artefak ringan."),
        ("v3_2_tabel_utama.zip", tabel_utama, "Tabel biaya/kebijakan origin 4 (split resmi) + sensitivitas."),
        ("v3_3_tabel_lain.zip", tabel_lain, "Tabel biaya/kebijakan origin 1-3 (robustness)."),
        ("v3_4a_prediksi_rasio_o12.zip", pred_rasio_12, "Prediksi mentah model rasio, origin 1-2."),
        ("v3_4b_prediksi_rasio_o34.zip", pred_rasio_34, "Prediksi mentah model rasio, origin 3-4 (+ varian H)."),
        ("v3_5a_prediksi_ml_o12.zip", pred_ml_12 + err_12, "Prediksi mentah model ML, origin 1-2."),
        ("v3_5b_prediksi_ml_o34.zip", pred_ml_34 + err_34, "Prediksi mentah model ML, origin 3-4."),
        ("v3_6_artefak_model.zip", artefak, "Artefak model RF (.joblib) untuk reproduksi & inspeksi."),
    ]
    print(f"# paket v3 dipecah — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    total = 0
    for nama, daftar, ket in paket:
        zp = a.out_dir / nama
        if not daftar:
            print(f"{nama:24s} (kosong, dilewati)")
            continue
        buat(zp, daftar, f"{ket}\nDibangun {time.strftime('%Y-%m-%d %H:%M:%S')} dari {ROOT}")
        mb = zp.stat().st_size / 1e6
        total += mb
        tanda = "OK " if mb < 45 else "BESAR"
        print(f"{tanda} {nama:24s} {len(daftar):4d} berkas {mb:8.2f} MB  sha256 {sha256(zp)[:16]}...")
    print(f"total {total:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
