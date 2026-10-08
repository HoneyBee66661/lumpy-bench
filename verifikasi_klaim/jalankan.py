"""jalankan.py — verifikasi satu-klik angka klaim artikel pada data M5 milik pengguna.

Alur (mode penuh, setelan artikel):
    1. periksa SHA-256 berkas M5 yang diberikan
    2. data_prep  : berkas mentah M5 -> sales_long.parquet
    3. sample_v3  : klasifikasi ADI/CV2 + sampel berstrata 300 SKU (seed 42)
    4. forecasts  : walk-forward 4 titik evaluasi, model rf_global_direct + croston_sba
    5. policy/stats: safety stock kuantil empiris, penyetaraan tingkat layanan, uji Wilcoxon/Page
    6. tabel 2 metode + tabel terbuka (rev8)
    7. bandingkan keluaran dengan klaim/klaim_artikel.json -> laporan_verifikasi.md/.json

Contoh:
    python jalankan.py --m5 ~/m5 --kerja kerja
    python jalankan.py --m5 ~/m5 --mode cepat --n-sku 30 --origins 4
    python jalankan.py --hanya-cek --hasil-dir /path/ke/repo-yang-sudah-dijalankan
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

import klaim_cek

BUNDLE = Path(__file__).resolve().parent
KODE_ASLI = BUNDLE / "kode_asli"
KLAIM = BUNDLE / "klaim" / "klaim_artikel.json"
BERKAS_M5 = ("sales_train_evaluation.csv", "sell_prices.csv", "calendar.csv")
# Tabel H1/H2 (rev8) selalu memerlukan keempat titik evaluasi: mode cepat memperingan jumlah SKU,
# bukan jumlah titik, supaya seluruh rantai tabel tetap terbentuk.
SETELAN_ARTIKEL = {"penuh": {"n_sku": 300, "origins": [1, 2, 3, 4]},
                   "cepat": {"n_sku": 30, "origins": [1, 2, 3, 4]}}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blok in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(blok)
    return h.hexdigest()


def periksa_m5(m5: Path, abaikan_sha: bool) -> tuple[bool, list[str]]:
    pesan = []
    kurang = [b for b in BERKAS_M5 if not (m5 / b).exists()]
    if kurang:
        raise SystemExit(f"berkas M5 tidak ditemukan di {m5}: {kurang}")
    rujukan = {}
    f_sha = KODE_ASLI / "raw_sha" / "raw_sha256.txt"
    if f_sha.exists():
        for baris in f_sha.read_text().splitlines():
            bagian = baris.split()
            if len(bagian) == 2:
                rujukan[bagian[1]] = bagian[0]
    sha_ok = True
    for nama in BERKAS_M5:
        harap = rujukan.get(nama)
        if harap is None:
            pesan.append(f"{nama}: tidak ada nilai rujukan SHA-256")
            continue
        terukur = sha256(m5 / nama)
        if terukur == harap:
            pesan.append(f"{nama}: SHA-256 cocok")
        else:
            sha_ok = False
            pesan.append(f"{nama}: SHA-256 BERBEDA (diharapkan {harap[:16]}..., terukur {terukur[:16]}...)")
    if not sha_ok and not abaikan_sha:
        raise SystemExit("SHA-256 tidak cocok dengan berkas M5 resmi. Ulangi dengan --abaikan-sha "
                         "hanya bila Anda sadar datanya berbeda.")
    return sha_ok, pesan


def siapkan_kerja(kerja: Path, segarkan: bool) -> None:
    if kerja.exists() and segarkan:
        shutil.rmtree(kerja)
    kerja.mkdir(parents=True, exist_ok=True)
    for nama in ("src", "src_v3"):
        tujuan = kerja / nama
        if not tujuan.exists():
            shutil.copytree(KODE_ASLI / nama, tujuan,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for nama in ("config_v3.yaml", "config.yaml", "requirements.lock.txt"):
        tujuan = kerja / nama
        if not tujuan.exists():
            shutil.copy2(KODE_ASLI / nama, tujuan)


def patche_config(kerja: Path, m5: Path, n_sku: int, origins: list[int]) -> None:
    """Setel path data + jumlah SKU/titik evaluasi pada config kerja (komentar dipertahankan)."""
    p = kerja / "config_v3.yaml"
    teks = p.read_text(encoding="utf-8")
    ganti = {
        "data_raw": str(m5.resolve()),
        "data_interim": str((kerja / "data" / "interim").resolve()),
        "results": str((kerja / "results_v3").resolve()),
        "interim_v3": str((kerja / "results_v3" / "interim").resolve()),
    }
    for kunci, nilai in ganti.items():
        teks, n = re.subn(rf"^(\s*){kunci}:.*$", rf"\g<1>{kunci}: {nilai}", teks, flags=re.M)
        if n == 0:
            raise SystemExit(f"kunci '{kunci}' tidak ditemukan di config_v3.yaml")
    teks, n = re.subn(r"^(\s*)n_sku:.*$", rf"\g<1>n_sku: {n_sku}", teks, count=1, flags=re.M)
    if n == 0:
        raise SystemExit("kunci 'n_sku' tidak ditemukan di config_v3.yaml")
    # n_origin harus cukup besar agar semua titik yang diminta ada; flag --origins yang memilih
    teks, n = re.subn(r"^(\s*)n_origin:.*$", rf"\g<1>n_origin: {max(origins)}", teks, count=1, flags=re.M)
    if n == 0:
        raise SystemExit("kunci 'n_origin' tidak ditemukan di config_v3.yaml")
    p.write_text(teks, encoding="utf-8")
    for d in ("data/interim", "results_v3/tables", "results_v3/interim", "results_v3/2metode/tables"):
        (kerja / d).mkdir(parents=True, exist_ok=True)


def jalankan_perintah(perintah: list[str], kerja: Path, sudah_ada: Path | None, paksa: bool) -> None:
    if sudah_ada is not None and sudah_ada.exists() and not paksa:
        log(f"lewat  : {sudah_ada} sudah ada (pakai --paksa untuk mengulang)")
        return
    log(f"jalan  : {' '.join(perintah)}")
    hasil = subprocess.run(perintah, cwd=kerja)
    if hasil.returncode != 0:
        raise SystemExit(f"perintah gagal (exit {hasil.returncode}): {' '.join(perintah)}")


def tahapan(kerja: Path, m5: Path, n_sku: int, origins: list[int], paksa: bool,
            parquet_siap: Path | None = None) -> None:
    py = sys.executable
    parquet = kerja / "data" / "interim" / "sales_long.parquet"
    # 1) berkas mentah -> parquet panjang (atau pakai yang sudah ada)
    if parquet_siap is not None:
        if not parquet.exists():
            shutil.copy2(parquet_siap, parquet)
        log(f"prep   : memakai parquet yang diberikan ({parquet_siap})")
    else:
        jalankan_perintah([py, "src/data_prep.py", "--raw", str(m5.resolve()),
                           "--interim", str((kerja / "data" / "interim").resolve()),
                           "--config", "config_v3.yaml"], kerja, parquet, paksa)
    # 2) klasifikasi + sampel berstrata
    jalankan_perintah([py, "src_v3/sample_v3.py", "--config", "config_v3.yaml"],
                      kerja, kerja / "results_v3" / "tables" / "sku_sample_v3.csv", paksa)
    # 3) walk-forward per titik evaluasi (bagian terberat)
    minta = ",".join(str(o) for o in origins)
    pred_terakhir = kerja / "results_v3" / "interim" / f"pred_croston_sba_o{origins[-1]}_s42.csv.gz"
    jalankan_perintah([py, "src_v3/run_forecasts_v3.py", "--config", "config_v3.yaml",
                       "--n-sku", str(n_sku), "--origins", minta,
                       "--models", "rf_global_direct,croston_sba", "--seeds", "42", "--resume"],
                      kerja, pred_terakhir, paksa)
    # pengaman: pastikan peramalan benar-benar menghasilkan berkas sebelum tahap berikutnya
    ada = sorted((kerja / "results_v3" / "interim").glob("pred_*.csv.gz"))
    if not ada:
        raise SystemExit("tahap peramalan tidak menghasilkan berkas pred_*.csv.gz — periksa "
                         "apakah titik evaluasi yang diminta ada di config (n_origin) dan "
                         "apakah `results_v3/tables/sku_sample_v3.csv` terisi")
    log(f"prediksi: {len(ada)} berkas pred_*.csv.gz")
    # 4) kebijakan, statistik, fakta
    # 'sens' wajib: scope_2metode.py membaca sensitivitas_v3.csv.gz
    jalankan_perintah([py, "src_v3/run_all_v3.py", "--config", "config_v3.yaml",
                       "--stages", "policy,stats,sens,facts", "--n-sku", str(n_sku),
                       "--origins", *[str(o) for o in origins]],
                      kerja, kerja / "results_v3" / "article_facts_v3.txt", paksa)
    # 5) tabel 2 metode + tabel terbuka (rev8) — keduanya butuh sapuan kebijakan semua titik
    kurang = [o for o in origins if not (kerja / "results_v3" / "tables" / f"policy_sweep_o{o}.csv.gz").exists()]
    if kurang:
        raise SystemExit(
            f"sapuan kebijakan titik {kurang} belum ada. Tabel H1/H2 memerlukan keempat titik "
            f"evaluasi; jalankan dengan --origins 1 2 3 4 (atau biarkan default).")
    for skrip, keluaran in (
        ("src_v3/scope_2metode.py", kerja / "results_v3" / "2metode" / "tables" / "akurasi_2metode.csv"),
        ("src_v3/cek_hol_mrel_rev8.py", kerja / "results_v3" / "2metode" / "hasil_h1_relatif_rev8.csv"),
        ("src_v3/cek_h2_relatif.py", kerja / "results_v3" / "2metode" / "hasil_h2_absolut_vs_relatif.csv"),
    ):
        jalankan_perintah([py, skrip], kerja, keluaran, paksa)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m5", type=Path, help="folder berisi sales_train_evaluation.csv, sell_prices.csv, calendar.csv")
    ap.add_argument("--kerja", type=Path, default=BUNDLE / "kerja", help="folder kerja (kode + hasil)")
    ap.add_argument("--hasil-dir", type=Path, default=None,
                    help="tempat mencari results_v3 (default: folder kerja)")
    ap.add_argument("--mode", choices=sorted(SETELAN_ARTIKEL), default="penuh")
    ap.add_argument("--n-sku", type=int, default=None)
    ap.add_argument("--origins", type=int, nargs="*", default=None)
    ap.add_argument("--abaikan-sha", action="store_true", help="lewati pemeriksaan SHA-256 berkas M5")
    ap.add_argument("--segarkan", action="store_true", help="salin ulang kode ke folder kerja")
    ap.add_argument("--paksa", action="store_true", help="jalankan semua tahap walau keluaran sudah ada")
    ap.add_argument("--parquet", type=Path, default=None,
                    help="pakai sales_long.parquet yang sudah ada (melewati tahap prep yang berat)")
    ap.add_argument("--hanya-cek", action="store_true", help="jangan jalankan pipeline, hanya nilai klaim")
    a = ap.parse_args(argv)

    setelan = SETELAN_ARTIKEL[a.mode]
    n_sku = a.n_sku or setelan["n_sku"]
    origins = a.origins or setelan["origins"]
    kerja = a.kerja.resolve()
    hasil_dir = (a.hasil_dir.resolve() if a.hasil_dir else kerja)
    sha_ok = False

    if not a.hanya_cek:
        if a.m5 is None:
            raise SystemExit("--m5 wajib diisi (atau pakai --hanya-cek)")
        m5 = a.m5.resolve()
        sha_ok, pesan = periksa_m5(m5, a.abaikan_sha)
        for baris in pesan:
            log(f"sha    : {baris}")
        log(f"mode   : {a.mode} | SKU {n_sku} | titik {origins}")
        siapkan_kerja(kerja, a.segarkan)
        patche_config(kerja, m5, n_sku, origins)
        tahapan(kerja, m5, n_sku, origins, a.paksa,
                parquet_siap=a.parquet.resolve() if a.parquet else None)

    klaim = klaim_cek.muat_klaim(KLAIM)
    log(f"periksa: {len(klaim)} klaim dari {KLAIM.name} (mode {a.mode})")
    baris = klaim_cek.periksa_semua(klaim, hasil_dir, mode=a.mode)
    ringkasan = klaim_cek.ringkas(baris)

    for b in baris:
        tanda = {"cocok": "OK  ", "tidak cocok": "BEDA", "tidak dapat diverifikasi": "--  "}[b["status"]]
        print(f"  {tanda} {b['id']:>4}  {b['teks']}")
        if b.get("pesan"):
            print(f"         {b['pesan']}")

    konteks = {"mode": a.mode, "n_sku": n_sku, "origins": origins,
               "m5": str(a.m5) if a.m5 else hasil_dir, "sha_ok": sha_ok}
    (hasil_dir / "laporan_verifikasi.md").write_text(
        klaim_cek.laporan_markdown(baris, konteks), encoding="utf-8")
    (hasil_dir / "laporan_verifikasi.json").write_text(
        json.dumps({"konteks": konteks, "ringkasan": ringkasan, "klaim": baris},
                   indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")

    log(f"ringkasan: {ringkasan['cocok']} cocok | {ringkasan['tidak_cocok']} tidak cocok | "
        f"{ringkasan['belum_terverifikasi']} belum terverifikasi (dari {ringkasan['total']})")
    log(f"laporan: {hasil_dir / 'laporan_verifikasi.md'}")
    return 0 if ringkasan["tidak_cocok"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
