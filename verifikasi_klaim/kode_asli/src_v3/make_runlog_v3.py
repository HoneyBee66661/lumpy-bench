"""make_runlog_v3.py — bangun results_v3/run_log_v3.txt (provenance lengkap, tanpa angka manual).

Yang dicatat (semua dibaca dari berkas/lingkungan, tidak ada yang ditulis tangan):
  * versi lingkungan (python, pustaka kunci) + versi di runner Kaggle (dari log kernel bila ada)
  * data mentah: SHA-256 5 berkas vs results/raw_sha256.txt (kalau berkasnya ada di host)
  * perintah yang dijalankan: sampling, forecast (per kernel Kaggle), policy/stats/sens/facts
  * waktu tiap berkas prediksi (results_v3/interim/forecast_log.txt)
  * kernel Kaggle: id, versi, URL (results_v3/run_registry.json)
  * keputusan desain yang menyimpang dari v2 + rujukan berkasnya
  * commit git yang relevan

Pakai:  .venv/bin/python src_v3/make_runlog_v3.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
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


def versi_venv() -> list[str]:
    py = ROOT / ".venv" / "bin" / "python"
    if not py.exists():
        return ["  (venv tidak ada)"]
    kode = (
        "import importlib.metadata as m\n"
        "for p in ['pandas','numpy','scikit-learn','scipy','statsforecast','pyarrow','matplotlib','joblib']:\n"
        "    try: print(f'  {p:14s} {m.version(p)}')\n"
        "    except Exception: print(f'  {p:14s} (tidak terpasang)')\n"
    )
    out = subprocess.run([str(py), "-c", kode], capture_output=True, text=True)
    return out.stdout.rstrip().splitlines() or ["  (gagal membaca versi)"]


def main() -> int:
    cfg = yaml.safe_load((ROOT / "config_v3.yaml").read_text())
    res = ROOT / cfg["paths"]["results"]
    interim = res / "interim"
    L: list[str] = []
    add = L.append

    add("=" * 100)
    add("RUN LOG v3 — Evaluasi Kinerja Random Forest versus Croston untuk Lumpy Demand: Pendekatan")
    add("Asymmetric Cost (pipeline v3: simulator net, forecast direct multi-horizon tanpa look-ahead)")
    add(f"dibangun: {time.strftime('%Y-%m-%d %H:%M:%S')} | host: {Path('/etc/hostname').read_text().strip() if Path('/etc/hostname').exists() else '-'}")
    add("=" * 100)
    add("")
    add("KONVENSI & KEPUTUSAN DESAIN YANG BERBEDA DARI v2 (rujukan: results_v3/analysis_plan_v3.md)")
    add("  * simulator: akuntansi NET (net_t = net_{t-1} + received_t - demand_t). Kedatangan melunasi")
    add("    backlog lebih dulu; stockout harian hanya bagian demand yang TIDAK terlayani hari itu.")
    add("    Regresi ke v2: simulate_legacy() mereproduksi cost_total v2 pada 240 baris (selisih maks")
    add("    3,6e-12, syarat < 1e-8) — tests/test_simulator_v3.py.")
    add("  * forecast periode proteksi TANPA look-ahead: target = jumlah demand t+1..t+H (H = R+L = 8),")
    add("    fitur hanya informasi <= t. Uji mutasi: mengubah demand setelah hari t tidak mengubah fitur")
    add("    hari t (selisih 0,00e+00) — tests/test_features_v3.py.")
    add("  * safety stock = kuantil empiris error periode proteksi pada jendela VALIDASI (per SKU, fallback")
    add("    pooled per kategori bila < 30 observasi) — bukan z*sigma*sqrt(L+R) berasumsi Normal.")
    add("  * perbandingan SERVICE-EQUALIZED: tau disapu 0,50..0,99 -> kurva biaya vs fill rate -> biaya")
    add("    pada fill 90/95/98%; bukan pada safety factor yang sama untuk semua model.")
    add("  * hari yang dibebankan biaya = hari yang jendela proteksinya (t+1..t+H) lengkap di dalam data;")
    add("    warm-up 28 hari pertama periode uji tidak dibebankan. Aturan sama untuk semua model.")
    add("  * ADI/CV2 pada PERIODE AKTIF SKU (rentang hari yang punya harga di sell_prices), bukan 1.761")
    add("    hari seragam; kriteria lumpy: ADI > 1,32 DAN CV2 > 0,49 DAN n_nonzero >= 3.")
    add("  * statistik: Wilcoxon berpasangan satu sisi (arah pra-registrasi) + Holm DI DALAM keluarga uji,")
    add("    Hodges-Lehmann + CI cluster bootstrap (toko & departemen), Friedman lintas skenario (H2).")
    add("")
    add("-" * 100)
    add("PARAMETER (config_v3.yaml)")
    add("-" * 100)
    add(f"  seed utama {cfg['seed']} | seed RF dijalankan {cfg['seeds_rf_dijalankan']} (triase dari {cfg['seeds_rf']})")
    add(f"  origin: {cfg['origin']['n_origin']} (expanding window), {cfg['origin']['test_days']} hari uji + "
        f"{cfg['split']['validation_days']} hari validasi per origin")
    add(f"  kebijakan: R = {cfg['policy_v3']['review_period_days']}, L = {cfg['policy_v3']['lead_time_days']}, "
        f"H = {cfg['policy_v3']['protection_days']}, konvensi biaya {cfg['policy_v3']['cost_convention_utama']}")
    add(f"  sapuan tau: {cfg['policy_v3']['quantile_levels']}")
    add(f"  skenario biaya (Cs:Ch): {cfg['cost_scenarios']} | fill target {cfg['policy_v3']['fill_rate_targets']}")
    add(f"  sampel: {cfg['sampling_v3']['n_sku']} SKU dari kriteria lumpy periode aktif, stratifikasi "
        f"{cfg['sampling_v3']['stratify_by']}")
    add("")
    add("-" * 100)
    add("LINGKUNGAN LOKAL (.venv)")
    add("-" * 100)
    add(f"  python: {sys.version.split()[0]} | executable: {sys.executable}")
    L.extend(versi_venv())
    add("  runner Kaggle: lihat blok 'pip exit' di log kernel (statsforecast 2.0.3, lightgbm 4.6.0) —")
    add("  dalam berkas ini disalin dari results_v3/log_kernel_*.txt bila tersedia.")
    for f in sorted(res.glob("log_kernel_*.txt")):
        add(f"  --- {f.name} ---")
        for line in f.read_text().splitlines():
            if line.startswith(("pip exit", "statsforecast ", "lightgbm ")):
                add(f"    {line}")
    add("")
    add("-" * 100)
    add("DATA MENTAH")
    add("-" * 100)
    raw_sha = ROOT / "results" / "raw_sha256.txt"
    if raw_sha.exists():
        add(f"  rujukan SHA: results/raw_sha256.txt")
        for line in raw_sha.read_text().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            sha, nama = line.split()[0], line.split()[-1]
            p = ROOT / "data" / "raw" / Path(nama).name
            if p.exists():
                ada = sha256(p)
                add(f"  {'COCOK' if ada == sha else 'BEDA '}  {Path(nama).name}  {ada[:16]}...")
            else:
                add(f"  (tidak ada di host) {Path(nama).name}  rujukan {sha[:16]}...")
    else:
        add("  results/raw_sha256.txt tidak ada")
    add("")
    add("-" * 100)
    add("PERINTAH YANG DIJALANKAN")
    add("-" * 100)
    add("  1) sampling + klasifikasi lumpy (lokal, streaming parquet 59,2 juta baris):")
    add("     .venv/bin/python src_v3/sample_v3.py --config config_v3.yaml")
    add("  2) forecast periode proteksi (Kaggle, 3 kernel; sumber kode ditanam + SHA diverifikasi di notebook):")
    reg = res / "run_registry.json"
    if reg.exists():
        for k in json.loads(reg.read_text()):
            add(f"     - {k['kernel']} v{k.get('version', '?')}  {k.get('url', '')}")
            if k.get("catatan"):
                add(f"       catatan: {k['catatan']}")
    else:
        add("     (results_v3/run_registry.json belum ada)")
    add("  3) kebijakan + statistik + sensitivitas + fakta (lokal, memakai keluaran kernel):")
    add("     .venv/bin/python src_v3/run_all_v3.py --stages policy,stats,sens,facts")
    add("  4) verifikasi lokal independen (WAJIB lulus sebelum angka dikutip):")
    add("     .venv/bin/python tests/test_simulator_v3.py")
    add("     .venv/bin/python tests/test_features_v3.py")
    add("     .venv/bin/python tests/test_stats_v3.py")
    add("     .venv/bin/python tests/verify_v3_forecasts.py --n-sku 6 --n-titik 12")
    add("     .venv/bin/python tests/verify_v3_policy.py --n-sku 5 --n-tau 2")
    add("  5) gambar: .venv/bin/python src_v3/make_figures_v3.py --origin 4")
    add("")
    log = interim / "forecast_log.txt"
    if log.exists():
        add("-" * 100)
        add("WAKTU PRODUKSI FORECAST (dari results_v3/interim/forecast_log.txt)")
        add("-" * 100)
        rows = []
        for line in log.read_text().splitlines():
            bagian = line.split()
            if len(bagian) < 8:
                continue
            rows.append({"waktu": f"{bagian[0]} {bagian[1]}", "model": bagian[2], "origin": bagian[3],
                         "seed": bagian[4], "baris": bagian[5].split("=")[-1], "nan": bagian[6].split("=")[-1],
                         "durasi_s": bagian[7].replace("s", "")})
        if rows:
            df = pd.DataFrame(rows)
            add(f"  {len(df)} berkas prediksi | total durasi {pd.to_numeric(df.durasi_s).sum():,.0f} s "
                f"({pd.to_numeric(df.durasi_s).sum() / 60:,.1f} menit)")
            grup = df.assign(durasi_s=pd.to_numeric(df.durasi_s)).groupby("model")["durasi_s"].agg(["count", "sum"])
            for m, r in grup.iterrows():
                add(f"    {m:26s} {int(r['count']):3d} berkas | {r['sum']:8.1f} s")
    add("")
    add("-" * 100)
    add("COMMIT GIT YANG RELEVAN")
    add("-" * 100)
    out = subprocess.run(["git", "log", "--oneline", "-25"], cwd=ROOT, capture_output=True, text=True)
    for line in out.stdout.splitlines():
        add(f"  {line}")
    add("")
    tujuan = res / "run_log_v3.txt"
    tujuan.write_text("\n".join(L) + "\n")
    print(f"run log -> {tujuan.relative_to(ROOT)} ({len(L)} baris)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
