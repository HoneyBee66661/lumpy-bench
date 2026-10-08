"""
siapkan_dataset_kaggle.py — rakit dataset Kaggle untuk lingkup 2 METODE.

Isi dataset: hanya berkas yang dipakai artikel 2 metode (RF vs Croston-SBA), sudah dipersempit
kolomnya supaya bersih dan ringan. Data mentah v3 tidak diubah; penyaringan ini hanya memilih
baris/kolom, dan setiap berkas sumber dicatat SHA-256-nya di README_sumber_data.md.

Keluaran: kaggle/dataset/ (siap untuk `kaggle datasets create`).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results_v3"
TAB = RES / "tables"
DS = ROOT / "kaggle" / "dataset"
DS.mkdir(parents=True, exist_ok=True)

METODE = ("rf_global_direct", "croston_sba")
KOLOM_COST = [
    "sku_id", "model", "tau", "ss", "demand", "unmet", "backlog_hari", "on_hand",
    "fill_rate", "pis", "hari_order", "n_hari_dibebankan",
    "cost_stockout", "cost_holding", "cost_total", "cs", "ch", "ratio_cs_ch",
    "store_id", "dept_id", "cat_id", "adi", "cv2", "adi_kuartil", "cv2_kuartil", "volume_kuartil",
]

log: list[str] = []


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dasar(nama: str) -> str:
    return str(nama).split("#")[0]


# ── 1. biaya per SKU (4 origin, 2 metode, kolom penting saja) ─────────────────────
for o in (1, 2, 3, 4):
    src = TAB / f"costs_o{o}.csv.gz"
    df = pd.read_csv(src, usecols=KOLOM_COST)
    df = df[df["model"].map(dasar).isin(METODE)].copy()
    df["metode"] = df["model"].map(dasar)
    out = DS / f"biaya_2metode_o{o}.parquet"
    df.to_parquet(out, index=False)
    log.append(f"biaya_2metode_o{o}.parquet: {len(df):,} baris dari {src.name} (sha sumber {sha(src)[:16]})")

# ── 2b. biaya pada fill target (service-equalized), 2 metode saja ──────────────────
# Ini masukan uji berpasangan: biaya pada fill rate target (interpolasi kurva tau per SKU).
for o in (1, 2, 3, 4):
    src = TAB / f"service_equalized_o{o}.csv.gz"
    df = pd.read_csv(src)
    df = df[df["model"].map(dasar).isin(METODE)].copy()
    df["metode"] = df["model"].map(dasar)
    out = DS / f"biaya_fill_target_2metode_o{o}.csv.gz"
    df.to_csv(out, index=False, compression="gzip")
    log.append(f"biaya_fill_target_2metode_o{o}.csv.gz: {len(df):,} baris dari {src.name}")

# ── 3. akurasi per SKU + rekap ────────────────────────────────────────────────────
ak = pd.read_csv(TAB / "metrics_akurasi_v3.csv.gz")
ak = ak[ak["model"].isin(METODE)].copy()
ak.to_csv(DS / "akurasi_per_sku_2metode.csv.gz", index=False, compression="gzip")
log.append(f"akurasi_per_sku_2metode.csv.gz: {len(ak):,} baris dari metrics_akurasi_v3.csv.gz")

rek = pd.read_csv(TAB / "metrics_rekap_v3.csv")
rek = rek[rek["model"].isin(METODE)].copy()
rek.to_csv(DS / "akurasi_rekap_2metode.csv", index=False)
log.append(f"akurasi_rekap_2metode.csv: {len(rek)} baris dari metrics_rekap_v3.csv")

# ── 3. hasil uji v3 sebagai pembanding verifikasi ────────────────────────────────
uji = pd.read_csv(TAB / "stats_pairwise_v3.csv")
uji = uji[(uji["model_a"].map(dasar) == METODE[0]) & (uji["model_b"].map(dasar) == METODE[1])]
uji.to_csv(DS / "pembanding_uji_berpasangan_v3.csv", index=False)
log.append(f"pembanding_uji_berpasangan_v3.csv: {len(uji)} baris (untuk uji silang di notebook)")

shutil.copy2(TAB / "stats_heterogenitas_v3.csv", DS / "pembanding_heterogenitas_v3.csv")
shutil.copy2(TAB / "stats_friedman_v3.csv", DS / "pembanding_friedman_v3.csv")
log.append("pembanding_heterogenitas_v3.csv + pembanding_friedman_v3.csv disalin")

# ── 4. dokumen pendukung ─────────────────────────────────────────────────────────
for nama, asal in [
    ("article_facts_2metode.txt", RES / "2metode" / "article_facts_2metode.txt"),
    ("AUDIT_2metode.txt", RES / "2metode" / "AUDIT_2metode.txt"),
    ("SCOPE_2METODE.md", RES / "2metode" / "SCOPE_2METODE.md"),
]:
    shutil.copy2(asal, DS / nama)

(DS / "README_sumber_data.md").write_text(
    "# Dataset: RF vs Croston-SBA (lingkup 2 metode)\n\n"
    "Turunan dari artefak pipeline v3 penelitian EMBS4460 yang sudah diverifikasi (0 FAIL).\n"
    "Isinya HANYA dua metode: `rf_global_direct` (Random Forest global) dan `croston_sba`.\n"
    "Metode lain tidak disertakan supaya artikel tetap berfokus pada keputusan persediaan.\n\n"
    "## Berkas\n\n"
    "| berkas | isi |\n|---|---|\n"
    "| `biaya_2metode_o{1..4}.parquet` | hasil simulasi persediaan per SKU per kebijakan (tau, ss) per origin; termasuk biaya kekurangan, biaya penyimpanan, total, dan profil SKU (ADI, CV2, kategori) |\n"
    "| `akurasi_per_sku_2metode.csv.gz` | MAE/MASE/RMSSE/ME/pinball per SKU per origin per split |\n"
    "| `akurasi_rekap_2metode.csv` | rekap agregat v3 (MAE = **median antar SKU**, bukan rata-rata) |\n"
    "| `pembanding_uji_berpasangan_v3.csv` | hasil uji v3 untuk pasangan dua metode ini — dipakai notebook sebagai uji silang |\n"
    "| `pembanding_heterogenitas_v3.csv`, `pembanding_friedman_v3.csv` | hasil uji v3 per profil SKU dan per skenario biaya |\n"
    "| `article_facts_2metode.txt` | daftar angka siap kutip |\n"
    "| `AUDIT_2metode.txt`, `SCOPE_2METODE.md` | jejak penyaringan dan lingkup artikel |\n\n"
    "## Konvensi yang wajib dibaca\n\n"
    "- `r_rank_biserial` dihitung dari `d = biaya_RF - biaya_Croston`; **r positif = RF lebih murah**.\n"
    "- Kebijakan persediaan: periode tinjauan R = 7 hari, tenggang waktu L = 1 hari, H = R + L = 8 hari.\n"
    "- Konvensi biaya utama `per_unit_once`; alternatif `per_unit_day` tersedia di tabel sensitivitas v3.\n"
    "- Angka v2 (termasuk p = 0,0098) sudah dicabut karena bug simulator dan kebocoran informasi.\n\n"
    "## Jejak penyaringan\n\n" + "\n".join(f"- {x}" for x in log) + "\n"
)

json.dump(
    {
        "title": "RF vs Croston 2metode hasil",
        "id": "honeybee66661/rf-vs-croston-2metode-hasil",
        "licenses": [{"name": "CC0-1.0"}],
    },
    open(DS / "dataset-metadata.json", "w"),
    indent=2,
)

print("dataset disiapkan di", DS)
for p in sorted(DS.iterdir()):
    print(f"  {p.stat().st_size/1024:8.1f} KB  {p.name}")
print(f"\ntotal: {sum(p.stat().st_size for p in DS.iterdir())/1024/1024:.2f} MB")
