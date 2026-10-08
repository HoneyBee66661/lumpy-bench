"""
scope_2metode.py — siapkan bahan artikel versi 2 METODE SAJA (RF global vs Croston-SBA).

Latar: pipeline v3 membandingkan 13 metode (naive, MA7, MA28, SES, Croston klasik, Croston-SBA,
Croston-SBA optimized, TSB, ADIDA, IMAPA, LightGBM-Tweedie, RF recursive, RF global). Untuk artikel
manajemen, lingkup dipersempit: hanya RF global vs Croston-SBA, supaya bahasannya tidak berbelok ke
teknik peramalan/informatika.

ATURAN: skrip ini TIDAK menghitung ulang statistik apa pun. Semua angka diambil apa adanya dari
artefak v3 yang sudah diverifikasi (0 FAIL), lalu disaring ke dua metode. Setiap baris ditelusuri
balik ke berkas asalnya dan dicatat di audit.

Keluaran: results_v3/2metode/  (tabel siap pakai + fakta angka + catatan lingkup)
"""
from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results_v3"
TAB = RES / "tables"
OUT = RES / "2metode"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "tables").mkdir(exist_ok=True)

METHOD_A = "rf_global_direct"  # Random Forest (model global)
METHOD_B = "croston_sba"       # Croston-SBA
# RF dijalankan pada beberapa seed; semuanya bagian dari metode A.
A_VARIANTS = {"rf_global_direct", "rf_global_direct#s42", "rf_global_direct#s43", "rf_global_direct#s44"}

audit: list[str] = []


def catat(pesan: str) -> None:
    audit.append(pesan)
    print("  ", pesan)


def dasar_model(nama: str) -> str:
    """'rf_global_direct#s42' -> 'rf_global_direct'."""
    return str(nama).split("#")[0]


def relevan(nama: str) -> bool:
    """True kalau kolom model termasuk dua metode yang dipertahankan."""
    return dasar_model(nama) in {METHOD_A, METHOD_B}


# ── 1. akurasi peramalan (angka yang boleh dikutip: rekap agregat v3) ───────────────
# Catatan penting: angka yang dikutip artikel harus berasal dari rekap agregat v3
# (metrics_rekap_v3.csv), bukan rata-rata sendiri atas baris per-SKU — dua agregasi itu
# berbeda nilai (mis. MAE o1 test croston_sba: 4,5278 di rekap vs 7,99 bila dirata-ratakan
# per SKU). Untuk ketertelusuran, yang dipakai selalu rekap.
rekap = pd.read_csv(TAB / "metrics_rekap_v3.csv")
rekap = rekap[rekap["model"].isin([METHOD_A, METHOD_B])].copy()
rekap = rekap.rename(columns={"model": "metode"}).sort_values(["split", "origin", "metode"])
rekap.to_csv(OUT / "tables" / "akurasi_2metode.csv", index=False)
catat(f"akurasi: {len(rekap)} baris dari metrics_rekap_v3.csv (agregat terverifikasi v3)")

# Detail per SKU tetap disimpan sebagai lampiran (bukan untuk dikutip langsung).
ak = pd.read_csv(TAB / "metrics_akurasi_v3.csv.gz")
ak = ak[ak["model"].isin([METHOD_A, METHOD_B])].copy()
ak.to_csv(OUT / "tables" / "akurasi_per_sku_2metode.csv.gz", index=False, compression="gzip")
catat(f"lampiran per SKU: {len(ak)} baris dari metrics_akurasi_v3.csv.gz")

# ── 2. uji berpasangan biaya (sudah dihitung di v3, hanya disaring) ─────────────────
uji = pd.read_csv(TAB / "stats_pairwise_v3.csv")
uji = uji[uji["model_a"].map(relevan) & uji["model_b"].map(relevan)].copy()
uji = uji[(uji["model_a"].map(dasar_model) == METHOD_A) & (uji["model_b"].map(dasar_model) == METHOD_B)]
kol = [
    "origin", "cs", "fill_target", "model_a", "model_b", "n_sku",
    "median_a", "median_b", "median_diff", "hodges_lehmann",
    "p_one_sided_a_lebih_murah", "r_rank_biserial", "menang_a", "menang_b",
    "ci_store_id_lo", "ci_store_id_hi", "p_one_sided_holm",
]
uji[kol].to_csv(OUT / "tables" / "uji_berpasangan_2metode.csv", index=False)
catat(f"uji berpasangan: {len(uji)} baris (sumber stats_pairwise_v3.csv, tanpa hitung ulang)")

# ── 3. uji interaksi skenario biaya (Friedman) ─────────────────────────────────────
fr = pd.read_csv(TAB / "stats_friedman_v3.csv")
fr = fr[(fr["model_a"].map(relevan)) & (fr["model_b"].map(relevan))]
fr.to_csv(OUT / "tables" / "friedman_2metode.csv", index=False)
catat(f"friedman (interaksi skenario biaya): {len(fr)} baris")

# ── 4. heterogenitas: pada profil SKU mana bedanya berarti ─────────────────────────
he = pd.read_csv(TAB / "stats_heterogenitas_v3.csv")
he.to_csv(OUT / "tables" / "heterogenitas_2metode.csv", index=False)
catat(f"heterogenitas: {len(he)} baris (semuanya sudah pasangan RF vs Croston-SBA)")

# ── 5. sensitivitas: hanya baris dua metode ────────────────────────────────────────
sens_rows = []
with gzip.open(TAB / "sensitivitas_v3.csv.gz", "rt") as fh:
    header = fh.readline().rstrip("\n").split(",")
    idx_model = header.index("model")
    for line in fh:
        parts = line.rstrip("\n").split(",")
        if len(parts) > idx_model and relevan(parts[idx_model]):
            sens_rows.append(parts)
sens = pd.DataFrame(sens_rows, columns=header)
sens.to_csv(OUT / "tables" / "sensitivitas_2metode.csv.gz", index=False, compression="gzip")
varian = sorted(sens["varian"].unique()) if "varian" in sens.columns else []
status = sorted(sens["status"].unique()) if "status" in sens.columns else []
catat(f"sensitivitas: {len(sens)} baris, varian={varian}, status={status}")

# ── 6. fakta angka versi 2 metode (bahasa manajemen, tanpa jargon ML) ──────────────
def baris(pesan: str = "") -> str:
    return pesan


L: list[str] = []
L.append("FAKTA ANGKA — VERSI 2 METODE (RF vs CROSTON-SBA)")
L.append("disaring dari artefak pipeline v3 yang sudah diverifikasi (0 FAIL) tanpa menghitung ulang")
L.append("sumber: results_v3/tables/*  |  lingkup: hanya Random Forest (global) dan Croston-SBA")
L.append("")
L.append("KONVENSI ARAH (WAJIB SEBELUM MENGUTIP ANGKA)")
L.append("  * r_rank_biserial dari d = biaya_RF - biaya_Croston. r POSITIF = RF LEBIH MURAH.")
L.append("    Uji utama satu sisi (arah: RF lebih murah).")
L.append("  * biaya konvensi utama: per_unit_once (unit tidak terlayani dibebankan sekali).")
L.append("  * kebijakan persediaan: periode tinjauan R = 1 hari, tenggang waktu L = 7 hari, H = R + L = 8 hari.")
L.append("  * hari dibebankan hanya bila jendela proteksi t+1..t+H lengkap di dalam data.")
L.append("")
L.append("AKURASI PERAMALAN (rekap agregat v3; 300 SKU per origin)")
for split in ("val", "test"):
    L.append(f"  [{split}]")
    sub = rekap[rekap["split"] == split]
    for _, r in sub.iterrows():
        L.append(f"    o{int(r['origin'])} {r['metode']:<16} MAE {r['mae']:.4f} | MASE {r['mase']:.3f} | ME {r['me']:+.4f}")
L.append("")
L.append("KELUARGA UJI (WAJIB DISEBUT DI METODE ARTIKEL)")
L.append("  * Artikel ini melakukan 36 uji berpasangan (4 origin x 3 rasio biaya x 3 tingkat layanan).")
L.append("  * p_holm pada tabel di bawah ini berasal dari tabel v3 dengan keluarga 117 uji (13 metode).")
L.append("  * Bila keluarga dipersempit ke 36 uji artikel: p Holm terkecil 0,0996 -> tidak ada yang signifikan.")
L.append("  * Bila keluarga 9 uji per origin: 6 uji lolos Holm (terkecil 0,0249).")
L.append("  * Angka yang dikutip artikel harus menyebut keluarga ujinya. Sumber hitung ulang:")
L.append("    notebook Kaggle honeybee66661/rf-vs-croston-2metode (identik dengan tabel v3 sampai 1e-13).")
L.append("")
L.append("BIAYA PERSEDIAAN BERPASANGAN (median per SKU, konvensi per_unit_once)")
for _, r in uji.sort_values(["cs", "origin"]).iterrows():
    L.append(
        f"  cs {r['cs']:>4.0f}:1  o{int(r['origin'])}  "
        f"RF {r['median_a']:>10.2f} | Croston-SBA {r['median_b']:>10.2f} | selisih {r['median_diff']:>+9.2f} "
        f"| p(1-sisi) {r['p_one_sided_a_lebih_murah']:.4f} | r_rb {r['r_rank_biserial']:+.3f} "
        f"| RF menang {int(r['menang_a'])}/{int(r['menang_a'] + r['menang_b'])} SKU"
    )
L.append("")
L.append("UJI INTERAKSI SKENARIO BIAYA (Friedman per SKU, H2)")
for _, r in fr.iterrows():
    L.append(f"  o{int(r['origin'])}  n_sku {int(r['n_sku'])}  chi2 {r['statistic']:.4f}  p {r['p_value']:.4f}")
L.append("")
L.append("HETEROGENITAS: pada profil SKU mana selisihnya berarti (fill 95%)")
for dim in sorted(he["dimensi"].unique()):
    sub = he[(he["dimensi"] == dim) & (he["origin"] == 1) & (he["cs"] == 5.0)].sort_values("p_one_sided")
    if sub.empty:
        continue
    L.append(f"  [{dim}]")
    for _, r in sub.iterrows():
        L.append(
            f"    {str(r['nilai']):<9} n {int(r['n_sku']):>3} | median relatif {r['median_relatif_pct']:+.2f}% "
            f"| p {r['p_one_sided']:.4f} | r_rb {r['r_rb']:+.3f} | p_holm {r['p_holm_dalam_dimensi']:.4f}"
        )
L.append("")
L.append("SENSITIVITAS (hanya dua metode)")
if varian:
    L.append(f"  varian diuji: {', '.join(varian)}")
for (v, st), sub in sens.groupby(["varian", "status"]) if varian and status else []:
    L.append(f"    {v} / {st}: {len(sub)} baris")
L.append("")
L.append("CATATAN PENULISAN")
L.append("  * Tidak ada angka baru di berkas ini: semuanya hasil penyaringan artefak v3.")
L.append("  * Metode lain yang dibandingkan di v3 (naive, MA7/MA28, SES, Croston klasik, Croston-SBA")
L.append("    optimized, TSB, ADIDA, IMAPA, LightGBM-Tweedie, RF recursive) tidak dipakai di artikel ini.")
L.append("  * Bagian teknis (tuning hiperparameter, feature importance, ragam seed, RMSSE/pinball) tidak")
L.append("    masuk batang tubuh; kalau perlu, cukup sebagai catatan kaki atau lampiran.")
(OUT / "article_facts_2metode.txt").write_text("\n".join(L) + "\n")
catat(f"fakta angka: article_facts_2metode.txt ({len(L)} baris)")

# ── 7. audit ketertelusuran ────────────────────────────────────────────────────────
audit_path = OUT / "AUDIT_2metode.txt"
audit_path.write_text(
    "AUDIT PENYARINGAN 2 METODE\n"
    "Prinsip: tidak ada angka baru. Semua nilai disalin dari artefak v3 hasil verifikasi.\n\n"
    + "\n".join(f"- {a}" for a in audit)
    + "\n\nBerkas masukan (SHA-256 dihitung ulang di sini):\n"
    + "\n".join(
        f"- {p.name}: {__import__('hashlib').sha256(p.read_bytes()).hexdigest()[:16]}"
        for p in sorted(TAB.glob("*")) if p.is_file()
    )
    + "\n"
)
print("\nselesai ->", OUT)
