
"""Diagram alur proses pengujian: data mentah M5 -> hasil akhir (proyek RF vs Croston-SBA)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib import font_manager

# pakai font serif bila tersedia
for nama in ("Liberation Serif", "DejaVu Serif", "Times New Roman"):
    if any(f.name == nama for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = nama
        break
plt.rcParams["font.size"] = 8.5

TAHAP = [
    ("A", "Data mentah (M5 / Walmart)", [
        "sales_train_evaluation.csv (30.490 deret item-toko, 1.941 hari)",
        "calendar.csv (event, SNAP) | sell_prices.csv (harga jual) | sample_submission.csv",
        "SHA-256 berkas resmi diverifikasi lebih dahulu"]),
    ("B", "Penyiapan data", [
        "Gabung kalender + harga, ubah ke format panjang (melt)",
        "Tandai periode aktif per deret: hari yang punya harga jual (peluncuran s.d. penghentian)",
        "Keluaran: data/interim/sales_long.parquet"]),
    ("C", "Klasifikasi pola permintaan", [
        "ADI = n_hari_aktif / n_hari_berpermintaan ; CV2 = (sd/mean)2 ukuran permintaan",
        "Lumpy bila ADI > 1,32 dan CV2 > 0,49 (Syntetos dkk., 2005)",
        "Hasil: 5.602 dari 30.490 deret lumpy (18,4%) -> results_v3/tables/lumpy_v3.csv"]),
    ("D", "Pengambilan sampel berstrata", [
        "Strata: kuartil ADI x kuartil CV2 x kategori produk x negara bagian",
        "300 SKU terpilih -> sku_sample_v3.csv",
        "Pemeriksaan ketahanan: ADI/CV2 dihitung ulang pada hari <= 1.221 (pra-uji terawal), 206 SKU bertahan"]),
    ("E", "Pembagian jendela evaluasi (rolling origin)", [
        "4 titik evaluasi, tiap titik: latih (expanding) | validasi 180 hari | uji 180 hari",
        "Titik 1 uji 1.222-1.401 ... titik 4 uji 1.762-1.941",
        "Hanya data sebelum jendela uji masuk ke pelatihan (uji kebocoran fitur)"]),
    ("F", "Pembentukan fitur", [
        "Lag 1/7/14/28 hari, rata-rata bergerak 7 & 28 hari, simpangan baku 28 hari",
        "Penanda kalender: event, SNAP, agregat jendela proteksi (hari t+1..t+8)",
        "Identitas SKU/kategori/toko; harga jual TIDAK dijadikan penjelas",
        "Target langsung (direct multi-horizon) H = 8 hari = R + L = 1 + 7"]),
    ("G", "Dua metode yang dibandingkan", [
        "Random Forest global: 600 pohon, kedalaman 10, daun min. 10 (scikit-learn 1.9.1);",
        "   dilatih ulang tiap titik evaluasi; benih 42/43/44 sebagai pemeriksaan",
        "Croston-SBA: pustaka statsforecast 2.0.3, pemulusan alfa = 0,10, faktor debiasing 0,95",
        "Keluaran: prediksi per titik/model -> pred_*_o*_s*.csv.gz"]),
    ("H", "Simulasi kebijakan persediaan", [
        "Order-up-to periodik: R = 1 hari, L = 7 hari, jendela proteksi 8 hari",
        "Persediaan pengaman = kuantil empiris galat validasi per SKU, grid tau 0,50-0,99 (10 level)",
        "Model kekurangan stok: backorder (tunggakan dipenuhi saat barang datang); pemanasan 28 hari",
        "Diuji invarian I1-I6 (mass balance, tidak ada stok negatif, tunggakan turun saat datang)",
        "Keluaran: policy_sweep_o*.csv.gz (fill rate, unmet, on_hand per tau)"]),
    ("I", "Penyetaraan tingkat layanan", [
        "Target fill rate 90%, 95%, 98% (volume fill rate = 1 - unit tak terlayani / permintaan)",
        "Biaya pada target = interpolasi linear kurva (fill rate, biaya) per SKU x model",
        "SKU yang kurvanya tidak mencapai target dikeluarkan dari perbandingan berpasangan",
        "Keluaran: service_equalized_o*.csv.gz (N efektif 171-266 dari 300 SKU)"]),
    ("J", "Penilaian biaya dan akurasi", [
        "Biaya = biaya kehabisan stok (Cs x unit belum terlayani) + biaya simpan (Ch x on_hand)",
        "Skenario Cs:Ch = 5:1, 10:1, 20:1 (sensitifitas skala; identik karena layanan disamakan)",
        "Metrik akurasi: MAE, MASE, RMSSE, ME, pinball loss pada kuantil titik operasi"]),
    ("K", "Inferensi statistik", [
        "H1: Wilcoxon signed-rank satu sisi atas selisih biaya berpasangan per SKU + koreksi Holm",
        "H2: uji Page satu sisi (pola berurutan 90% -> 95% -> 98%) + kontras 98% vs 90%",
        "Selang kepercayaan: bootstrap median, klaster per toko (indikatif, 10 toko)",
        "Pemeriksaan heterogenitas per kategori/kuartil ADI-CV2; diagnostik pengguguran SKU"]),
    ("L", "Hasil akhir", [
        "H1 tidak didukung: 0 dari 4 titik lolos Holm (p terkecil 0,1233 pada ukuran relatif)",
        "H2 didukung sebagian: arah naik 4/4 titik; Page lolos Holm di 2/4 (pergeseran 0,9-2,2% biaya)",
        "Pemeriksaan ekor: RF unggul pada MAE dan pinball -> dugaan ekor galat tidak didukung",
        "Tabel/gambar: hasil_h1_*.csv, hasil_h2_*.csv, fig_biaya_dan_layanan_vs_tau.png, fig_heterogenitas.png"]),
    ("M", "Reproduksi dan publikasi", [
        "Dataset turunan di Kaggle (7,2 MB, 18 berkas) + notebook publikasi v13: 23 pemeriksaan lulus",
        "Perhitungan ulang di Kaggle identik dengan pipeline lokal (selisih median 2,3e-13)",
        "Naskah: Pendahuluan - Kajian Teori & Hipotesis (+ Metode, Hasil, Pembahasan pada tahap berikutnya)"]),
]

WARNA = {
    "A": "#E8EEF7", "B": "#E8EEF7", "C": "#E9F3E9", "D": "#E9F3E9",
    "E": "#FBF3E2", "F": "#FBF3E2", "G": "#FBF3E2",
    "H": "#F6E9F2", "I": "#F6E9F2", "J": "#F6E9F2",
    "K": "#EDEDF7", "L": "#EDEDF7", "M": "#EDEDF7",
}
FASE = {"A": "MASUKAN", "B": "MASUKAN", "C": "POPULASI & SAMPEL", "D": "POPULASI & SAMPEL",
        "E": "PERAMALAN", "F": "PERAMALAN", "G": "PERAMALAN",
        "H": "SIMULASI & PENILAIAN", "I": "SIMULASI & PENILAIAN", "J": "SIMULASI & PENILAIAN",
        "K": "INFERENSI & HASIL", "L": "INFERENSI & HASIL", "M": "INFERENSI & HASIL"}

# hitung tinggi tiap kotak
tinggi, GAP = [], 0.42
for _, judul, detail in TAHAP:
    tinggi.append(0.34 + 0.20 * len(detail) + 0.10)
total = sum(tinggi) + GAP * (len(TAHAP) - 1) + 1.0
LEBAR = 11.0
TINGGI_FIG = total          # 1 satuan = 1 inci agar tidak terpotong
fig = plt.figure(figsize=(LEBAR, TINGGI_FIG))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, LEBAR); ax.set_ylim(0, TINGGI_FIG)
ax.axis("off")

y = TINGGI_FIG - 0.55
ax.text(LEBAR / 2, y + 0.30, "Alur Proses Pengujian: Random Forest vs Croston-SBA",
        ha="center", va="center", fontsize=13, fontweight="bold")
ax.text(LEBAR / 2, y + 0.08, "dari data mentah M5 sampai hasil akhir (pipeline v3, lingkup 2 metode)",
        ha="center", va="center", fontsize=9, style="italic")
y -= 0.30

x0, x1 = 1.45, LEBAR - 0.35
for (kode, judul, detail), h in zip(TAHAP, tinggi):
    # pita fase
    ax.add_patch(Rectangle((0.15, y - h), 1.15, h, facecolor="#DCE3EC", edgecolor="#9AA7B8", lw=0.6))
    ax.text(0.72, y - h / 2, FASE[kode], rotation=90, ha="center", va="center", fontsize=6.8,
            color="#2E3B4E", fontweight="bold")
    # kotak tahap
    ax.add_patch(FancyBboxPatch((x0, y - h), x1 - x0, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor=WARNA[kode], edgecolor="#5A6B80", lw=0.8))
    ax.text(x0 + 0.12, y - 0.22, f"{kode}. {judul}", ha="left", va="center", fontsize=9.2, fontweight="bold")
    for i, d in enumerate(detail):
        ax.text(x0 + 0.32, y - 0.46 - 0.20 * i, "\u2022 " + d, ha="left", va="center", fontsize=7.4)
    y -= h
    if kode != "M":
        ax.add_patch(FancyArrowPatch((LEBAR / 2, y - 0.02), (LEBAR / 2, y - GAP + 0.04),
                                     arrowstyle="-|>", mutation_scale=11, lw=1.0, color="#5A6B80"))
        y -= GAP

OUT = "/home/ubuntu/demand-forecast-rf-vs-croston/docs/figures/alur_proses_pengujian"
import os
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT + ".png", dpi=200, facecolor="white")
fig.savefig(OUT + ".svg", facecolor="white")
print("tersimpan:", OUT + ".png", "dan", OUT + ".svg")
