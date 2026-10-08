# Verifikasi angka klaim pada data M5 Anda

Folder ini membuat klaim angka di artikel (Random Forest global vs Croston-SBA) bisa **dibuktikan
sendiri oleh pembaca**: cukup berikan data M5, jalankan satu perintah (atau satu tombol di UI),
dan setiap klaim diperiksa terhadap keluaran pipeline-nya.

```bash
pip install -r requirements-verifikasi.txt

# 1) lewat UI sederhana
streamlit run app.py

# 2) lewat perintah (setelan artikel: 300 SKU, 4 titik evaluasi)
python jalankan.py --m5 ~/m5 --kerja kerja

# uji cepat di subset (hasil klaim otomatis ditandai "belum terverifikasi")
python jalankan.py --m5 ~/m5 --mode cepat --n-sku 30 --origins 4
```

## Yang perlu Anda siapkan

Folder `--m5` berisi tiga berkas M5 (unduh sendiri dari Kaggle, ketentuannya melarang
redistribusi): `sales_train_evaluation.csv`, `sell_prices.csv`, `calendar.csv`.
Skrip memeriksa **SHA-256** ketiganya terhadap nilai resmi supaya Anda tahu datanya identik dengan
yang dipakai artikel:

```
d12b5914… calendar.csv
4b4a47c4… sales_train_evaluation.csv
9da3ad1f… sell_prices.csv
```

Kalau hash berbeda, verifikasi berhenti (kecuali Anda menambahkan `--abaikan-sha`).

## Alur yang dijalankan

| Tahap | Perintah asli | Keluaran |
|---|---|---|
| prep | `src/data_prep.py` | `data/interim/sales_long.parquet` |
| klasifikasi + sampel | `src_v3/sample_v3.py` | `results_v3/tables/sku_sample_v3.csv` |
| peramalan | `src_v3/run_forecasts_v3.py` | `results_v3/interim/pred_*.csv.gz` |
| kebijakan & statistik | `src_v3/run_all_v3.py` | tabel uji, akurasi, sensitivitas |
| lingkup 2 metode | `src_v3/scope_2metode.py` | `results_v3/2metode/tables/*` |
| tabel terbuka | `src_v3/cek_hol_mrel_rev8.py`, `src_v3/cek_h2_relatif.py` | tabel H1/H2 rev8 |
| penilaian klaim | `jalankan.py` | `laporan_verifikasi.md` / `.json` |

Kode pipeline di `kode_asli/` disalin apa adanya dari repo kerja: angka klaim diverifikasi oleh
**kode yang memang menghasilkannya**, bukan oleh implementasi lain.

## Arti status

- **cocok** — keluaran memenuhi klaim dalam toleransi yang ditulis di `klaim/klaim_artikel.json`
- **tidak cocok** — keluaran berbeda; bandingkan angka terukurnya dengan klaim
- **tidak dapat diverifikasi** — berkas keluaran belum ada (mis. klaim yang hanya sah diuji pada
  mode penuh, tetapi Anda menjalankan mode cepat)

## Catatan kejujuran

- Waktu jalan mode penuh bisa berjam-jam dan butuh RAM besar; pada mesin kecil pakailah Kaggle
  Notebook (mode aslinya memang begitu) atau mode cepat.
- Kecocokan sampai digit terakhir hanya dijamin pada lingkungan dengan versi pustaka yang sama
  (`kode_asli/requirements.lock.txt`) dan seed yang sama (42). Versi pustaka berbeda atau jumlah
  thread berbeda dapat menggeser angka di digit terakhir — perbedaan seperti itu dilaporkan apa
  adanya sebagai "tidak cocok", bukan disembunyikan.
- Modul `lumpy_bench` di repo ini adalah tulis ulang bersih untuk dataset lain, **bukan** jalur
  pembuktian angka artikel. Jangan mencamp keduanya.
