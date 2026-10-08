# lumpy-bench

[![CI](https://github.com/HoneyBee66661/lumpy-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/HoneyBee66661/lumpy-bench/actions/workflows/ci.yml)

Kerangka pengujian **peramalan permintaan lumpy pada kebijakan persediaan**: membandingkan satu
model pembelajaran mesin (Random Forest global) dengan satu metode intermiten klasik
(Croston-SBA) pada tingkat layanan yang disamakan, lalu menguji apakah selisih biaya bergeser
ketika target layanan diketatkan.

Repositori ini adalah **kerangka kerja (framework)**, bukan paket hasil. Seluruh alur dapat
dijalankan dari data contoh sintetis — tanpa perlu mengunduh data kompetisi mana pun — dan
menghasilkan artefak yang dapat dibuktikan regenerasinya lewat manifest SHA-256.

## Mulai cepat (3 perintah)

```bash
pip install -r requirements.txt
python -m lumpy_bench regenerate          # jalankan alur penuh -> folder out/
python -m lumpy_bench verify              # hitung ulang, bandingkan checksum -> "verify: SAH"
```

Data contoh sintetis (150 deret x 900 hari) sudah disertakan sebagai
`data_contoh/sample_retail_long.csv.gz` (760 KB). Untuk membuat ulang atau mengubah ukurannya:

```bash
python -m lumpy_bench synth-data --out data_contoh/sample_retail_long.csv --n-sku 150 --n-hari 900
```

Keluaran ada di `out/`:

| Berkas | Isi |
|---|---|
| `tables/statistik_sku.csv` | ADI, CV2, dan penanda lumpy per deret |
| `tables/sampel.csv` | sampel berstrata yang dipakai |
| `tables/kebijakan.csv` | hasil simulasi order-up-to per SKU x model x tau |
| `tables/penyetaraan.csv` | biaya terinterpolasi pada target fill rate |
| `tables/akurasi.csv` | MAE per SKU x model x titik evaluasi |
| `tables/hasil_h1.csv`, `tables/hasil_h2.csv` | uji hipotesis (Wilcoxon + Holm; Page + kontras) |
| `ringkasan.json` | ringkasan keputusan |
| `manifest.sha256.json` | checksum seluruh artefak |

## Alur

1. **Periode aktif** — hari yang punya harga jual (peluncuran s.d. penghentian); di luar itu nol struktural.
2. **Klasifikasi lumpy** — ADI > 1,32 dan CV2 > 0,49 (Syntetos dkk., 2005).
3. **Sampel berstrata** — kuartil ADI x kuartil CV2 x kategori x negara bagian.
4. **Evaluasi bergulir (rolling origin)** — jendela latih / validasi / uji yang saling tidak tumpang tindih.
5. **Fitur** — lag 1/7/14/28, rata-rata bergerak 7 & 28, simpangan baku 28, penanda kalender dan SNAP;
   target langsung H-hari dengan H = R + L.
6. **Dua metode** — Random Forest global (scikit-learn) dan Croston-SBA (pemulusan alpha = 0,10,
   faktor debiasing 0,95 = 1 - alpha/2).
7. **Simulasi kebijakan** — order-up-to periodik, tenggang waktu L hari, model backorder,
   persediaan pengaman = kuantil empiris galat validasi per SKU (grid tau).
8. **Penyetaraan tingkat layanan** — biaya pada target fill rate = interpolasi linear kurva
   (fill rate, biaya) per SKU x model; SKU yang kurvanya tidak mencapai target dikeluarkan,
   bukan dipaksa.
9. **Statistik** — Wilcoxon signed-rank satu sisi berpasangan per SKU dengan koreksi Holm,
   uji Page untuk pola berurutan antar target layanan, kontras 90% vs 98%, dan bootstrap median
   berklaster (toko).
10. **Manifest SHA-256** — `verify` menghitung ulang seluruh alur dan membandingkan checksum;
    hanya kalau identik, regenerasi dinyatakan sah.

## Membuktikan regenerasi

```bash
python -m lumpy_bench regenerate      # tulis out/ + manifest.sha256.json
rm -rf out_bukti && python -m lumpy_bench regenerate --out out_bukti
python -m lumpy_bench verify         # bandingkan out_bukti dengan manifest rujukan
```

`verify` mengembalikan status `SAH` bila seluruh berkas identik byte-per-byte, dan menyebutkan
berkas mana yang berbeda bila tidak. Determinisme dijaga oleh: seed tunggal di `config.yaml`,
`random_state` pada Random Forest, penulisan CSV dengan pembulatan tetap dan urutan baris tetap.

## Memakai data asli

Letakkan berkas panjang (kolom wajib: `item_id`, `store_id`, `day`, `demand`, `sell_price`)
lalu jalankan:

```bash
python -m lumpy_bench regenerate --data /path/ke/data_panjang.csv --out out_asli
```

Berkas M5 (Kaggle) **tidak disertakan dan tidak boleh didistribusikan ulang** di repositori ini.
`lumpy_bench.data.muat_m5()` disediakan untuk pengguna yang mengunduh sendiri dari Kaggle dan
menyetujui ketentuannya; data contoh sintetis dipakai agar kerangka ini dapat diuji tanpa data
berlisensi.

## Struktur

```
lumpy_bench/
  synth.py        pembuat data contoh sintetis
  data.py         pemuatan data + periode aktif (+ loader M5 opsional)
  classify.py     ADI/CV2 dan penanda lumpy
  sampling.py     sampel berstrata
  features.py     fitur direct multi-horizon
  forecasters.py  Random Forest + Croston-SBA
  policy.py       simulasi order-up-to + penyetaraan tingkat layanan
  stats.py        Wilcoxon, Holm, Page, bootstrap klaster
  manifest.py     checksum SHA-256
  pipeline.py     orkestrasi alur penuh
  sweep.py        sapuan beberapa kombinasi parameter sekaligus
tests/            pytest: invarian simulator, uji statistik, determinisme, audit parameter
config.yaml       seluruh parameter (tidak ada nilai kebijakan yang di-hardcode)
sweep_contoh.yaml contoh berkas sapuan parameter
```

## Membuktikan angka klaim artikel (data M5)

Folder `verifikasi_klaim/` menjalankan **pipeline asli** yang menghasilkan angka artikel pada data
M5 milik Anda, lalu membandingkan setiap angka klaim dengan keluaran itu:

```bash
cd verifikasi_klaim
pip install -r requirements-verifikasi.txt
streamlit run app.py          # UI sederhana: pilih folder M5, tekan satu tombol
# atau
python jalankan.py --m5 ~/m5  # setelan artikel: 300 SKU, 4 titik evaluasi
```

Keluaran: tabel status per klaim (`cocok` / `tidak cocok` / `tidak dapat diverifikasi`) plus
`laporan_verifikasi.md` dan `.json`. Modul `lumpy_bench/` di repo ini adalah tulis ulang bersih
untuk dataset lain dan **bukan** jalur pembuktian angka artikel.

## Menyetel dan menyapu parameter

Semua nilai yang memengaruhi hasil ada di `config.yaml`; tidak ada yang perlu diedit di kode:

| Blok | Isi |
|---|---|
| `seed` | seed tunggal untuk sampling dan pemodelan |
| `lumpy` | ambang ADI/CV2 dan `min_nonzero` (filter minimum hari berpermintaan) |
| `features` | jendela `lag`, `rolling`, `rolling_std` |
| `sampling` | jumlah SKU sampel berstrata |
| `origin` | jumlah titik evaluasi + panjang jendela validasi dan uji |
| `rf` | hyperparameter Random Forest + `n_jobs` |
| `croston` | `alpha` (faktor debiasing selalu diturunkan sebagai `1 - alpha/2`) |
| `policy` | R, L, warmup, grid kuantil, target fill rate |
| `stats` | ambang signifikansi dan seed bootstrap |
| `output` | presisi pembulatan artefak (menentukan checksum manifest) |
| `cost_scenarios` | rasio biaya (Cs, Ch) yang disapu |

Audit otomatis menjaga ini: `tests/test_params.py` gagal bila ada kunci config yang tidak
dibaca kode, atau kode membaca kunci yang tidak ada di config.

Untuk menyapu beberapa kombinasi sekaligus:

```bash
cp sweep_contoh.yaml sweep.yaml   # sunting daftar cells
python -m lumpy_bench sweep --sweep sweep.yaml
```

Setiap sel dijalankan penuh ke `out_sweep/<nama>/` lengkap dengan manifest-nya, lalu satu baris
ringkasan per sel ditulis ke `out_sweep/ringkasan_sweep.csv` (H1/H2 didukung atau tidak,
p terkoreksi minimum, dan besaran efek median).

## Batasan

- Kerangka ini untuk **pengujian dan pembuktian alur**, bukan estimasi biaya absolut: biaya
  dinyatakan dalam rasio (Cs:Ch), bukan satuan uang.
- Uji berpasangan mengasumsikan SKU saling bebas; bootstrap berklaster hanya indikatif bila
  jumlah klaster (toko) sedikit.
- Kesetaraan tingkat layanan dicapai lewat interpolasi kurva kebijakan, sehingga ketelitiannya
  bergantung pada kerapatan grid tau.

## Sitasi

Lihat `CITATION.cff`. Lisensi kode: MIT (lihat `LICENSE`).
