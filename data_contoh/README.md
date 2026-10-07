# Data contoh

`sample_retail_long.csv.gz` adalah **data sintetis** hasil `python -m lumpy_bench synth-data`
(seed 20261007): 150 deret item-toko x 900 hari, memuat campuran pola lancar, intermiten, dan
lumpy, dengan periode aktif (kolom `sell_price`) yang tidak selalu penuh.

Data ini tidak berasal dari data penjualan nyata dan tidak memuat data kompetisi mana pun,
sehingga bebas didistribusikan. Skemanya sengaja disamakan dengan data ritel nyata:

| kolom | arti |
|---|---|
| item_id, store_id, sku_id | identitas deret item-toko |
| day, date | indeks dan tanggal harian |
| demand | permintaan unit harian |
| sell_price | harga jual (kosong = hari di luar periode aktif) |
| snap, event_name_1 | penanda kalender |
| cat_id, state_id | kategori dan wilayah (untuk stratifikasi) |

Untuk memakai data asli, jalankan `python -m lumpy_bench regenerate --data <berkas.csv>`.
Berkas M5 dari Kaggle tidak boleh disertakan di repositori ini.
