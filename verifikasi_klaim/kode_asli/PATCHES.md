# Catatan perubahan pada salinan kode pipeline

Salinan di folder ini identik dengan repo kerja **kecuali** satu pengaman berikut, yang
ditambahkan agar verifikasi tetap jalan pada skala kecil. Perubahan ini hanya menyentuh
blok diagnostik (cetak), bukan perhitungan angka:

## `src_v3/cek_hol_mrel_rev8.py`

Blok "kenapa biaya kosong pada baris tercakup" mengasumsikan ada minimal satu SKU yang baris
biayanya kosong (pada setelan artikel: 52 baris dari 300 SKU). Pada verifikasi berskala kecil
jumlah itu nol, sehingga `kosong[0]` melempar `IndexError` dan skrip berhenti sebelum menulis
tabel H1/H2. Perbaikannya: seluruh blok tersebut dibungkus `if len(kosong):`.

Tidak ada baris lain yang diubah; perhitungan H1, H2, dan p-value tetap sama persis.
