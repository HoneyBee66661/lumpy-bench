#!/usr/bin/env bash
# kirim_ke_drive_v3.sh — unggah paket v3 ke folder Drive pengguna, dengan verifikasi md5 per berkas.
#
# Alat: skrip lama yang sudah terbukti (skills/devops/google-drive-uploads/scripts/drive_upload_file.py)
# — idempoten berdasarkan nama (PATCH bila sudah ada), parents hanya dikirim saat CREATE (kalau dikirim
# saat PATCH, Drive menolak 403 dan berkas lama tetap tidak berubah), dan memverifikasi md5 lokal vs
# md5Checksum Drive. `--no-share` dipakai supaya berkas riset TIDAK dibuka ke publik.
set -uo pipefail
export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"
REPO=/home/ubuntu/demand-forecast-rf-vs-croston
SKRIP=/home/ubuntu/.hermes/skills/devops/google-drive-uploads/scripts/drive_upload_file.py
FOLDER_ID=${FOLDER_ID:-1ELdgvtespJKl_l-YivIvNQp9Nk0ZRLAB}   # folder "Hermes File"
LOG="$REPO/results_v3/kirim_drive_v3.log"
OUT=/home/ubuntu

{
  echo "=== kirim ke Drive mulai $(date '+%Y-%m-%d %H:%M:%S') | folder $FOLDER_ID"
  for f in "$OUT"/v3_1_inti.zip "$OUT"/v3_2_tabel_utama.zip "$OUT"/v3_3_tabel_lain.zip \
           "$OUT"/v3_4a_prediksi_rasio_o12.zip "$OUT"/v3_4b_prediksi_rasio_o34.zip \
           "$OUT"/v3_5a_prediksi_ml_o12.zip "$OUT"/v3_5b_prediksi_ml_o34.zip \
           "$OUT"/v3_6_artefak_model.zip; do
    nama=$(basename "$f")
    echo "--- $nama ($(du -h "$f" | cut -f1), md5 $(md5sum "$f" | cut -c1-12)...)"
    python3 "$SKRIP" "$f" "$nama" --no-share --folder-id "$FOLDER_ID" 2>&1 | sed 's/^/    /'
    echo "    exit=$?"
  done
  echo "=== daftar isi folder setelah unggah ==="
  # catatan: skrip butuh argumen berkas walau hanya --list; /dev/null DITOLAK ("no such file"),
  # jadi dipakai berkas paket pertama yang memang ada
  python3 "$SKRIP" "$OUT/v3_1_inti.zip" --list --folder-id "$FOLDER_ID" 2>&1 | sed 's/^/  /' | head -20
  echo "=== selesai $(date '+%Y-%m-%d %H:%M:%S')"
} | tee "$LOG"
