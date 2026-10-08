#!/usr/bin/env bash
# kirim_paket_v3.sh — kirim paket v3 yang sudah dipecah ke Telegram, dengan VERIFIKASI message id.
#
# Pelajaran: paket 98 MB melewati batas dokumen Bot API Telegram (50 MB) sementara `hermes send`
# tetap keluar status 0 — jadi ukuran diperiksa dulu dan hasil kiriman DIREKAM (JSON penuh, termasuk
# message_id) ke results_v3/kirim_paket_v3.jsonl supaya bisa diverifikasi, bukan diasumsikan.
set -uo pipefail
export PATH="/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
HERMES=/home/ubuntu/.local/bin/hermes
REPO=/home/ubuntu/demand-forecast-rf-vs-croston
OUT=/home/ubuntu
REKAM="$REPO/results_v3/kirim_paket_v3.jsonl"

kirim_satu() {  # kirim_satu <berkas> <judul>
  local f="$1" judul="$2"
  local mb id
  mb=$(du -m "$f" | cut -f1)
  if [ "$mb" -ge 45 ]; then
    echo "[lewat] $(basename "$f") ${mb} MB (batas 45 MB)"
    return 1
  fi
  local json
  json=$("$HERMES" send --to telegram --json "MEDIA:$f

$judul
$(sha256sum "$f" | cut -c1-16)... | $(du -h "$f" | cut -f1)")
  id=$(echo "$json" | tr -d '\n' | sed -n 's/.*"message_id": "\([0-9]*\)".*/\1/p')
  if [ -n "$id" ]; then
    echo "[ok] $(basename "$f") -> message_id $id"
    echo "{\"berkas\":\"$(basename "$f")\",\"mb\":$mb,\"message_id\":\"$id\",\"sha256_16\":\"$(sha256sum "$f" | cut -c1-16)\"}" >> "$REKAM"
  else
    echo "[GAGAL] $(basename "$f") -> $json"
    echo "{\"berkas\":\"$(basename "$f")\",\"mb\":$mb,\"message_id\":null,\"hasil\":$(echo "$json" | tr -d '\n')}" >> "$REKAM"
  fi
}

: > "$REKAM"
kirim_satu "$OUT/v3_1_inti.zip" "Paket v3 • 1/8 — INTI: kode (src_v3, tests, kaggle), config, README paket, LAPORAN_PERUBAHAN_v3.md (claim ledger + vonis H1/H2 + keterbatasan), article_facts_v3.txt, run_log_v3, manifest, gambar, tabel statistik & metrik ringkas, importance fitur, peta identitas."
kirim_satu "$OUT/v3_2_tabel_utama.zip" "Paket v3 • 2/8 — TABEL ORIGIN 4 (split resmi): costs_o4 (biaya per SKU x model x tau x skenario), policy_sweep_o4, service_equalized_o4, sensitivitas_v3."
kirim_satu "$OUT/v3_3_tabel_lain.zip" "Paket v3 • 3/8 — TABEL ORIGIN 1-3 (robustness): costs/policy_sweep/service_equalized/ss_diagnostik."
kirim_satu "$OUT/v3_4a_prediksi_rasio_o12.zip" "Paket v3 • 4/8 — PREDIKSI MENTAH model rasio, origin 1-2 (naive, MA7/28, SES, Croston classic/SBA/optimized, TSB, ADIDA, IMAPA)."
kirim_satu "$OUT/v3_4b_prediksi_rasio_o34.zip" "Paket v3 • 5/8 — PREDIKSI MENTAH model rasio, origin 3-4 + varian horizon H=4/14/15 (sensitivitas L/R)."
kirim_satu "$OUT/v3_5a_prediksi_ml_o12.zip" "Paket v3 • 6/8 — PREDIKSI MENTAH model ML, origin 1-2 (RF direct 3 seed) + error validasi."
kirim_satu "$OUT/v3_5b_prediksi_ml_o34.zip" "Paket v3 • 7/8 — PREDIKSI MENTAH model ML, origin 3-4 (RF direct 3 seed, RF recursive, LightGBM) + error validasi."
kirim_satu "$OUT/v3_6_artefak_model.zip" "Paket v3 • 8/8 — ARTEFAK MODEL RF global (.joblib, dilatih scikit-learn 1.6.1) untuk reproduksi & inspeksi."

echo "--- rekaman kiriman ---"
cat "$REKAM"
