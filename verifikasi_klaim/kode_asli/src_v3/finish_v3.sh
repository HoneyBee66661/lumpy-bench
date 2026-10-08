#!/usr/bin/env bash
# finish_v3.sh — menyelesaikan v3 (dipakai cron no_agent DAN bisa dijalankan manual / background).
#
# BATASAN YANG DITEMUKAN: cron mematikan skrip setelah 3600 s ("Script timed out after 3600s",
# job 4ad8b4aa6070). Karena itu skrip ini RESUMABLE:
#   * menunggu kernel dengan anggaran TUNGGU_MENIT (default 15) -> kalau belum selesai, keluar 0 diam
#   * tiap langkah yang selesai menandai dirinya (results_v3/.done_<nama>) dan dilewati pada run berikut
#   * sebelum tiap langkah ada cek anggaran waktu: berhenti rapi supaya tidak dibunuh di tengah langkah
#   * flock: cron + background/manual tidak mungkin menulis bersamaan
#
# Aturan keras: paket hanya dikirim kalau SEMUA verifikasi lulus; kalau tidak, kirim laporan gagal.
set -uo pipefail

REPO=/home/ubuntu/demand-forecast-rf-vs-croston
PY="$REPO/.venv/bin/python"
RES="$REPO/results_v3"
LOG="$RES/finish_v3.log"
DL=/tmp/v3dl
KERNELS=(rf-vs-croston-v3-rate rf-vs-croston-v3-ml rf-vs-croston-v3-h)
WAJIB=(rf-vs-croston-v3-rate rf-vs-croston-v3-ml)     # wajib COMPLETE agar pipeline bisa jalan
OWNER=honeybee66661
# jalur absolut: cron tidak mewarisi PATH shell interaktif
export PATH="/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
KAGGLE=/home/ubuntu/.local/bin/kaggle
HERMES=/home/ubuntu/.local/bin/hermes
UNZIP=/usr/bin/unzip
TUNGGU_MENIT=${TUNGGU_MENIT:-15}      # anggaran menunggu kernel per run
ANGGARAN_MENIT=${ANGGARAN_MENIT:-40}  # anggaran kerja per run (batas cron 60 menit, sisa untuk tunggu)

mkdir -p "$RES" "$DL"
exec > >(tee -a "$LOG") 2>&1
T0=$(date +%s)
lewat() { [ $(( ($(date +%s) - T0) / 60 )) -ge "$1" ]; }

exec 9>/tmp/finish_v3.lock
if ! flock -n 9; then
  echo "[kunci] proses finish_v3 lain sedang berjalan — keluar tanpa mengubah apa pun"
  exit 0
fi

kirim() {
  "$HERMES" send --to telegram "$1" >/dev/null 2>&1 && echo "[kirim] terkirim" || echo "[kirim] GAGAL mengirim"
}
gagal() {
  echo "[GAGAL] $1"
  kirim "v3 GAGAL: $1
Log: $LOG"
  exit 1
}

echo "=============================================================="
echo "finish_v3 mulai $(date '+%Y-%m-%d %H:%M:%S') | tunggu ${TUNGGU_MENIT}m | anggaran ${ANGGARAN_MENIT}m"
echo "=============================================================="
cd "$REPO" || gagal "repo tidak ditemukan: $REPO"

if [ -f "$RES/.done_all" ]; then
  echo "[selesai] penanda .done_all ada — tidak ada yang dikerjakan"
  exit 0
fi

# ---------------------------------------------------------------- 1) tunggu kernel
selesai=0
for i in $(seq 1 "$TUNGGU_MENIT"); do
  kurang=0
  for k in "${KERNELS[@]}"; do
    s=$(timeout 120 "$KAGGLE" kernels status "$OWNER/$k" 2>&1 | tail -1)
    echo "$(date +%H:%M:%S) $k: $s"
    case "$s" in *RUNNING*|*QUEUED*) kurang=$((kurang + 1));; esac
  done
  if [ "$kurang" -eq 0 ]; then selesai=1; break; fi
  sleep 60
done
if [ "$selesai" -ne 1 ]; then
  echo "[tunggu] masih ada kernel berjalan setelah ${TUNGGU_MENIT} menit — keluar rapi, run berikutnya melanjutkan"
  exit 0
fi

# ---------------------------------------------------------------- 2) status wajib
for k in "${WAJIB[@]}"; do
  s=$(timeout 120 "$KAGGLE" kernels status "$OWNER/$k" 2>&1 | tail -1)
  case "$s" in
    *COMPLETE*) echo "[ok] $k COMPLETE" ;;
    *) gagal "kernel wajib $k berstatus: $s" ;;
  esac
done
s=$(timeout 120 "$KAGGLE" kernels status "$OWNER/rf-vs-croston-v3-h" 2>&1 | tail -1)
case "$s" in
  *COMPLETE*) echo "[ok] kernel H COMPLETE" ;;
  *) echo "[peringatan] kernel H tidak COMPLETE ($s) — varian L dilaporkan sebagai batasan" ;;
esac

# ---------------------------------------------------------------- 3) tarik keluaran + bersihkan (sekali)
if [ ! -f "$RES/.downloaded_full" ]; then
  # BERSIHKAN DULU (sebelum menyalin apa pun): sisa uji skala kecil tidak boleh bercampur dengan hasil
  # 300 SKU. Urutan ini penting — versi sebelumnya menghapus artifacts/* SESUDAH menyalinnya sehingga
  # artefak model hilang (bug nyata yang ketangkap karena verifikasi gagal).
  rm -f "$RES"/tables/policy_sweep_o*.csv.gz "$RES"/tables/costs_o*.csv.gz \
        "$RES"/tables/service_equalized_o*.csv.gz "$RES"/tables/ss_diagnostik_o*.csv \
        "$RES"/tables/stats_*.csv "$RES"/tables/metrics_*.csv "$RES"/tables/sample_meta_v3.csv \
        "$RES"/tables/sensitivitas_v3.csv.gz "$RES"/interim/err_val_*.csv.gz \
        "$RES"/figures/*.png "$RES"/run_log_v3.txt "$RES"/article_facts_v3.txt \
        "$RES"/manifest_v3.txt "$RES"/LAPORAN_PERUBAHAN_v3.md "$RES"/README_paket_v3.md \
        "$RES"/paket_v3_isi.txt "$RES"/artifacts/*
  # masukan baru -> keluaran turunan lama tidak berlaku; penanda langkah dicabut supaya dihitung ulang
  rm -f "$RES"/.done_policy "$RES"/.done_stats "$RES"/.done_sens "$RES"/.done_facts \
        "$RES"/.done_verify "$RES"/.done_gambar "$RES"/.done_runlog "$RES"/.done_laporan \
        "$RES"/.done_paket "$RES"/.done_all
  echo "[bersih] sisa keluaran skala kecil dihapus + penanda langkah dicabut"

  for k in "${KERNELS[@]}"; do
    mkdir -p "$DL/$k"
    timeout 900 "$KAGGLE" kernels output "$OWNER/$k" -p "$DL/$k" >/dev/null 2>&1
    log=$(ls "$DL/$k"/*.log 2>/dev/null | head -1)
    [ -n "$log" ] && cp "$log" "$RES/log_kernel_$k.txt"
    cp "$DL/$k"/pred_*.csv.gz "$RES/interim/" 2>/dev/null
    for z in "$DL/$k"/v3_*_out.zip; do
      [ -f "$z" ] && (cd "$RES/interim" && "$UNZIP" -o -q "$z" 'pred_*' 'forecast_log.txt' 'artifacts/*' 2>/dev/null)
    done
    echo "[tarik] $k -> $(ls "$RES/interim"/pred_*.csv.gz 2>/dev/null | wc -l) berkas prediksi di interim/"
  done
  rm -rf /tmp/v3art && mkdir -p /tmp/v3art "$RES/artifacts"
  for z in "$DL"/*/v3_*_out.zip; do
    [ -f "$z" ] && "$UNZIP" -o -q "$z" -d /tmp/v3art 2>/dev/null
  done
  cp /tmp/v3art/artifacts/* "$RES/artifacts/" 2>/dev/null
  cp /tmp/v3art/forecast_manifest_*.csv "$RES/" 2>/dev/null
  echo "[artefak] $(ls "$RES/artifacts" 2>/dev/null | wc -l) berkas di results_v3/artifacts/"

  n_pred=$(ls "$RES/interim"/pred_*.csv.gz 2>/dev/null | wc -l)
  [ "$n_pred" -ge 20 ] || gagal "hanya $n_pred berkas prediksi tertarik — keluaran kernel tidak lengkap"
  touch "$RES/.downloaded_full"
fi

# ---------------------------------------------------------------- 4) langkah-langkah idempoten
langkah() {  # langkah <nama> <fungsi...>
  local nama=$1; shift
  if [ -f "$RES/.done_$nama" ]; then echo "[lewati] $nama sudah selesai"; return 0; fi
  if lewat "$ANGGARAN_MENIT"; then echo "[anggaran] waktu habis sebelum $nama — keluar rapi"; exit 0; fi
  echo "--- langkah: $nama ($(date +%H:%M:%S)) ---"
  "$@" || gagal "langkah $nama gagal"
  touch "$RES/.done_$nama"
}

s_policy()  { "$PY" src_v3/run_all_v3.py --stages policy --n-sku 300; }
s_stats()   { "$PY" src_v3/run_all_v3.py --stages stats  --n-sku 300; }
s_sens()    { "$PY" src_v3/run_all_v3.py --stages sens   --n-sku 300; }
s_facts()   { "$PY" src_v3/run_all_v3.py --stages facts  --n-sku 300; }
s_verify()  {
  "$PY" tests/test_simulator_v3.py || return 1
  "$PY" tests/test_features_v3.py || return 1
  "$PY" tests/test_stats_v3.py || return 1
  "$PY" tests/test_pred_name_v3.py || return 1
  "$PY" tests/verify_v3_forecasts.py --n-sku 6 --n-titik 12 || return 1
  "$PY" tests/verify_v3_policy.py --n-sku 6 --n-tau 2 || return 1
}
s_gambar()  { "$PY" src_v3/make_figures_v3.py --origin 4; }
s_runlog()  { "$PY" src_v3/make_runlog_v3.py; }
s_laporan() { "$PY" src_v3/make_report_v3.py; }
s_paket()   {
  # Paket dipecah < 45 MB (batas dokumen Telegram 50 MB) dan DIKIRIM dengan verifikasi message id,
  # bukan lewat satu zip besar: kiriman 98 MB sebelumnya ditolak Telegram sementara exit code 0.
  "$PY" src_v3/package_v3_split.py --out-dir /home/ubuntu || return 1
  bash src_v3/kirim_paket_v3.sh || return 1
}

langkah policy  s_policy
langkah stats   s_stats
langkah sens    s_sens
langkah facts   s_facts
langkah verify  s_verify
langkah gambar  s_gambar
langkah runlog  s_runlog
langkah laporan s_laporan
langkah paket   s_paket

# ---------------------------------------------------------------- 5) penutup
git add -A && git commit -q -m "v3 SELESAI (otomatis): hasil penuh 300 SKU x 4 origin + verifikasi lulus" \
  || echo "[git] tidak ada perubahan untuk di-commit"
echo "paket sudah dikirim dengan verifikasi message id (lihat results_v3/kirim_paket_v3.jsonl)"
touch "$RES/.done_all"
echo "=============================================================="
echo "finish_v3 selesai $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="
