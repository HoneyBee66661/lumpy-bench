"""app.py — UI sederhana (Streamlit) untuk membuktikan angka klaim artikel pada data M5 Anda.

Jalankan:
    pip install -r requirements-verifikasi.txt
    streamlit run app.py

Alurnya: pilih folder M5 -> tekan satu tombol -> pipeline asli berjalan -> tabel klaim muncul
dengan status per klaim, plus laporan yang bisa diunduh.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

BUNDLE = Path(__file__).resolve().parent
BERKAS_M5 = ["sales_train_evaluation.csv", "sell_prices.csv", "calendar.csv"]
STATUS_IKON = {"cocok": "✅", "tidak cocok": "❌", "tidak dapat diverifikasi": "➖"}

st.set_page_config(page_title="Verifikasi klaim artikel — RF vs Croston-SBA", page_icon="📊",
                   layout="wide")

st.title("Verifikasi angka klaim artikel pada data M5 Anda")
st.markdown(
    "Halaman ini menjalankan **pipeline asli** yang memakai data M5 (Random Forest global vs "
    "Croston-SBA) lalu membandingkan keluarannya dengan angka yang diklaim di artikel. "
    "Anda hanya perlu menunjuk folder berisi berkas M5; sisanya otomatis."
)

with st.sidebar:
    st.header("Masukan")
    m5_dir = st.text_input("Folder M5", value=str(Path.home() / "m5"),
                           help="Harus berisi sales_train_evaluation.csv, sell_prices.csv, calendar.csv")
    mode = st.radio("Mode", options=["penuh", "cepat"], index=0,
                    help="penuh = setelan artikel (300 SKU, 4 titik). cepat = subset untuk uji coba.")
    with st.expander("Setelan lanjutan"):
        n_sku = st.number_input("Jumlah SKU sampel", min_value=3, max_value=3000,
                                value=300 if mode == "penuh" else 30, step=10)
        origins = st.multiselect("Titik evaluasi", options=[1, 2, 3, 4],
                                 default=[1, 2, 3, 4] if mode == "penuh" else [4])
        abaikan_sha = st.checkbox("Lewati pemeriksaan SHA-256", value=False,
                                  help="Hanya bila data Anda memang bukan M5 resmi.")
        paksa = st.checkbox("Ulangi semua tahap (abaikan hasil yang ada)", value=False)
    tombol = st.button("Jalankan verifikasi", type="primary", use_container_width=True)

    st.divider()
    st.caption("Perkiraan waktu: mode penuh bisa berjam-jam bergantung perangkat; mode cepat "
               "umumnya di bawah 15 menit setelah berkas M5 terbaca.")

folder = Path(m5_dir).expanduser()
if not st.session_state.get("berjalan"):
    ada = [b for b in BERKAS_M5 if (folder / b).exists()]
    if folder.exists():
        st.info(f"Berkas M5 terdeteksi: {len(ada)} dari {len(BERKAS_M5)} "
                f"({', '.join(ada) if ada else 'belum ada'})")
    else:
        st.warning("Folder belum ada. Unduh M5 dari Kaggle lalu isi jalur di panel kiri.")

if tombol:
    st.session_state["berjalan"] = True
    perintah = [sys.executable, str(BUNDLE / "jalankan.py"), "--m5", str(folder),
                "--mode", mode, "--n-sku", str(int(n_sku)),
                "--origins", *[str(o) for o in (origins or [4])]]
    if abaikan_sha:
        perintah.append("--abaikan-sha")
    if paksa:
        perintah.append("--paksa")

    st.subheader("Keluaran proses")
    kotak = st.empty()
    baris_log: list[str] = []
    t0 = time.time()
    proses = subprocess.Popen(perintah, cwd=str(BUNDLE), stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1)
    for baris in proses.stdout:                      # type: ignore[union-attr]
        baris_log.append(baris.rstrip())
        kotak.code("\n".join(baris_log[-25:]), language="text")
    proses.wait()
    durasi = time.time() - t0
    st.caption(f"Selesai dalam {durasi:.0f} detik (exit {proses.returncode}).")

    laporan = BUNDLE / "kerja" / "laporan_verifikasi.json"
    if laporan.exists():
        isi = json.loads(laporan.read_text())
        ring = isi["ringkasan"]
        kol = st.columns(3)
        kol[0].metric("Cocok", ring["cocok"])
        kol[1].metric("Tidak cocok", ring["tidak_cocok"])
        kol[2].metric("Belum terverifikasi", ring["belum_terverifikasi"])

        tabel = pd.DataFrame([{
            "Status": f"{STATUS_IKON.get(b['status'], '')} {b['status']}",
            "ID": b["id"],
            "Klaim": b["teks"],
            "Bukti terukur": b.get("pesan", ""),
        } for b in isi["klaim"]])
        st.dataframe(tabel, use_container_width=True, hide_index=True)

        st.download_button("Unduh laporan (Markdown)",
                           (BUNDLE / "kerja" / "laporan_verifikasi.md").read_text(),
                           file_name="laporan_verifikasi.md")
        st.download_button("Unduh laporan (JSON)", laporan.read_text(),
                           file_name="laporan_verifikasi.json")
        if ring["tidak_cocok"] == 0 and ring["belum_terverifikasi"] == 0:
            st.success("Semua klaim yang dapat diuji pada mode ini cocok dengan keluaran pipeline.")
        else:
            st.warning("Ada klaim yang tidak cocok atau belum dapat diverifikasi — lihat tabel.")
    else:
        st.error("Laporan belum terbentuk. Periksa keluaran proses di atas.")
    st.session_state["berjalan"] = False
else:
    st.subheader("Klaim yang akan diperiksa")
    klaim = json.loads((BUNDLE / "klaim" / "klaim_artikel.json").read_text())["klaim"]
    st.dataframe(pd.DataFrame([{"ID": k["id"], "Klaim": k["teks"],
                                "Wajib mode": k.get("mode", "penuh")} for k in klaim]),
                 use_container_width=True, hide_index=True)
