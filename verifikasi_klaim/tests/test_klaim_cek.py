"""Uji evaluator klaim (cepat, tanpa data M5)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import klaim_cek  # noqa: E402


def _tulis(tmp: Path, nama: str, df: pd.DataFrame) -> Path:
    p = tmp / nama
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return p


def test_jumlah_baris(tmp_path: Path):
    _tulis(tmp_path, "a.csv", pd.DataFrame({"x": [1, 2, 3]}))
    k = {"id": "T1", "teks": "t", "berkas": "a.csv", "cek": "jumlah_baris", "nilai": 3}
    assert klaim_cek.periksa(k, tmp_path)["status"] == klaim_cek.STATUS_COCOK
    k["nilai"] = 4
    assert klaim_cek.periksa(k, tmp_path)["status"] == klaim_cek.STATUS_TIDAK_COCOK


def test_berkas_hilang_ditandai_belum_terverifikasi(tmp_path: Path):
    k = {"id": "T2", "teks": "t", "berkas": "tidak_ada.csv", "cek": "jumlah_baris", "nilai": 1}
    assert klaim_cek.periksa(k, tmp_path)["status"] == klaim_cek.STATUS_TIDAK_TERVERIFIKASI


def test_rentang_dan_toleransi(tmp_path: Path):
    _tulis(tmp_path, "b.csv", pd.DataFrame({"mae": [3.77, 3.96]}))
    k = {"id": "T3", "teks": "t", "berkas": "b.csv", "kolom": "mae", "cek": "rentang",
         "lo": 3.77, "hi": 3.96, "toleransi": 0.01}
    assert klaim_cek.periksa(k, tmp_path)["status"] == klaim_cek.STATUS_COCOK
    k["hi"] = 3.80
    assert klaim_cek.periksa(k, tmp_path)["status"] == klaim_cek.STATUS_TIDAK_COCOK


def test_semua_lebih_besar_dan_kecil(tmp_path: Path):
    _tulis(tmp_path, "c.csv", pd.DataFrame({"p": [0.5, 1.0], "d": [-0.2, -0.3]}))
    besar = {"id": "T4", "teks": "t", "berkas": "c.csv", "kolom": "p",
             "cek": "semua_lebih_besar", "nilai": 0.05}
    kecil = {"id": "T5", "teks": "t", "berkas": "c.csv", "kolom": "d",
             "cek": "semua_lebih_kecil", "nilai": 0.0}
    assert klaim_cek.periksa(besar, tmp_path)["status"] == klaim_cek.STATUS_COCOK
    assert klaim_cek.periksa(kecil, tmp_path)["status"] == klaim_cek.STATUS_COCOK


def test_jumlah_memenuhi(tmp_path: Path):
    _tulis(tmp_path, "d.csv", pd.DataFrame({"p": [0.01, 0.04, 0.3, 0.9]}))
    k = {"id": "T6", "teks": "t", "berkas": "d.csv", "kolom": "p", "cek": "jumlah_memenuhi",
         "op": "<", "nilai": 0.05, "harap": 2}
    assert klaim_cek.periksa(k, tmp_path)["status"] == klaim_cek.STATUS_COCOK
    k["harap"] = 3
    assert klaim_cek.periksa(k, tmp_path)["status"] == klaim_cek.STATUS_TIDAK_COCOK


def test_deret_nilai_diurutkan(tmp_path: Path):
    _tulis(tmp_path, "e.csv", pd.DataFrame({"titik": [2, 1], "n": [212, 197]}))
    k = {"id": "T7", "teks": "t", "berkas": "e.csv", "kolom": "n", "cek": "deret_nilai",
         "nilai": [197, 212], "toleransi": 0, "urut": "titik"}
    assert klaim_cek.periksa(k, tmp_path)["status"] == klaim_cek.STATUS_COCOK


def test_filter_dan_rf_lebih_rendah(tmp_path: Path):
    df = pd.DataFrame({"origin": [1, 1, 2, 2], "split": ["test"] * 4,
                       "metode": ["rf_global_direct", "croston_sba"] * 2,
                       "mae": [3.9, 4.5, 3.8, 4.4]})
    _tulis(tmp_path, "f.csv", df)
    k = {"id": "T8", "teks": "t", "berkas": "f.csv", "filter": {"split": "test"},
         "cek": "rf_lebih_rendah_di_semua_titik"}
    hasil = klaim_cek.periksa(k, tmp_path)
    assert hasil["status"] == klaim_cek.STATUS_COCOK and hasil["terukur"] == "2/2 titik"


def test_klaim_mode_penuh_tidak_dinilai_pada_mode_cepat(tmp_path: Path):
    klaim = [{"id": "T9", "teks": "t", "berkas": "a.csv", "cek": "jumlah_baris",
              "nilai": 3, "mode": "penuh"}]
    baris = klaim_cek.periksa_semua(klaim, tmp_path, mode="cepat")
    assert baris[0]["status"] == klaim_cek.STATUS_TIDAK_TERVERIFIKASI
    baris = klaim_cek.periksa_semua(klaim, tmp_path, mode="penuh")
    assert baris[0]["status"] == klaim_cek.STATUS_TIDAK_TERVERIFIKASI  # berkasnya memang tidak ada


def test_klaim_artikel_valid_dan_lengkap():
    path = Path(__file__).resolve().parents[1] / "klaim" / "klaim_artikel.json"
    isi = json.loads(path.read_text(encoding="utf-8"))
    klaim = isi["klaim"]
    assert len(klaim) >= 15
    id_set = [k["id"] for k in klaim]
    assert len(id_set) == len(set(id_set)), "id klaim tidak boleh duplikat"
    for k in klaim:
        assert k["cek"] in klaim_cek.JENIS_CEK
        assert k.get("mode") in {"penuh", "cepat"}
