"""Evaluator klaim: bandingkan artefak hasil jalan dengan tabel klaim artikel.

Setiap klaim di `klaim/klaim_artikel.json` menunjuk satu berkas keluaran pipeline dan satu
jenis pemeriksaan. Modul ini sengaja kecil dan tanpa efek samping supaya bisa diuji cepat di CI.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

JENIS_CEK = (
    "jumlah_baris",
    "rentang",
    "semua_lebih_besar",
    "semua_lebih_kecil",
    "semua_benar",
    "jumlah_memenuhi",
    "deret_nilai",
    "rf_lebih_rendah_di_semua_titik",
)

STATUS_COCOK = "cocok"
STATUS_TIDAK_COCOK = "tidak cocok"
STATUS_TIDAK_TERVERIFIKASI = "tidak dapat diverifikasi"


def muat_klaim(path: Path) -> list[dict]:
    isi = json.loads(Path(path).read_text(encoding="utf-8"))
    klaim = isi.get("klaim") or []
    for k in klaim:
        if k["cek"] not in JENIS_CEK:
            raise ValueError(f"klaim {k.get('id')}: jenis cek tidak dikenal ({k['cek']})")
    return klaim


def _baca(hasil_dir: Path, berkas: str, klaim: dict) -> pd.DataFrame | None:
    p = Path(hasil_dir) / berkas
    if not p.exists():
        return None
    df = pd.read_csv(p)
    saring = klaim.get("filter") or {}
    for kolom, nilai in saring.items():
        if kolom not in df.columns:
            return None
        df = df[df[kolom] == nilai]
    return df


def _angka(x) -> float:
    return round(float(x), 10)


def periksa(klaim: dict, hasil_dir: Path) -> dict:
    """Kembalikan {status, terukur, pesan} untuk satu klaim."""
    df = _baca(hasil_dir, klaim["berkas"], klaim)
    if df is None:
        return {"status": STATUS_TIDAK_TERVERIFIKASI,
                "terukur": None,
                "pesan": f"berkas keluaran belum ada: {klaim['berkas']}"}
    if df.empty:
        return {"status": STATUS_TIDAK_TERVERIFIKASI, "terukur": None,
                "pesan": "berkas ada tetapi tidak berisi baris yang cocok dengan filter"}

    jenis = klaim["cek"]
    kolom = klaim.get("kolom")
    tol = float(klaim.get("toleransi", 0.0))

    if jenis == "jumlah_baris":
        terukur = int(len(df))
        harap = int(klaim["nilai"])
        ok = abs(terukur - harap) <= tol
        return {"status": STATUS_COCOK if ok else STATUS_TIDAK_COCOK, "terukur": terukur,
                "pesan": f"{terukur} baris (klaim {harap})"}

    if jenis == "rf_lebih_rendah_di_semua_titik":
        if not {"metode", "origin", "mae"}.issubset(df.columns):
            return {"status": STATUS_TIDAK_TERVERIFIKASI, "terukur": None,
                    "pesan": "kolom metode/origin/mae tidak lengkap"}
        pil = df.pivot_table(index="origin", columns="metode", values="mae", aggfunc="first")
        if not {"rf_global_direct", "croston_sba"}.issubset(pil.columns):
            return {"status": STATUS_TIDAK_TERVERIFIKASI, "terukur": None,
                    "pesan": "kedua metode tidak ada di tabel akurasi"}
        menang = int((pil["rf_global_direct"] < pil["croston_sba"]).sum())
        ok = menang == len(pil)
        return {"status": STATUS_COCOK if ok else STATUS_TIDAK_COCOK,
                "terukur": f"{menang}/{len(pil)} titik",
                "pesan": f"RF lebih rendah di {menang} dari {len(pil)} titik"}

    if kolom not in df.columns:
        return {"status": STATUS_TIDAK_TERVERIFIKASI, "terukur": None,
                "pesan": f"kolom '{kolom}' tidak ada di {klaim['berkas']}"}
    s = pd.to_numeric(df[kolom], errors="coerce")

    if jenis == "semua_benar":
        ok = bool(df[kolom].astype(bool).all())
        return {"status": STATUS_COCOK if ok else STATUS_TIDAK_COCOK,
                "terukur": f"{int(df[kolom].astype(bool).sum())}/{len(df)} benar",
                "pesan": f"{int(df[kolom].astype(bool).sum())} dari {len(df)} baris bernilai benar"}

    if jenis == "rentang":
        lo, hi = float(klaim["lo"]), float(klaim["hi"])
        mn, mx = _angka(s.min()), _angka(s.max())
        ok = (mn >= lo - tol) and (mx <= hi + tol)
        return {"status": STATUS_COCOK if ok else STATUS_TIDAK_COCOK,
                "terukur": [mn, mx], "pesan": f"rentang terukur {mn} - {mx} (klaim {lo} - {hi})"}

    if jenis in ("semua_lebih_besar", "semua_lebih_kecil"):
        batas = float(klaim["nilai"])
        putusan = (s > batas) if jenis == "semua_lebih_besar" else (s < batas)
        putusan = putusan | (s - batas).abs().le(tol)
        ok = bool(putusan.all())
        tanda = ">" if jenis == "semua_lebih_besar" else "<"
        return {"status": STATUS_COCOK if ok else STATUS_TIDAK_COCOK,
                "terukur": f"{int(putusan.sum())}/{len(s)} baris memenuhi {tanda} {batas}",
                "pesan": (f"{int(putusan.sum())} dari {len(s)} nilai memenuhi {tanda} {batas}; "
                          f"nilai ekstrem {_angka(s.min())} - {_angka(s.max())}")}

    if jenis == "jumlah_memenuhi":
        op, batas = klaim["op"], float(klaim["nilai"])
        putusan = {"<": s < batas, "<=": s <= batas, ">": s > batas, ">=": s >= batas,
                   "==": s.eq(batas)}[op]
        putusan = putusan | (s - batas).abs().le(tol)
        jumlah = int(putusan.sum())
        ok = jumlah == int(klaim["harap"])
        return {"status": STATUS_COCOK if ok else STATUS_TIDAK_COCOK, "terukur": jumlah,
                "pesan": f"{jumlah} baris memenuhi {kolom} {op} {batas} (klaim {klaim['harap']})"}

    if jenis == "deret_nilai":
        urut = klaim.get("urut")
        if urut and urut in df.columns:
            df = df.sort_values(urut)
            s = pd.to_numeric(df[kolom], errors="coerce")
        harap = [float(x) for x in klaim["nilai"]]
        terukur = [_angka(v) for v in s.tolist()]
        if len(terukur) != len(harap):
            return {"status": STATUS_TIDAK_COCOK, "terukur": terukur,
                    "pesan": f"panjang deret terukur {len(terukur)} != klaim {len(harap)}"}
        beda = [abs(a - b) for a, b in zip(terukur, harap)]
        ok = max(beda) <= max(tol, 0.0)
        return {"status": STATUS_COCOK if ok else STATUS_TIDAK_COCOK, "terukur": terukur,
                "pesan": f"selisih maksimum {max(beda):.3g} terhadap klaim (toleransi {tol:g})"}

    raise ValueError(f"jenis cek tidak tertangani: {jenis}")   # pragma: no cover


def periksa_semua(klaim: list[dict], hasil_dir: Path, mode: str = "penuh") -> list[dict]:
    """Jalankan semua klaim. Klaim ber-mode 'penuh' ditandai belum terverifikasi bila mode cepat."""
    baris = []
    for k in klaim:
        if mode != "penuh" and k.get("mode") == "penuh":
            baris.append({**k, "status": STATUS_TIDAK_TERVERIFIKASI, "terukur": None,
                          "pesan": "klaim hanya sah diverifikasi pada mode penuh (300 SKU, 4 titik)"})
            continue
        hasil = periksa(k, Path(hasil_dir))
        baris.append({**k, **hasil})
    return baris


def ringkas(baris: list[dict]) -> dict:
    jumlah = {s: sum(1 for b in baris if b["status"] == s)
              for s in (STATUS_COCOK, STATUS_TIDAK_COCOK, STATUS_TIDAK_TERVERIFIKASI)}
    return {"total": len(baris), "cocok": jumlah[STATUS_COCOK],
            "tidak_cocok": jumlah[STATUS_TIDAK_COCOK],
            "belum_terverifikasi": jumlah[STATUS_TIDAK_TERVERIFIKASI]}


def laporan_markdown(baris: list[dict], konteks: dict) -> str:
    s = ringkas(baris)
    baris_md = [
        "# Laporan verifikasi angka klaim",
        "",
        f"- Mode: **{konteks.get('mode')}** | SKU: {konteks.get('n_sku')} | "
        f"titik evaluasi: {konteks.get('origins')}",
        f"- Data M5: `{konteks.get('m5')}` (SHA-256 {'terverifikasi' if konteks.get('sha_ok') else 'TIDAK diverifikasi'})",
        f"- Hasil: **{s['cocok']} cocok**, {s['tidak_cocok']} tidak cocok, "
        f"{s['belum_terverifikasi']} belum dapat diverifikasi (dari {s['total']} klaim)",
        "",
        "| Klaim | Isi | Status | Bukti terukur |",
        "|---|---|---|---|",
    ]
    for b in baris:
        baris_md.append(f"| {b['id']} | {b['teks']} | {b['status']} | {b.get('pesan', '')} |")
    baris_md += [
        "",
        "Artefak mentah hasil jalan ini ada di folder kerja (`results_v3/`), termasuk manifest "
        "SHA-256 bila stage `facts` dijalankan.",
        "",
        "Catatan: kecocokan angka berlaku untuk lingkungan dengan versi pustaka yang sama "
        "(`kode_asli/requirements.lock.txt`). Perbedaan versi atau jumlah thread dapat menggeser "
        "angka di digit terakhir.",
    ]
    return "\n".join(baris_md) + "\n"
