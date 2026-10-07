"""Manifest SHA-256 untuk membuktikan regenerasi."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blok in iter(lambda: fh.read(1 << 20), b""):
            h.update(blok)
    return h.hexdigest()


def tulis_manifest(folder: Path, nama: str = "manifest.sha256.json") -> Path:
    berkas = sorted(p for p in Path(folder).rglob("*") if p.is_file() and p.name != nama)
    isi = {str(p.relative_to(folder)): sha256(p) for p in berkas}
    out = Path(folder) / nama
    out.write_text(json.dumps(isi, indent=2, sort_keys=True) + "\n")
    return out


def bandingkan_manifest(folder: Path, nama: str = "manifest.sha256.json") -> dict:
    """Bandingkan isi folder dengan manifest. Mengembalikan ringkasan perbedaan."""
    folder = Path(folder)
    rujukan = json.loads((folder / nama).read_text())
    sekarang = {str(p.relative_to(folder)): sha256(p)
                for p in sorted(folder.rglob("*")) if p.is_file() and p.name != nama}
    hilang = sorted(set(rujukan) - set(sekarang))
    baru = sorted(set(sekarang) - set(rujukan))
    beda = sorted(k for k in set(rujukan) & set(sekarang) if rujukan[k] != sekarang[k])
    return {"berkas": len(sekarang), "identik": len(sekarang) - len(beda) - len(baru),
            "beda": beda, "hilang": hilang, "baru": baru,
            "sah": not beda and not hilang}
