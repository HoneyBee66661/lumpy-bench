"""Antarmuka baris perintah: synth-data, regenerate, verify."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from . import manifest, pipeline, synth


def _cfg(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lumpy-bench", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth-data", help="buat data contoh sintetis")
    s.add_argument("--out", default="data_contoh/sample_retail_long.csv")
    s.add_argument("--seed", type=int, default=20261007)
    s.add_argument("--n-sku", type=int, default=150)
    s.add_argument("--n-hari", type=int, default=900)

    r = sub.add_parser("regenerate", help="jalankan alur penuh dan tulis artefak + manifest")
    r.add_argument("--config", default="config.yaml")
    r.add_argument("--data", default=None, help="berkas panjang; kosong = pakai data_contoh")
    r.add_argument("--out", default="out")

    v = sub.add_parser("verify", help="hitung ulang lalu bandingkan dengan manifest")
    v.add_argument("--config", default="config.yaml")
    v.add_argument("--data", default=None)
    v.add_argument("--out", default="out")

    a = ap.parse_args(argv)
    if a.cmd == "synth-data":
        df = synth.buat_data_contoh(n_sku=a.n_sku, n_hari=a.n_hari, seed=a.seed)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        synth.simpan(a.out, df)
        print(f"data contoh: {a.out} | {len(df):,} baris | {df['sku_id'].nunique()} deret")
        return 0

    cfg = _cfg(Path(a.config))
    sumber = Path(a.data) if a.data else Path(cfg["paths"]["data_contoh"])
    out = Path(a.out)
    if a.cmd == "regenerate":
        if not sumber.exists():
            print(f"berkas data tidak ada ({sumber}); jalankan `synth-data` lebih dahulu")
            return 2
        pipeline.jalankan(cfg, sumber, out)
        return 0

    # verify: hitung ulang di folder sementara, bandingkan checksum
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp_out = Path(tmp) / "out"
        pipeline.jalankan(cfg, sumber, tmp_out, verbose=False)
        hasil = manifest.bandingkan_manifest(tmp_out)
        rujukan = out / "manifest.sha256.json"
        if rujukan.exists():
            import json
            lama = json.loads(rujukan.read_text())
            baru = json.loads((tmp_out / "manifest.sha256.json").read_text())
            beda = sorted(k for k in set(lama) & set(baru) if lama[k] != baru[k])
            hilang = sorted(set(lama) - set(baru))
            status = "SAH" if not beda and not hilang else "TIDAK IDENTIK"
            print(f"verify: {len(baru)} artefak | {status}")
            for k in beda:
                print(f"   beda  : {k}")
            for k in hilang:
                print(f"   hilang: {k}")
            return 0 if status == "SAH" else 1
        print("manifest rujukan belum ada; jalankan `regenerate` lebih dahulu")
        return 2


if __name__ == "__main__":
    sys.exit(main())
