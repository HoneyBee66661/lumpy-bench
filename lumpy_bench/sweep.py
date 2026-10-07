"""Sapuan parameter: jalankan alur penuh untuk beberapa kombinasi config sekaligus.

Format berkas sweep (YAML):

    cells:
      - nama: rf_200_d10          # opsional; dipakai sebagai nama folder
        rf: {n_estimators: 200, max_depth: 10}
      - nama: croston_a020
        croston.alpha: 0.20       # kunci bertitik juga diterima

Setiap sel dijalankan dengan config dasar + override, ditulis ke `out/<nama>/` lengkap dengan
manifest SHA-256-nya, lalu satu baris ringkasan per sel ditambahkan ke `out/ringkasan_sweep.csv`.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import yaml

from . import pipeline


def _terapkan(cfg: dict, override: dict) -> None:
    """Terapkan override ke salinan config; mendukung kunci bersarang maupun bertitik."""
    for kunci, nilai in override.items():
        if isinstance(nilai, dict):
            _terapkan(cfg.setdefault(kunci, {}), nilai)
            continue
        bagian = kunci.split(".") if "." in kunci else [kunci]
        simpul = cfg
        for b in bagian[:-1]:
            simpul = simpul.setdefault(b, {})
        simpul[bagian[-1]] = nilai


def _ringkas(nama: str, override: dict, ringkasan: dict) -> dict:
    h1 = ringkasan.get("h1") or []
    h2 = ringkasan.get("h2") or []
    return {
        "sel": nama,
        "override": json.dumps(override, sort_keys=True),
        "h1_didukung": ringkasan.get("h1_didukung"),
        "h1_n_titik": len(h1),
        "h1_p_holm_min": min((r["p_holm"] for r in h1), default=float("nan")),
        "h1_besaran_median": (sum(r["median_selisih"] for r in h1) / len(h1)) if h1 else float("nan"),
        "h2_didukung": ringkasan.get("h2_didukung"),
        "h2_n_titik": len(h2),
        "h2_p_page_holm_min": min((r["p_page_holm"] for r in h2), default=float("nan")),
        "h2_besaran_median": (sum(r["median_delta"] for r in h2) / len(h2)) if h2 else float("nan"),
    }


def jalankan(cfg: dict, path_sweep: Path, sumber: Path, out: Path) -> pd.DataFrame:
    with open(path_sweep, encoding="utf-8") as fh:
        isi = yaml.safe_load(fh) or {}
    sel = isi.get("cells") or []
    if not sel:
        raise ValueError(f"tidak ada `cells` di {path_sweep}")
    out.mkdir(parents=True, exist_ok=True)
    baris = []
    for i, satu in enumerate(sel, 1):
        satu = dict(satu)
        nama = str(satu.pop("nama", f"sel_{i:02d}"))
        c = copy.deepcopy(cfg)
        _terapkan(c, satu)
        print(f"[sweep {i}/{len(sel)}] {nama} <- {json.dumps(satu, sort_keys=True)}", flush=True)
        out_sel = out / nama
        ringkasan = pipeline.jalankan(c, sumber, out_sel, verbose=False)
        baris.append(_ringkas(nama, satu, ringkasan))
    tabel = pd.DataFrame(baris)
    tabel.to_csv(out / "ringkasan_sweep.csv", index=False)
    print(f"ringkasan sweep: {out / 'ringkasan_sweep.csv'} ({len(tabel)} sel)")
    return tabel
