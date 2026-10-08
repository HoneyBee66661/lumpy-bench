#!/usr/bin/env python3
"""Gabungkan blok run_log dari dua sumber (mis. lokal + Kaggle) tanpa saling menimpa.

Masalah yang diselesaikan: menarik `run_log.txt` dari Kaggle menimpa bagian M1/M2 yang hanya
ada di lokal (dan sebaliknya). Skrip ini membaca blok `=== <stage> ===` dari berkas,
menggabungkan berdasarkan nama blok, dan menulis ulang hasilnya.

    .venv/bin/python src/merge_runlog.py --base results/run_log.txt --incoming kaggle/v2/out/results/run_log.txt
    # --incoming boleh diulang; blok dengan nama stage sama akan diambil dari --incoming (lebih baru)

Urutan blok: mengikuti urutan kemunculan pertama di (base, lalu incoming), kecuali blok pengganti
yang ditaruh di posisi blok aslinya supaya urutan M1..M6 tetap masuk akal.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

BLOCK = re.compile(r"^=== (.+?) ===$", re.M)


def parse(path: Path) -> tuple[str, dict[str, str]]:
    """Kembalikan (pengantar, {nama stage: isi blok}) — isi blok termasuk header '=== ... ==='."""
    text = path.read_text()
    marks = list(BLOCK.finditer(text))
    if not marks:
        return text, {}
    preamble = text[: marks[0].start()]
    blocks: dict[str, str] = {}
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        blocks[m.group(1).strip()] = text[m.start():end]
    return preamble, blocks


def main() -> int:
    ap = argparse.ArgumentParser(description="Gabungkan blok run_log tanpa menimpa.")
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--incoming", type=Path, action="append", default=[])
    ap.add_argument("--out", type=Path, default=None, help="default: timpa --base")
    a = ap.parse_args()

    base_text = a.base.read_text()
    preamble, blocks = parse(a.base)
    order = list(blocks)

    replaced: list[str] = []
    added: list[str] = []
    for inc in a.incoming:
        _, inc_blocks = parse(inc)
        for name, body in inc_blocks.items():
            if name in blocks:
                replaced.append(name)
            else:
                added.append(name)
                order.append(name)
            blocks[name] = body

    out = preamble + "".join(blocks[name] for name in order)
    (a.out or a.base).write_text(out)
    print(f"{a.base} -> {a.out or a.base}")
    print(f"  blok: {len(order)} | diganti dari incoming: {len(replaced)} ({', '.join(replaced)})")
    print(f"  blok baru dari incoming: {len(added)} ({', '.join(added)})")
    print(f"  total baris: {len(base_text.splitlines())} -> {len(out.splitlines())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
