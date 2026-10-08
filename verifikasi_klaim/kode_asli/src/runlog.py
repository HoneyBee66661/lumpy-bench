"""run_log.txt — jejak eksekusi (spec Bagian 6).

Menulis: waktu mulai/selesai, durasi, versi python + tiap library, seed,
jumlah SKU final, dan status tiap tahap. Dipanggil oleh run_all.py.
"""
from __future__ import annotations

import platform
import sys
import time
from pathlib import Path

LIB_NAMES = ["pandas", "numpy", "sklearn", "statsforecast", "scipy", "matplotlib", "seaborn"]


def _lib_versions() -> str:
    lines = []
    for name in LIB_NAMES:
        try:
            mod = __import__(name)
            lines.append(f"  {name}=={getattr(mod, '__version__', 'unknown')}")
        except Exception as exc:  # noqa: BLE001
            lines.append(f"  {name}=NOT_INSTALLED ({type(exc).__name__})")
    return "\n".join(lines)


def write(
    path: Path,
    stage: str,
    seed: int,
    started_at: float,
    extra: str = "",
    status: str = "ok",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ended_at = time.time()
    block = (
        f"\n=== {stage} ===\n"
        f"status      : {status}\n"
        f"started     : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(started_at))}\n"
        f"ended       : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ended_at))}\n"
        f"duration_s  : {ended_at - started_at:.1f}\n"
        f"seed        : {seed}\n"
        f"python      : {platform.python_version()} ({sys.executable})\n"
        f"platform    : {platform.platform()}\n"
        f"libraries   :\n{_lib_versions()}\n"
        + (f"extra       :\n{extra}\n" if extra else "")
    )
    with open(path, "a") as fh:
        fh.write(block)


def header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(
            "PaperOrchestra-independent run log — Random Forest vs Croston-SBA "
            "(lumpy demand, asymmetric cost)\n"
            f"log opened  : {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
