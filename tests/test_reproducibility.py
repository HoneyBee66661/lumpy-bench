"""Determinisme menyeluruh: dua kali jalan harus menghasilkan artefak identik."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from lumpy_bench import manifest, pipeline, synth

CFG = {
    "seed": 4242,
    "paths": {"data_contoh": "data_contoh/sample_retail_long.csv"},
    "lumpy": {"adi_min": 1.32, "cv2_min": 0.49},
    "sampling": {"n_sku": 8},
    "origin": {"n_origin": 1, "test_days": 90, "val_days": 90},
    "rf": {"n_estimators": 40, "max_depth": 6, "min_samples_leaf": 5, "n_jobs": 1},
    "croston": {"alpha": 0.10},
    "policy": {"review": 1, "lead": 7, "warmup": 7, "taus": [0.5, 0.8, 0.95, 0.99],
               "fill_targets": [0.90, 0.95, 0.98]},
    "cost_scenarios": [[5, 1], [20, 1]],
}


def test_dua_kali_jalan_identik(tmp_path: Path):
    data = tmp_path / "sampel.csv"
    synth.simpan(str(data), synth.buat_data_contoh(n_sku=24, n_hari=420, seed=1))
    a, b = tmp_path / "a", tmp_path / "b"
    pipeline.jalankan(CFG, data, a, verbose=False)
    pipeline.jalankan(CFG, data, b, verbose=False)
    ma = json.loads((a / "manifest.sha256.json").read_text())
    mb = json.loads((b / "manifest.sha256.json").read_text())
    assert ma == mb
    assert manifest.bandingkan_manifest(b)["sah"]
