"""Determinisme menyeluruh: dua kali jalan harus menghasilkan artefak identik."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml

from lumpy_bench import manifest, pipeline, synth

AKAR = Path(__file__).resolve().parents[1]


def _cfg_uji() -> dict:
    """Config asli dipakai sebagai dasar supaya uji ini tidak pernah ketinggalan kunci baru."""
    cfg = copy.deepcopy(yaml.safe_load((AKAR / "config.yaml").read_text()))
    cfg["seed"] = 4242
    cfg["sampling"]["n_sku"] = 8
    cfg["origin"].update({"n_origin": 1, "test_days": 90, "val_days": 90})
    cfg["rf"].update({"n_estimators": 40, "max_depth": 6, "min_samples_leaf": 5, "n_jobs": 1})
    cfg["policy"]["taus"] = [0.50, 0.80, 0.95, 0.99]
    cfg["cost_scenarios"] = [[5, 1], [20, 1]]
    return cfg


def test_dua_kali_jalan_identik(tmp_path: Path):
    cfg = _cfg_uji()
    data = tmp_path / "sampel.csv"
    synth.simpan(str(data), synth.buat_data_contoh(n_sku=24, n_hari=420, seed=1))
    a, b = tmp_path / "a", tmp_path / "b"
    pipeline.jalankan(cfg, data, a, verbose=False)
    pipeline.jalankan(cfg, data, b, verbose=False)
    ma = json.loads((a / "manifest.sha256.json").read_text())
    mb = json.loads((b / "manifest.sha256.json").read_text())
    assert ma == mb
    assert manifest.bandingkan_manifest(b)["sah"]


def test_perubahan_parameter_mengubah_hasil(tmp_path: Path):
    """Meta-uji: kalau override benar-benar diterapkan, artefaknya harus ikut berubah."""
    cfg = _cfg_uji()
    data = tmp_path / "sampel.csv"
    synth.simpan(str(data), synth.buat_data_contoh(n_sku=24, n_hari=420, seed=1))
    a, b = tmp_path / "a", tmp_path / "b"
    pipeline.jalankan(cfg, data, a, verbose=False)
    cfg2 = _cfg_uji()
    cfg2["rf"]["max_depth"] = 4
    pipeline.jalankan(cfg2, data, b, verbose=False)
    assert (json.loads((a / "manifest.sha256.json").read_text())
            != json.loads((b / "manifest.sha256.json").read_text()))
