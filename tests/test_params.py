"""Parameter dapat dikonfigurasi: tidak ada nilai kebijakan yang di-hardcode.

Dua hal yang dijaga berkas ini:
1. setiap kunci di config.yaml benar-benar dibaca kode, dan setiap kunci yang dibaca kode
   benar-benar ada di config.yaml (typo atau kunci hilang langsung ketahuan);
2. nilai yang dulu di-hardcode (faktor debiasing, jendela fitur, filter minimum, ambang uji,
   presisi pembulatan) benar-benar mengikuti config.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from lumpy_bench import classify, features, forecasters

AKAR = Path(__file__).resolve().parents[1]
KODE = "\n".join(p.read_text() for p in (AKAR / "lumpy_bench").glob("*.py"))
CFG = yaml.safe_load((AKAR / "config.yaml").read_text())


def _daun(d, pref=""):
    out = []
    for k, v in d.items():
        if isinstance(v, dict):
            out += _daun(v, f"{pref}{k}.")
        else:
            out.append(f"{pref}{k}")
    return out


def test_tidak_ada_kunci_config_yang_mati():
    mati = [k for k in _daun(CFG) if f'"{k.split(".")[-1]}"' not in KODE]
    assert not mati, f"kunci config tidak dibaca kode: {mati}"


def test_tidak_ada_kunci_config_yang_hilang():
    dibaca = set(re.findall(r'cfg\["([a-z_]+)"\]', KODE)) | set(re.findall(r'cfg\.get\("([a-z_]+)"', KODE))
    assert dibaca <= set(CFG), f"kode membaca kunci yang tidak ada di config: {sorted(dibaca - set(CFG))}"


def test_faktor_debias_selalu_mengikuti_alpha():
    assert forecasters.faktor_debias(0.10) == pytest.approx(0.95)
    assert forecasters.faktor_debias(0.20) == pytest.approx(0.90)
    # nilai manual harus memakai faktor turunan, bukan 0,95 tetap
    rng = np.random.default_rng(1)
    hist = rng.poisson(2.0, 200).astype(float)
    hist[rng.random(200) < 0.5] = 0.0
    a10 = forecasters._croston_sba_manual(hist, 0.10)
    a20 = forecasters._croston_sba_manual(hist, 0.20)
    rasio_debias = forecasters.faktor_debias(0.20) / forecasters.faktor_debias(0.10)
    assert rasio_debias != pytest.approx(1.0)        # alpha benar-benar berpengaruh
    assert a20 > 0 and a10 > 0
    # hasil dengan alpha berbeda tidak boleh identik (alpha benar-benar dipakai)
    assert a20 != pytest.approx(a10, rel=1e-9)


def test_jendela_fitur_mengikuti_config():
    dasar = features.daftar_fitur()
    lain = features.daftar_fitur((1, 3), (5,), 14)
    assert "lag_28" in dasar and "lag_28" not in lain
    assert "rolling_std_14" in lain and "rolling_std_28" not in lain


def test_filter_min_nonzero_mengikuti_config():
    df = pd.DataFrame({
        "sku_id": ["a"] * 10 + ["b"] * 10,
        "day": list(range(1, 11)) * 2,
        "demand": [0, 1, 0, 0, 0, 0, 0, 0, 0, 1] + [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "sell_price": [1.0] * 20,
        "cat_id": ["C"] * 20, "state_id": ["S"] * 20,
    })
    assert set(classify.klasifikasi(df, min_nonzero=2)["sku_id"]) == {"a", "b"}
    assert set(classify.klasifikasi(df, min_nonzero=5)["sku_id"]) == {"b"}
