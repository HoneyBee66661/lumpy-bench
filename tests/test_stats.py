"""Uji statistik dan determinasi."""
from __future__ import annotations

import numpy as np

from lumpy_bench import stats


def test_holm_monoton_dan_terkoreksi():
    p = np.array([0.01, 0.04, 0.03, 0.20])
    adj = stats.holm(p)
    assert (adj >= p - 1e-12).all()
    assert adj[np.argmin(p)] == np.min(adj)


def test_page_menangkap_tren_naik():
    rng = np.random.default_rng(3)
    naik = rng.normal(0, 1, (40, 3)) + np.array([0, 1, 2])
    acak = rng.normal(0, 1, (40, 3))
    assert stats.page_test(naik)[1] < 0.01
    assert stats.page_test(acak)[1] > 0.05


def test_wilcoxon_arah():
    positif = np.arange(1.0, 21.0)
    assert stats.wilcoxon_satu_sisi(positif, arah_lebih_kecil=False) < 0.01
    assert stats.wilcoxon_satu_sisi(positif, arah_lebih_kecil=True) > 0.9
