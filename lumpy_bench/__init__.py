"""lumpy-bench — kerangka pengujian peramalan permintaan lumpy pada kebijakan persediaan.

Alur: data mentah -> periode aktif -> klasifikasi ADI/CV2 -> sampel berstrata -> pembagian
rolling origin -> fitur -> peramalan -> simulasi kebijakan order-up-to -> penyetaraan tingkat
layanan -> statistik -> artefak + manifest SHA-256.
"""
__version__ = "0.1.0"
