"""Orkestrasi alur penuh: data -> peramalan -> kebijakan -> penyetaraan -> statistik -> artefak."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import classify, data, features, forecasters, manifest, policy, sampling, stats


def _tulis(df: pd.DataFrame, path: Path, digits: int = 6) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.round(int(digits)).to_csv(path, index=False)


def _frame_h1_kosong() -> pd.DataFrame:
    """Frame H1 kosong dengan kolom dan dtype yang tetap (dipakai bila data terlalu tipis)."""
    angka = ["n", "median_selisih", "p_satu_sisi", "pct_lebih_murah", "p_holm"]
    f = pd.DataFrame({k: pd.Series(dtype="float64") for k in angka})
    f["uji"] = pd.Series(dtype=object)
    f["origin"] = pd.Series(dtype="int64")
    return f[["uji", "n", "median_selisih", "p_satu_sisi", "pct_lebih_murah", "origin", "p_holm"]]


def _didukung(frame: pd.DataFrame, kolom: str, ambang: float) -> bool:
    """True bila ada p terkoreksi di bawah ambang; aman untuk frame kosong."""
    if frame is None or len(frame) == 0 or kolom not in frame.columns:
        return False
    nilai = frame[kolom]
    if isinstance(nilai, pd.DataFrame):            # kolom berganda -> ambil yang pertama
        nilai = nilai.iloc[:, 0]
    return bool((pd.to_numeric(nilai, errors="coerce") < ambang).any())


def jendela_rolling(panjang: pd.DataFrame, n_origin: int, test_days: int, val_days: int) -> list[dict]:
    total = int(panjang["day"].max())
    out = []
    for i in range(1, n_origin + 1):
        val_end = total - test_days * (n_origin - i + 1)
        out.append({
            "origin": i,
            "train_end": val_end - val_days,
            "val_end": val_end,
            "days_val": np.arange(val_end - val_days + 1, val_end + 1),
            "days_test": np.arange(val_end + 1, min(val_end + 1 + test_days, total + 1)),
        })
    return out


def jalankan(cfg: dict, sumber: Path, out: Path, verbose: bool = True) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    catat = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    digits = int(cfg.get("output", {}).get("round_digits", 6))
    simpan = lambda d, p: _tulis(d, p, digits)          # presisi artefak dari config

    # 1-3: data, periode aktif, klasifikasi
    panjang = data.muat_long(str(sumber))
    aktif = data.potong_periode_aktif(panjang)
    stat = classify.klasifikasi(aktif, cfg["lumpy"]["adi_min"], cfg["lumpy"]["cv2_min"],
                                cfg["lumpy"].get("min_nonzero", 2))
    meta = aktif.groupby("sku_id", observed=True)[["cat_id", "state_id"]].first().reset_index()
    catat(f"[1] deret: {panjang['sku_id'].nunique()} | lumpy: {int(stat['lumpy'].sum())} "
          f"({100 * stat['lumpy'].mean():.1f}%)")

    # 4: sampel berstrata
    sampel = sampling.ambil_sampel(stat, meta, n_sku=cfg["sampling"]["n_sku"], seed=cfg["seed"])
    sku_ids = sampel["sku_id"].tolist()
    catat(f"[2] sampel: {len(sku_ids)} SKU")
    simpan(stat, out / "tables" / "statistik_sku.csv")
    simpan(sampel, out / "tables" / "sampel.csv")

    # 5: matriks fitur
    horizon = int(cfg["policy"]["lead"] + cfg["policy"]["review"])
    blok_fitur = features.dari_config(cfg.get("features"))
    matriks = features.bangun_matriks(panjang[panjang["sku_id"].isin(sku_ids)], sku_ids, horizon,
                                      blok_fitur)
    kolom_fitur = features.daftar_fitur(**blok_fitur)
    origins = jendela_rolling(panjang[panjang["sku_id"].isin(sku_ids)], cfg["origin"]["n_origin"],
                              cfg["origin"]["test_days"], cfg["origin"]["val_days"])

    # 6-8: per titik evaluasi -> peramalan, sapuan kebijakan, penyetaraan
    taus = cfg["policy"]["taus"]
    target = cfg["policy"]["fill_targets"]
    baris_fill, baris_akurasi = [], []
    for o in origins:
        # tiga potongan: latih (<= train_end), validasi (train_end..val_end), uji (val_end..akhir)
        latih, _ = features.potong_latih_uji(matriks, o["train_end"], o["train_end"], kolom_fitur)
        _, val = features.potong_latih_uji(matriks, o["train_end"], o["val_end"], kolom_fitur)
        _, uji = features.potong_latih_uji(matriks, o["val_end"], o["days_test"][-1], kolom_fitur)
        val = val[val["day"] >= o["days_val"][0]]
        uji = uji[uji["day"] >= o["days_test"][0]]
        # model validasi (data <= train_end) dan model uji (data <= val_end)
        model_val = forecasters.latih_rf(latih, kolom_fitur, horizon, cfg["rf"], cfg["seed"])
        model_uji = forecasters.latih_rf(pd.concat([latih, val], ignore_index=True), kolom_fitur,
                                         horizon, cfg["rf"], cfg["seed"])
        pv = forecasters.ramal_rf(model_val, val, kolom_fitur)
        pt = forecasters.ramal_rf(model_uji, uji, kolom_fitur)

        for sku in sku_ids:
            seri = panjang[(panjang["sku_id"] == sku) & (panjang["day"].isin(o["days_test"]))]
            if seri.empty:
                continue
            permintaan = seri.sort_values("day")["demand"].to_numpy(float)
            hari = seri.sort_values("day")["day"].to_numpy(int)
            skor = {}
            for nama, ramal in (("rf", pt), ("croston_sba", None)):
                if ramal is None:
                    ramal_sku = forecasters.ramal_croston_sba(
                        panjang[panjang["sku_id"] == sku], hari, horizon, cfg["croston"]["alpha"])
                    pred = ramal_sku.sort_values("day")["prediksi"].to_numpy(float)
                else:
                    r = ramal[ramal["sku_id"] == sku].sort_values("day")
                    pred = r["prediksi"].to_numpy(float)
                    galat_val = (pv[pv["sku_id"] == sku]["aktual"] - pv[pv["sku_id"] == sku]["prediksi"]).to_numpy(float)
                if nama == "croston_sba":
                    rv = forecasters.ramal_croston_sba(
                        panjang[panjang["sku_id"] == sku], o["days_val"], horizon, cfg["croston"]["alpha"])
                    galat_val = None
                    # galat validasi dihitung terhadap target H-jumlah yang sebenarnya
                    m = matriks[(matriks["sku_id"] == sku) & (matriks["day"].isin(o["days_val"]))]
                    galat_val = (m.sort_values("day")["target"].to_numpy(float)
                                 - rv.sort_values("day")["prediksi"].to_numpy(float))
                skor[nama] = (pred, galat_val)

            for nama, (pred, galat_val) in skor.items():
                n = min(len(permintaan), len(pred))
                sweep = policy.sapu_kebijakan(permintaan[:n], pred[:n], galat_val, taus,
                                              review=cfg["policy"]["review"], lead=cfg["policy"]["lead"],
                                              warmup=cfg["policy"]["warmup"])
                for cs, ch in cfg["cost_scenarios"]:
                    s = policy.biaya(sweep, cs, ch)
                    s["sku_id"], s["origin"], s["model"] = sku, o["origin"], nama
                    baris_fill.append(s)
                mae = float(np.mean(np.abs(permintaan[:n] - pred[:n])))
                baris_akurasi.append({"sku_id": sku, "origin": o["origin"], "model": nama,
                                      "mae": mae, "n": n})
        catat(f"[3] titik {o['origin']}: uji {o['days_test'][0]}-{o['days_test'][-1]} selesai")

    fill = pd.concat(baris_fill, ignore_index=True)
    simpan(fill, out / "tables" / "kebijakan.csv")
    akurasi = pd.DataFrame(baris_akurasi)
    simpan(akurasi, out / "tables" / "akurasi.csv")

    # 9: penyetaraan tingkat layanan
    baris_set = []
    for (sku, model, origin, cs), g in fill.groupby(["sku_id", "model", "origin", "cs"], observed=True):
        setara = policy.setarakan_layanan(g, target)
        setara["sku_id"], setara["model"], setara["origin"], setara["cs"] = sku, model, origin, cs
        baris_set.append(setara)
    setara = pd.concat(baris_set, ignore_index=True)
    simpan(setara, out / "tables" / "penyetaraan.csv")

    # 10: statistik H1 dan H2 pada skenario dasar
    cs0 = float(cfg["cost_scenarios"][0][0])
    t95, t90, t98 = target[0], target[0], target[-1]
    hasil = {}
    for nama, t_target in (("H1", target[1] if len(target) > 1 else target[0]),):
        pivot = (setara[(setara["cs"] == cs0) & (setara["fill_target"] == t_target)]
                 .pivot_table(index=["sku_id", "origin"], columns="model", values="cost_at_target", aggfunc="first")
                 .dropna())
        kosong = _frame_h1_kosong()
        if pivot.empty or not {"rf", "croston_sba"}.issubset(pivot.columns):
            hasil[nama] = kosong
            catat(f"[!] {nama}: tidak ada SKU dengan kedua model pada target {t_target} "
                  f"(data terlalu tipis?); hasil dikosongkan")
            simpan(kosong, out / "tables" / "hasil_h1.csv")
            continue
        pivot = pivot[pivot["croston_sba"] != 0]
        pivot["relatif"] = (pivot["rf"] - pivot["croston_sba"]) / pivot["croston_sba"]
        baris = []
        for o, g in pivot.groupby("origin"):
            r = stats.ringkas_uji(f"{nama} titik {o}", g["relatif"].to_numpy(float),
                                  alpha=cfg["stats"]["alpha"], seed=cfg["stats"]["bootstrap_seed"])
            r["origin"] = o
            baris.append(r)
        h1 = pd.DataFrame(baris) if baris else kosong
        if not h1.empty:
            h1["p_holm"] = stats.holm(h1["p_satu_sisi"].to_numpy(float))
        hasil[nama] = h1
        simpan(h1, out / "tables" / "hasil_h1.csv")

    # H2: uji Page pada urutan target + kontras 90 vs 98
    baris = []
    pivot = (setara[setara["cs"] == cs0]
             .pivot_table(index=["sku_id", "origin", "fill_target"], columns="model",
                          values="cost_at_target", aggfunc="first").dropna())
    layak = (not pivot.empty) and {"rf", "croston_sba"}.issubset(pivot.columns)
    if not layak:
        catat("[!] H2: tidak ada SKU dengan kedua model pada semua target "
              "(data terlalu tipis?); hasil dikosongkan")
    else:
        pivot = pivot[pivot["croston_sba"] != 0]
        pivot["relatif"] = (pivot["rf"] - pivot["croston_sba"]) / pivot["croston_sba"]
        for o, g in pivot.groupby("origin"):
            tabel = g["relatif"].unstack("fill_target").dropna().sort_index(axis=1)
            if tabel.shape[1] < 3 or len(tabel) < 4:
                continue
            kol = list(tabel.columns)
            delta = tabel[kol[-1]].to_numpy(float) - tabel[kol[0]].to_numpy(float)
            L, p_page = stats.page_test(tabel[kol].to_numpy(float))
            p_kontras = stats.wilcoxon_satu_sisi(delta, arah_lebih_kecil=False)
            baris.append({"origin": o, "n": len(tabel), "median_90": float(np.median(tabel[kol[0]])),
                          "median_98": float(np.median(tabel[kol[-1]])), "median_delta": float(np.median(delta)),
                          "page_L": L, "p_page": p_page, "p_kontras_98_vs_90": p_kontras,
                          "arah_naik": bool(np.median(delta) > 0)})
    h2 = pd.DataFrame(baris)
    if not h2.empty:
        h2["p_page_holm"] = stats.holm(h2["p_page"].to_numpy(float))
        h2["p_kontras_holm"] = stats.holm(h2["p_kontras_98_vs_90"].to_numpy(float))
    simpan(h2, out / "tables" / "hasil_h2.csv")

    ambang = float(cfg["stats"]["alpha"])
    ringkas = {
        "n_deret": int(panjang["sku_id"].nunique()),
        "n_lumpy": int(stat["lumpy"].sum()),
        "n_sampel": len(sku_ids),
        "origin": [int(o["origin"]) for o in origins],
        "fill_targets": [float(t) for t in target],
        "cost_scenarios": cfg["cost_scenarios"],
        "h1": (hasil["H1"][["origin", "n", "median_selisih", "p_satu_sisi", "p_holm"]].to_dict("records")
               if not hasil["H1"].empty else []),
        "h1_didukung": _didukung(hasil["H1"], "p_holm", ambang),
        "h2": [] if h2.empty else h2[["origin", "n", "median_delta", "p_page", "p_page_holm",
                                      "p_kontras_holm", "arah_naik"]].to_dict("records"),
        "h2_didukung": bool(_didukung(h2, "p_page_holm", ambang) and len(h2) > 0
                            and bool(h2["arah_naik"].all())),
    }
    (out / "ringkasan.json").write_text(json.dumps(ringkas, indent=2, sort_keys=True) + "\n")
    manifest.tulis_manifest(out)
    catat(f"[4] artefak ditulis ke {out} | manifest SHA-256 dibuat")
    catat(f"    H1 didukung: {ringkas['h1_didukung']} | H2 didukung: {ringkas['h2_didukung']}")
    return ringkas
