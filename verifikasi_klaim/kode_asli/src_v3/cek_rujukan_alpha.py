
import json, urllib.request, urllib.parse, time, inspect, re
UA = {"User-Agent": "hermes-agent/1.0 (mailto:honeybee66661@gmail.com)"}
def cari(q, rows=4):
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode({
        "query.bibliographic": q, "rows": rows,
        "select": "DOI,title,container-title,volume,issue,page,published-print,published-online,author"})
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.load(r)["message"]["items"]

print("== B10: kandidat rujukan pengganti Theodorou dkk. (2023) ==")
for q in ["Spiliotis 2021 empirical estimates inventory performance intermittent demand International Journal of Production Economics",
          "Spiliotis Syntetos empirical safety stock intermittent demand inventory performance 2021"]:
    print("--", q[:70])
    for it in cari(q):
        t = (it.get("title") or ["?"])[0][:95]
        c = (it.get("container-title") or ["?"])[0][:50]
        y = (it.get("published-print") or it.get("published-online") or {}).get("date-parts", [[None]])[0][0]
        a = ", ".join(x.get("family", "?") for x in (it.get("author") or [{}])[:3])
        print(f"   {it['DOI']:40} | {y} | {a:32} | v{it.get('volume')}({it.get('issue')}) {it.get('page')} | {c}")
        print(f"      {t}")
    time.sleep(0.6)

print("\n== A9: alpha bawaan CrostonSBA di pustaka statsforecast 2.0.3 ==")
from statsforecast import models as M
import statsforecast
print("versi:", statsforecast.__version__)
src = inspect.getsource(M.CrostonSBA)
print("docstring CrostonSBA:")
print(inspect.getdoc(M.CrostonSBA)[:600])
# cari alpha di seluruh modul intermittent
mod = inspect.getsource(M)
for m in re.finditer(r"alpha\s*[:=]\s*[0-9.]+", mod):
    print("   parameter alpha ditemukan:", m.group(0))
for nama in ("CrostonClassic", "CrostonOptimized", "TSB", "ADIDA", "IMAPA"):
    if hasattr(M, nama):
        sig = str(inspect.signature(getattr(M, nama).__init__))
        print(f"   {nama}{sig[:110]}")
