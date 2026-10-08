
import json, urllib.request, urllib.parse, time
UA = {"User-Agent": "hermes-agent/1.0 (mailto:honeybee66661@gmail.com)"}
def q(params):
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=45) as r:
        return json.load(r)["message"]["items"]

print("== semua karya Spiliotis di IJPE 2019-2023 ==")
it = q({"query.author": "Spiliotis", "filter": "container-title:International Journal of Production Economics,from-pub-date:2019-01-01,until-pub-date:2023-12-31",
        "rows": 12, "select": "DOI,title,container-title,volume,issue,page,published-print,author"})
for x in it:
    t = (x.get("title") or ["?"])[0][:100]
    y = (x.get("published-print") or {}).get("date-parts", [[None]])[0][0]
    au = ", ".join(a.get("family","?") for a in (x.get("author") or [])[:4])
    print(f"  {x['DOI']:36} | {y} | v{x.get('volume')} {x.get('page')} | {au}\n     {t}")

print("\n== cari frasa kunci (judul) ==")
for judul in ["Empirical safety stocks intermittent demand",
              "simple empirical estimates inventory performance intermittent demand",
              "forecasting and stock control performance simple estimates intermittent"]:
    got = q({"query.bibliographic": judul, "rows": 3,
             "select": "DOI,title,container-title,volume,page,published-print,author"})
    print("--", judul)
    for x in got:
        t = (x.get("title") or ["?"])[0][:95]
        c = (x.get("container-title") or ["?"])[0][:45]
        y = (x.get("published-print") or x.get("published-online") or {}).get("date-parts", [[None]])[0][0]
        au = ", ".join(a.get("family","?") for a in (x.get("author") or [])[:3])
        print(f"   {x['DOI']:36} | {y} | {au[:36]:36} | {c} | {t}")
    time.sleep(0.5)
