import json, urllib.parse, urllib.request, sys

API="https://commons.wikimedia.org/w/api.php"
UA="SkyEyeTestHarness/1.0 (evaluacion de deteccion; contacto: equipo SkyEye)"
LIBRES=("cc0","cc-by","cc by","public domain","pd-","cc-zero","attribution")

def get(params):
    url=API+"?"+urllib.parse.urlencode(params)
    req=urllib.request.Request(url, headers={"User-Agent":UA})
    return json.load(urllib.request.urlopen(req, timeout=45))

def buscar(term, n=12):
    r=get({"action":"query","list":"search","srsearch":f'filetype:video {term}',
           "srnamespace":"6","srlimit":str(n),"format":"json"})
    return [x["title"] for x in r.get("query",{}).get("search",[])]

def info(titulos):
    out=[]
    for i in range(0,len(titulos),10):
        r=get({"action":"query","titles":"|".join(titulos[i:i+10]),
               "prop":"imageinfo","iiprop":"url|extmetadata|size|mediatype",
               "format":"json"})
        for p in r.get("query",{}).get("pages",{}).values():
            ii=(p.get("imageinfo") or [{}])[0]
            em=ii.get("extmetadata",{})
            lic=(em.get("LicenseShortName",{}).get("value") or "").strip()
            out.append({"titulo":p.get("title"),"url":ii.get("url"),
                        "licencia":lic,
                        "autor":(em.get("Artist",{}).get("value") or "")[:120],
                        "bytes":ii.get("size",0)})
    return out

def libre(lic):
    l=lic.lower()
    return any(k in l for k in LIBRES) and "nc" not in l.replace("include","") and "nd" not in l.split()

if __name__=="__main__":
    for term in sys.argv[1:]:
        print(f"\n########## {term} ##########")
        try: t=buscar(term)
        except Exception as e: print("  error:", e); continue
        if not t: print("  (sin resultados)"); continue
        for c in info(t):
            if not c["url"]: continue
            ok = "LIBRE " if libre(c["licencia"]) else "  ??? "
            print(f'  {ok} {c["licencia"][:22]:24} {c["bytes"]/1e6:6.1f}MB  {c["titulo"][:70]}')
