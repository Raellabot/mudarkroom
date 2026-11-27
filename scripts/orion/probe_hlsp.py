# Finds a working download address for the Orion Treasury ACS mosaics; prints what each candidate returns.
import re, urllib.request
from astroquery.mast import Observations

ds = "hlsp_orion_hst_acs_strip0l_f435w_v1_drz"
obs = Observations.query_criteria(obs_collection="HLSP", proposal_id="10246")
row = [o for o in obs if str(o["obs_id"]) == ds]
print("obs row:", {k: str(row[0][k]) for k in row[0].colnames} if row else None, flush=True)
prods = Observations.get_product_list(row) if row else []
for p in prods:
    print("product:", {k: str(p[k]) for k in p.colnames}, flush=True)
try:
    print("download_file:", Observations.download_file(str(prods[0]["dataURI"]), local_path="t.fits", cache=False), flush=True)
except Exception as e:
    print("download_file error:", repr(e), flush=True)


def probe(url, n=400):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "cosmic-darkroom builder", "Range": "bytes=0-399"})
        with urllib.request.urlopen(req, timeout=60) as r:
            b = r.read(n)
            print("OK", r.status, r.headers.get("Content-Type"), r.headers.get("Content-Range") or r.headers.get("Content-Length"), url, repr(b[:120]), flush=True)
            return b
    except Exception as e:
        print("FAIL", url, repr(e)[:200], flush=True)


for u in [f"https://archive.stsci.edu/cgi-bin/getdata.cgi?dataset={ds}", f"https://archive.stsci.edu/hlsps/orion/{ds}.fits", f"https://archive.stsci.edu/hlsps/orion/data/{ds}.fits", f"https://archive.stsci.edu/pub/hlsp/orion/{ds}.fits", f"https://archive.stsci.edu/missions/hlsp/orion/{ds}.fits", f"https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HLSP/orion/{ds}.fits"]:
    probe(u)
for d in ["https://archive.stsci.edu/hlsps/orion/", "https://archive.stsci.edu/pub/hlsp/orion/", "https://archive.stsci.edu/missions/hlsp/orion/", "https://archive.stsci.edu/hlsp/orion"]:
    try:
        req = urllib.request.Request(d, headers={"User-Agent": "cosmic-darkroom builder"})
        html = urllib.request.urlopen(req, timeout=60).read().decode("utf8", "replace")
        links = sorted(set(re.findall(r'href="([^"]+)"', html)))
        print("LIST", d, len(links), [l for l in links if "acs" in l.lower() or "data" in l.lower() or l.endswith("/")][:60], flush=True)
    except Exception as e:
        print("LIST FAIL", d, repr(e)[:200], flush=True)
