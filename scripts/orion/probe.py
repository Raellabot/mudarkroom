# Lists the Hubble (MAST) and Spitzer (IRSA) products around the Orion Nebula for a Hubble + Spitzer subject; usage: probe.py OUTDIR
import json, os, sys, time
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.ipac.irsa import Irsa
from astroquery.mast import Observations

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
pos = SkyCoord(83.7904, -5.4136, unit="deg")
rep = {"irsa": {}, "hlsp": [], "hlsp_products": [], "hst10246": []}
t0 = time.time()
for coll in ("spitzer_seip",):
    try:
        t = Irsa.query_sia(pos=(pos, 0.25 * u.deg), collection=coll)
        rep["irsa"][coll] = [{k: str(r[k]) for k in t.colnames if k in ("obs_id", "obs_title", "access_url", "access_estsize", "dataproduct_subtype", "energy_bandpassname", "instrument_name", "s_ra", "s_dec", "s_fov", "s_xel1", "s_xel2", "calib_level", "obs_collection")} for r in t]
        print(coll, len(t), f"{time.time() - t0:.0f}s", flush=True)
    except Exception as e:
        rep["irsa"][coll] = {"error": repr(e)}
        print(coll, "error", e, flush=True)
json.dump(rep, open(os.path.join(out, "probe.json"), "w"), indent=1)
for crit in ({"obs_collection": "HLSP", "proposal_id": "10246"}, {"obs_collection": "HLSP", "provenance_name": "ORION"}, {"obs_collection": "HLSP", "provenance_name": "orion"}):
    try:
        hl = Observations.query_criteria(**crit)
        print(crit, len(hl), f"{time.time() - t0:.0f}s", flush=True)
        for r in hl:
            rep["hlsp"].append({k: str(r[k]) for k in ("provenance_name", "proposal_id", "target_name", "instrument_name", "filters", "obs_id", "obsid", "s_ra", "s_dec", "dataproduct_type")})
        if len(hl):
            prods = Observations.get_product_list(hl[:60])
            for p in prods:
                rep["hlsp_products"].append({k: str(p[k]) for k in ("obs_id", "productFilename", "dataURI", "size", "productType")})
            print("  products", len(prods), f"{time.time() - t0:.0f}s", flush=True)
            break
    except Exception as e:
        print(crit, "error", e, flush=True)
    json.dump(rep, open(os.path.join(out, "probe.json"), "w"), indent=1)
try:
    h = Observations.query_criteria(obs_collection="HST", proposal_id="10246", calib_level=[3])
    print("HST 10246 level 3:", len(h), f"{time.time() - t0:.0f}s", flush=True)
    for r in h[:400]:
        rep["hst10246"].append({k: str(r[k]) for k in ("target_name", "instrument_name", "filters", "obs_id", "s_ra", "s_dec")})
except Exception as e:
    print("HST error", e, flush=True)
json.dump(rep, open(os.path.join(out, "probe.json"), "w"), indent=1)
print("== HLSP observations (first 40)")
for h in rep["hlsp"][:40]:
    print(" ", h["provenance_name"], h["target_name"], h["instrument_name"], h["filters"], h["obs_id"], h["s_ra"], h["s_dec"], h["dataproduct_type"])
print("== HLSP provenance names:", sorted({h["provenance_name"] for h in rep["hlsp"]}))
print("== HLSP products")
for p in rep["hlsp_products"]:
    print(" ", p["obs_id"], p["productFilename"], p["size"], p["productType"], p["dataURI"])
print("== SEIP rows")
for r in (rep["irsa"].get("spitzer_seip") or []):
    if isinstance(r, dict):
        print(" ", r.get("obs_id"), r.get("dataproduct_subtype"), r.get("energy_bandpassname"), r.get("access_estsize"), r.get("s_ra"), r.get("s_dec"), r.get("s_fov"), r.get("access_url"))
