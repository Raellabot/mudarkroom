# Lists the Hubble (MAST) and Spitzer (IRSA) products around the Orion Nebula for a Hubble + Spitzer subject; usage: probe.py OUTDIR
import json, os, sys
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.ipac.irsa import Irsa
from astroquery.mast import Observations

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
pos = SkyCoord(83.7904, -5.4136, unit="deg")
rep = {"mast": [], "mast_products": [], "irsa": {}}
obs = Observations.query_region(pos, radius=0.3 * u.deg)
keep = obs[(obs["obs_collection"] == "HLSP") | ((obs["obs_collection"] == "HST") & (obs["proposal_id"] == "10246"))]
print("MAST rows near Orion:", len(obs), "HLSP or 10246:", len(keep), flush=True)
for r in keep:
    rep["mast"].append({k: str(r[k]) for k in ("obs_collection", "provenance_name", "proposal_id", "target_name", "instrument_name", "filters", "obs_id", "dataproduct_type", "calib_level", "s_ra", "s_dec", "obsid", "dataURL")})
for p in sorted({(m["obs_collection"], m["provenance_name"]) for m in rep["mast"]}):
    print("  ", p, sum(1 for m in rep["mast"] if (m["obs_collection"], m["provenance_name"]) == p), flush=True)
hl = keep[keep["obs_collection"] == "HLSP"]
if len(hl):
    prods = Observations.get_product_list(hl)
    for p in prods:
        rep["mast_products"].append({k: str(p[k]) for k in ("obsID", "obs_id", "productFilename", "dataURI", "size", "productType", "description")})
    print("HLSP products:", len(prods), flush=True)
for coll in ("spitzer_seip", "spitzer_sha"):
    try:
        t = Irsa.query_sia(pos=(pos, 0.3 * u.deg), collection=coll)
        rows = []
        for r in t:
            rows.append({k: (str(r[k]) if k in t.colnames else "") for k in ("obs_id", "obs_title", "access_url", "access_estsize", "dataproduct_subtype", "energy_bandpassname", "instrument_name", "s_ra", "s_dec", "s_fov", "s_xel1", "s_xel2", "calib_level", "facility_name")})
        rep["irsa"][coll] = rows
        print(coll, len(rows), flush=True)
    except Exception as e:
        rep["irsa"][coll] = {"error": repr(e)}
        print(coll, "error", e, flush=True)
json.dump(rep, open(os.path.join(out, "probe.json"), "w"), indent=1)
