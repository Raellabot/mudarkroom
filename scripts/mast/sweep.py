# Counts the public Hubble and Webb image sets that could become pictures, without downloading any pictures; usage: sweep.py OUTDIR
import collections, csv, json, os, statistics, sys, time, urllib.parse, urllib.request

API = "https://mast.stsci.edu/api/v0/invoke"
UA = os.environ.get("CONTACT_INFO") or "CONTACT-INFO"
COLLECTIONS = os.environ.get("COLLECTIONS", "HST JWST HAP").split()
COLUMNS = "obs_collection,obs_id,target_name,instrument_name,filters,proposal_id,obs_title,s_ra,s_dec,t_exptime"
SKIP_TARGETS = {"ANY", "BIAS", "DARK", "NONE", "CALIBRATION", "DARK-EARTH-CALIB", "EARTH-CALIB", "FLAT", "WAVE", "UNKNOWN", ""}
PAGE = 25000


def ask(params, page):
    req = {"service": "Mast.Caom.Filtered", "format": "json", "params": {"columns": COLUMNS, "filters": params}, "pagesize": PAGE, "page": page}
    url = API + "?request=" + urllib.parse.quote(json.dumps(req, separators=(",", ":")))
    start = time.time()
    while time.time() - start < 900:
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=300) as r:
                d = json.loads(r.read())
            if d.get("status") == "COMPLETE":
                return d
            print(f"  MAST says {d.get('status')}; asking again", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  {str(e)[:160]}; asking again", flush=True)
        time.sleep(15)
    raise RuntimeError("MAST did not answer")


# The same filter rule the exporter uses: no combined or detection filters.
def usable(f):
    return bool(f) and ";" not in f and "detection" not in f.lower()


def sweep(collection, rows_out):
    base = [
        {"paramName": "obs_collection", "values": [collection]},
        {"paramName": "calib_level", "values": [3]},
        {"paramName": "dataproduct_type", "values": ["image"]},
        {"paramName": "intentType", "values": ["science"]},
        {"paramName": "dataRights", "values": ["PUBLIC"]},
    ]
    page = 1
    while True:
        d = ask(base, page)
        rows_out.extend(d["data"])
        paging = d.get("paging") or {}
        print(f"{collection}: page {page} of {paging.get('pagesFiltered')}, {len(rows_out)} rows so far", flush=True)
        if page >= (paging.get("pagesFiltered") or 1):
            break
        page += 1


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    groups = {}
    totals = collections.Counter()
    for collection in COLLECTIONS:
        rows = []
        try:
            sweep(collection, rows)
        except Exception as e:  # noqa: BLE001
            print(f"{collection}: failed: {e}", flush=True)
        totals[collection] = len(rows)
        for r in rows:
            name = (r.get("target_name") or "").strip()
            if name.upper() in SKIP_TARGETS or not usable(r.get("filters")):
                continue
            g = groups.setdefault((collection, str(r.get("proposal_id")), name), {"filters": set(), "inst": collections.Counter(), "ra": [], "dec": [], "title": r.get("obs_title") or "", "exp": 0.0})
            g["filters"].add(r["filters"])
            g["inst"][r.get("instrument_name")] += 1
            if r.get("s_ra") is not None and r.get("s_dec") is not None:
                g["ra"].append(r["s_ra"])
                g["dec"].append(r["s_dec"])
            g["exp"] += r.get("t_exptime") or 0.0
    table = []
    for (collection, proposal, name), g in groups.items():
        table.append({"collection": collection, "proposal": proposal, "target": name, "filters": len(g["filters"]), "filter_names": " ".join(sorted(g["filters"])), "instruments": "+".join(sorted(map(str, g["inst"]))), "ra": round(statistics.median(g["ra"]), 4) if g["ra"] else "", "dec": round(statistics.median(g["dec"]), 4) if g["dec"] else "", "exposure_s": round(g["exp"]), "title": g["title"][:80]})
    table.sort(key=lambda t: (-t["filters"], t["collection"], t["proposal"], t["target"]))
    with open(os.path.join(out, "sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()) if table else ["empty"])
        w.writeheader()
        w.writerows(table)
    lines = ["Rows returned per collection: " + ", ".join(f"{k} {v}" for k, v in totals.items()), ""]
    for collection in COLLECTIONS:
        mine = [t for t in table if t["collection"] == collection]
        lines.append(f"{collection}: {len(mine)} target groups with at least one usable filter")
        for need in (2, 3, 4, 5, 6):
            lines.append(f"  with {need}+ filters: {sum(1 for t in mine if t['filters'] >= need)}")
        proposals = {t["proposal"] for t in mine if t["filters"] >= 3}
        lines.append(f"  programs with a 3+ filter target: {len(proposals)}")
        lines.append("")
    both = {(t["target"].upper()) for t in table if t["filters"] >= 3}
    lines.append(f"Distinct target names across everything with 3+ filters: {len(both)}")
    open(os.path.join(out, "summary.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
