# Writes the MAST rows of one program to rows.json.gz and groups.txt; usage: find_rows.py OUTDIR COLLECTION PROPOSAL_ID [TARGET]
import collections, gzip, json, os, sys, time, urllib.parse, urllib.request

API = "https://mast.stsci.edu/api/v0/invoke"
# Your name and contact URL, sent to MAST as the User-Agent: edit this line or set the CONTACT_INFO environment variable.
UA = os.environ.get("CONTACT_INFO") or "CONTACT-INFO"
COLUMNS = ("obs_collection,obs_id,target_name,target_classification,instrument_name,filters,t_min,t_exptime,s_ra,s_dec,em_min,em_max,"
           "obs_title,proposal_id,proposal_pi,jpegURL,dataURL,dataRights,t_obs_release,calib_level,dataproduct_type,intentType")


def ask(filters):
    rows, page = [], 1
    while True:
        req = {"service": "Mast.Caom.Filtered", "format": "json", "params": {"columns": COLUMNS, "filters": filters}, "pagesize": 2000, "page": page}
        url = API + "?request=" + urllib.parse.quote(json.dumps(req, separators=(",", ":")))
        start, d = time.time(), None
        while time.time() - start < 600:
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=150) as r:
                    d = json.loads(r.read())
            except Exception as e:  # noqa: BLE001
                print(f"  {str(e)[:160]}; asking again", flush=True)
                time.sleep(10)
                continue
            if d.get("status") == "COMPLETE":
                break
            print(f"  MAST says {d.get('status')}; asking again", flush=True)
            d = None
            time.sleep(10)
        if d is None:
            raise RuntimeError("no answer from MAST within 600 s")
        rows += d["data"]
        pg = d.get("paging", {})
        print(f"  page {page}/{pg.get('pagesFiltered')} rows so far {len(rows)} of {pg.get('rowsFiltered')}", flush=True)
        if page >= pg.get("pagesFiltered", 1) or not d["data"]:
            return rows
        page += 1


def main():
    out, collection, proposal = sys.argv[1:4]
    target = sys.argv[4] if len(sys.argv) > 4 else None
    filters = [
        {"paramName": "obs_collection", "values": [collection]},
        {"paramName": "proposal_id", "values": [proposal]},
        {"paramName": "intentType", "values": ["science"]},
        {"paramName": "dataRights", "values": ["PUBLIC"]},
    ]
    if target:
        filters.append({"paramName": "target_name", "values": [target]})
    rows = [r for r in ask(filters) if r.get("dataproduct_type") == "image"]
    os.makedirs(out, exist_ok=True)
    with gzip.open(os.path.join(out, "rows.json.gz"), "wt") as f:
        json.dump({"complete": True, "rows": rows}, f)
    g = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        g[(r["obs_collection"], r["target_name"], r["proposal_id"], r["instrument_name"], r["calib_level"])][r.get("filters")] += 1
    with open(os.path.join(out, "groups.txt"), "w") as f:
        for k, c in sorted(g.items(), key=lambda kv: -sum(kv[1].values())):
            line = f"{'|'.join(map(str, k))}  rows={sum(c.values())}  filters={sorted(map(str, c))}"
            print(line)
            f.write(line + "\n")


if __name__ == "__main__":
    main()
