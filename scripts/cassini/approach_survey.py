# Surveys Cassini ISS narrow-angle frames of Saturn on the 2004 approach in OPUS: finds red, green and blue sets and saves a sample of the products each frame has.
# usage: approach_survey.py OUTDIR
import bisect, calendar, collections, csv, json, os, sys, time, urllib.parse, urllib.request

OPUS = "https://opus.pds-rings.seti.org/opus/api"
UA = "mudarkroom (github.com/Raellabot/mudarkroom)"
WINDOW_MIN = 20


def get(url, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=180) as r:
                return r.read()
        except Exception as e:
            print("retry", url, e, flush=True)
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))


def getj(path, **q):
    return json.loads(get(f"{OPUS}/{path}?{urllib.parse.urlencode(q)}"))


def save(out, name, obj):
    with open(os.path.join(out, name), "w") as f:
        json.dump(obj, f, indent=1)


def ts(s):
    try:
        return calendar.timegm(time.strptime(str(s)[:19], "%Y-%m-%dT%H:%M:%S"))
    except ValueError:
        return None


# Names the color in a Cassini filter pair (one wheel is usually clear).
def color(v):
    parts = [p.strip().upper() for p in str(v).replace("/", ",").split(",")]
    other = [p for p in parts if p not in ("CL1", "CL2", "")]
    if other == ["RED"]:
        return "RED"
    if other == ["GRN"]:
        return "GRN"
    if other == ["BL1"]:
        return "BL1"
    return "+".join(other) or "CLEAR"


def main(out):
    os.makedirs(out, exist_ok=True)
    q = dict(instrument="Cassini ISS", target="Saturn", time1="2003-10-01", time2="2004-07-01")
    d = getj("data.json", **q, limit="1")
    oid = d["page"][0][0]
    meta = getj(f"metadata/{oid}.json")
    save(out, "sample-metadata.json", meta)
    save(out, "sample-files.json", getj(f"files/{oid}.json"))
    flat = {k: c for c, vals in meta.items() if isinstance(vals, dict) for k in vals}
    fcol = [k for k, c in flat.items() if "filter" in k.lower() and "cassini" in c.lower()]
    ccol = [k for k, c in flat.items() if "camera" in k.lower() and "cassini" in c.lower()]
    print("filter fields", fcol, "camera fields", ccol, flush=True)
    cols = ["opusid", "time1", "target", "primaryfilespec"] + fcol[:2] + ccol[:1]
    rows = []
    while True:
        page = getj("data.json", **q, cols=",".join(cols), limit="1000", order="time1,opusid", startobs=str(len(rows) + 1))["page"]
        if not page:
            break
        rows += [dict(zip(cols, r)) for r in page]
    print("frames", len(rows), flush=True)
    for r in rows:
        r["f"] = color(",".join(str(r[c]) for c in fcol[:2]))
        r["t"] = ts(r["time1"])
        r["cam"] = str(r.get(ccol[0])) if ccol else "?"
    nac = sorted([r for r in rows if r["t"] and "narrow" in r["cam"].lower()], key=lambda r: r["t"])
    times, used, sets = [r["t"] for r in nac], set(), []
    for g in nac:
        if g["f"] != "GRN" or g["opusid"] in used:
            continue
        near = {}
        for j in range(bisect.bisect_left(times, g["t"] - WINDOW_MIN * 60), bisect.bisect_right(times, g["t"] + WINDOW_MIN * 60)):
            r = nac[j]
            if r["f"] in ("RED", "BL1") and r["opusid"] not in used and (r["f"] not in near or abs(r["t"] - g["t"]) < abs(near[r["f"]]["t"] - g["t"])):
                near[r["f"]] = r
        if len(near) == 2:
            used.update([g["opusid"], near["RED"]["opusid"], near["BL1"]["opusid"]])
            sets.append(dict(time=g["time1"], red=near["RED"]["opusid"], green=g["opusid"], blue=near["BL1"]["opusid"],
                             span_s=max(abs(near[f]["t"] - g["t"]) for f in near), green_file=g["primaryfilespec"]))
    report = dict(filter_fields=fcol, camera_fields=ccol, frames=len(rows), nac=len(nac),
                  by_camera=collections.Counter(r["cam"] for r in rows).most_common(),
                  nac_by_filter=collections.Counter(r["f"] for r in nac).most_common(30),
                  nac_by_month=collections.Counter(r["time1"][:7] for r in nac).most_common(),
                  sets=len(sets), sets_by_month=sorted(collections.Counter(s["time"][:7] for s in sets).items()),
                  sets_by_day=sorted(collections.Counter(s["time"][:10] for s in sets).items()))
    save(out, "report.json", report)
    if sets:
        save(out, "sample-set-files.json", getj(f"files/{sets[len(sets) // 2]['green']}.json"))
        with open(os.path.join(out, "sets.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(sets[0]))
            w.writeheader()
            w.writerows(sets)
    print(json.dumps({k: v for k, v in report.items() if k != "sets_by_day"}, indent=1), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
