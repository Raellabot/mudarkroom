# Surveys Voyager ISS frames in OPUS: counts per planet, target, camera and filter, finds color sets, and tries a few pilot color pictures.
# usage: survey.py OUTDIR
import bisect, calendar, collections, csv, json, os, re, sys, time, urllib.parse, urllib.request
import numpy as np
from PIL import Image

OPUS = "https://opus.pds-rings.seti.org/opus/api"
HOLD = "https://pds-rings.seti.org/holdings/volumes/"
UA = "mudarkroom (github.com/Raellabot/mudarkroom)"
PLANETS = ["Jupiter", "Saturn", "Uranus", "Neptune"]
WINDOW_MIN = 15
COLORS = ("ORANGE", "GREEN", "BLUE", "VIOLET")


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


def norm_filter(v):
    v = str(v).upper()
    for c in COLORS + ("CLEAR", "UV", "CH4", "SODIUM", "NAD"):
        if c in v:
            return c
    return v


# Finds the Voyager-specific filter and camera fields from one observation's full metadata.
def discover(out):
    d = getj("data.json", instrument="Voyager ISS", limit="1")
    save(out, "sample-data.json", d)
    oid = d["page"][0][0]
    meta = getj(f"metadata/{oid}.json")
    save(out, "sample-metadata.json", meta)
    flat = {}
    for cat, vals in meta.items():
        if isinstance(vals, dict):
            for k, v in vals.items():
                flat[k] = (cat, v)
    filt = [k for k, (c, v) in flat.items() if "filter" in k.lower() and "voyager" in c.lower()] or [k for k in flat if "filter" in k.lower()]
    cam = [k for k, (c, v) in flat.items() if "camera" in k.lower() and "voyager" in c.lower()] or [k for k in flat if "camera" in k.lower()]
    print("filter fields:", filt, "camera fields:", cam, flush=True)
    return filt[0], (cam[0] if cam else None)


def rows_for(planet, cols):
    q = dict(instrument="Voyager ISS", planet=planet, cols=",".join(cols), limit="1000", order="time1,opusid")
    d = getj("data.json", **q)
    page, avail = d["page"], int(d.get("available") or 0)
    rows = list(page)
    while len(rows) < avail:
        more = getj("data.json", **q, startobs=str(len(rows) + 1))["page"]
        if not more:
            break
        rows += more
    print(planet, len(rows), "of", avail, flush=True)
    return [dict(zip(cols, r)) for r in rows]


# Orange plus green plus violet or blue of one spacecraft, camera and target, anchored on the green frame.
def find_sets(rows, fcol, ccol):
    groups = collections.defaultdict(list)
    for r in rows:
        r["f"] = norm_filter(r[fcol])
        r["t"] = ts(r["time1"])
        r["sc"] = r["opusid"].split("-")[2] if r["opusid"].count("-") >= 3 else "?"
        if r["t"] is None:
            continue
        groups[(r["sc"], r.get(ccol), r["target"])].append(r)
    sets = []
    for key, g in groups.items():
        g.sort(key=lambda r: r["t"])
        used, times = set(), [r["t"] for r in g]
        for i, gr in enumerate(g):
            if gr["f"] != "GREEN" or gr["opusid"] in used:
                continue
            near = {}
            for j in range(bisect.bisect_left(times, gr["t"] - WINDOW_MIN * 60), bisect.bisect_right(times, gr["t"] + WINDOW_MIN * 60)):
                r = g[j]
                if r["opusid"] in used or r["f"] not in ("ORANGE", "BLUE", "VIOLET"):
                    continue
                dt = abs(r["t"] - gr["t"])
                if dt <= WINDOW_MIN * 60 and (r["f"] not in near or dt < near[r["f"]][0]):
                    near[r["f"]] = (dt, r)
            if "ORANGE" in near and ("VIOLET" in near or "BLUE" in near):
                members = [gr] + [v[1] for v in near.values()]
                used.update(m["opusid"] for m in members)
                span = max(m["t"] for m in members) - min(m["t"] for m in members)
                sets.append(dict(spacecraft=key[0], camera=key[1], target=key[2], time=gr["time1"], span_s=span,
                                 filters="".join(sorted(m["f"][0] for m in members)),
                                 green=gr["opusid"], orange=near["ORANGE"][1]["opusid"],
                                 violet=near.get("VIOLET", (0, {"opusid": ""}))[1]["opusid"], blue=near.get("BLUE", (0, {"opusid": ""}))[1]["opusid"],
                                 green_file=gr["primaryfilespec"]))
    return sets


def parse_vicar(b):
    m = re.match(rb"LBLSIZE=(\d+)", b)
    lbl = int(m.group(1))
    text = b[:lbl].decode("latin1")
    items = {}
    for k, v in re.findall(r"(\w+)=('(?:[^']|'')*'|\([^)]*\)|[^\s]+)", text):
        items.setdefault(k, v.strip("'"))
    ns, nl, nbb, nlb = int(items["NS"]), int(items["NL"]), int(items.get("NBB", 0)), int(items.get("NLB", 0))
    dt = {"BYTE": "u1", "HALF": "i2", "WORD": "i2", "FULL": "i4", "REAL": "f4"}[items["FORMAT"]]
    little = items.get("INTFMT", "LOW").upper() == "LOW" if dt != "f4" else items.get("REALFMT", "RIEEE").upper() == "RIEEE"
    dt = ("<" if little else ">") + dt if dt != "u1" else dt
    size = np.dtype(dt).itemsize
    rec = int(items.get("RECSIZE", nbb + ns * size))
    raw = np.frombuffer(b, dtype="u1", count=nl * rec, offset=lbl + nlb * rec).reshape(nl, rec)
    return items, text, np.ascontiguousarray(raw[:, nbb:nbb + ns * size]).view(dt).reshape(nl, ns).astype("float32")


def product_url(spec, kind):
    vol = spec.split("/")[0]
    base = re.sub(r"_(RAW|CLEANED|CALIB|GEOMED)\.(IMG|LBL)$", "", spec)
    return f"{HOLD}VGISS_{vol[6]}xxx/{base}_{kind}.IMG"


def shift_of(ref, mov):
    f = np.fft.fft2(ref - ref.mean()) * np.conj(np.fft.fft2(mov - mov.mean()))
    c = np.real(np.fft.ifft2(f / (np.abs(f) + 1e-6)))
    y, x = np.unravel_index(np.argmax(c), c.shape)
    if y > c.shape[0] // 2: y -= c.shape[0]
    if x > c.shape[1] // 2: x -= c.shape[1]
    return int(y), int(x)


# A quick orange, green, violet-or-blue picture from the GEOMED frames, each stretched on its own.
def pilot(out, s, rowsby, n):
    arrs = {}
    for f in ("orange", "green", "violet" if s["violet"] else "blue"):
        spec = rowsby[s[f]]["primaryfilespec"]
        url = product_url(spec, "GEOMED")
        items, text, a = parse_vicar(get(url))
        if n == 0 and f == "green":
            open(os.path.join(out, "sample-geomed-label.txt"), "w").write(text)
            raw = get(product_url(spec, "CALIB"))
            open(os.path.join(out, "sample-calib-label.txt"), "w").write(parse_vicar(raw)[1])
            try:
                open(os.path.join(out, "sample-pds-label.txt"), "wb").write(get(url.replace(".IMG", ".LBL")))
            except Exception as e:
                print("no .LBL:", e)
        print(s["target"], f, a.shape, np.percentile(a, [0, 1, 50, 99, 100]).round(3).tolist(), flush=True)
        arrs[f] = a
    keys = list(arrs)
    g = arrs["green"]
    chans = []
    for k in keys:
        a = arrs[k] if k == "green" else np.roll(arrs[k], shift_of(g, arrs[k]), axis=(0, 1))
        lo, hi = np.percentile(a, [1, 99.8])
        chans.append(np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1))
    rgb = (np.stack(chans, -1) ** 0.6 * 255).astype("uint8")
    Image.fromarray(rgb).save(os.path.join(out, f"pilot-{n}-{re.sub('[^a-z0-9]+', '-', s['target'].lower())}.jpg"), quality=90)


def main(out):
    os.makedirs(out, exist_ok=True)
    fcol, ccol = discover(out)
    report = {"filter_field": fcol, "camera_field": ccol, "total": getj("meta/result_count.json", instrument="Voyager ISS")}
    report["year_1990"] = getj("meta/result_count.json", instrument="Voyager ISS", time1="1990-01-01", time2="1990-12-31")
    cols = [c for c in ("opusid", "time1", "target", "duration", "primaryfilespec", fcol, ccol) if c]
    allsets, rowsby, per = [], {}, {}
    for p in PLANETS:
        rows = rows_for(p, cols)
        for r in rows:
            rowsby[r["opusid"]] = r
        sets = find_sets(rows, fcol, ccol)
        allsets += sets
        per[p] = dict(frames=len(rows),
                      by_filter=collections.Counter(r["f"] for r in rows).most_common(),
                      by_camera=collections.Counter(str(r.get(ccol)) for r in rows).most_common(),
                      by_spacecraft=collections.Counter(r["sc"] for r in rows).most_common(),
                      frames_by_target=collections.Counter(r["target"] for r in rows).most_common(40),
                      sets=len(sets), sets_by_target=collections.Counter(s["target"] for s in sets).most_common(40),
                      sets_by_span=collections.Counter("<2 min" if s["span_s"] < 120 else "<5 min" if s["span_s"] < 300 else "<15 min" for s in sets).most_common(),
                      sets_by_camera=collections.Counter(str(s["camera"]) for s in sets).most_common(),
                      sets_by_filters=collections.Counter(s["filters"] for s in sets).most_common())
        print(json.dumps(per[p])[:1500], flush=True)
    report["planets"] = per
    save(out, "report.json", report)
    with open(os.path.join(out, "sets.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(allsets[0]))
        w.writeheader()
        w.writerows(allsets)
    picks = []
    for p in PLANETS:
        ps = [s for s in allsets if s["target"].lower() == p.lower()]
        if ps:
            picks.append(ps[len(ps) // 2])
    for n, s in enumerate(picks):
        try:
            pilot(out, s, rowsby, n)
        except Exception as e:
            print("pilot failed", s["target"], repr(e), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
