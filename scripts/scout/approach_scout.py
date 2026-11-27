# Counts approach and departure frames for several missions (OPUS for New Horizons, Cassini, Galileo, Juno; PDS3 index tables for Dawn and MESSENGER) and sizes the target with JPL Horizons.
# usage: approach_scout.py OUTDIR [JOB,JOB...]
import bisect, calendar, collections, csv, json, os, re, sys, time, urllib.parse, urllib.request

OPUS = "https://opus.pds-rings.seti.org/opus/api"
HORIZONS = "https://ssd.jpl.nasa.gov/api/horizons.api"
UA = "mudarkroom (github.com/Raellabot/mudarkroom)"
DIAM = {"pluto": 2377, "charon": 1212, "jupiter": 142984, "vesta": 525, "ceres": 940, "earth": 12742, "moon": 3475}
# Pixel field of view in microradians for instruments sized with Horizons.
IFOV = {"dawn-fc": 93.7, "mdis-wac": 179.0, "mdis-nac": 25.0}

OPUS_JOBS = [
    dict(name="nh-pluto", instrument="New Horizons LORRI,New Horizons MVIC", time1="2015-01-01", time2="2015-08-01", bodies=["pluto", "charon"]),
    dict(name="nh-jupiter", instrument="New Horizons LORRI,New Horizons MVIC", time1="2006-09-01", time2="2007-06-30", bodies=["jupiter"]),
    dict(name="cassini-jupiter", instrument="Cassini ISS", time1="2000-10-01", time2="2001-03-31", bodies=["jupiter"]),
    dict(name="galileo-earth-1990", instrument="Galileo SSI", time1="1990-12-01", time2="1991-01-01", bodies=["earth", "moon"]),
    dict(name="galileo-earth-1992", instrument="Galileo SSI", time1="1992-12-01", time2="1993-01-01", bodies=["earth", "moon"]),
]
INDEX_JOBS = [
    dict(name="dawn-vesta", roots=["https://sbnarchive.psi.edu/pds3/dawn/fc/"], vol=r"(?i)^dwnv.*fc2_1a$", time1="2011-05-01", time2="2011-07-17", body="vesta", horizons="4;", sc="-203", cam="dawn-fc"),
    dict(name="dawn-ceres", roots=["https://sbnarchive.psi.edu/pds3/dawn/fc/"], vol=r"(?i)^dwnc.*fc2_1a$", time1="2014-12-01", time2="2015-05-31", body="ceres", horizons="1;", sc="-203", cam="dawn-fc"),
    dict(name="messenger-earth", roots=["https://planetarydata.jpl.nasa.gov/img/data/messenger/", "https://pds-geosciences.wustl.edu/messenger/mess-e_v_h-mdis-2-edr-rawdata-v1/", "https://pds-imaging.jpl.nasa.gov/data/messenger/"], vol=r"(?i)(mdis-2-edr|messmdis_1001|msgrmds_1001)", time1="2005-07-25", time2="2005-08-10", body="earth", horizons="399", sc="-236", cam="mdis-wac"),
]


def get(url, tries=4, raw=False):
    for i in range(tries):
        try:
            r = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=300)
            return r if raw else r.read()
        except Exception as e:
            print("retry", url, e, flush=True)
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))


def getj(path, **q):
    return json.loads(get(f"{OPUS}/{path}?{urllib.parse.urlencode(q)}"))


def save(out, name, obj):
    with open(os.path.join(out, name), "w") as f:
        json.dump(obj, f, indent=1, default=str)


def ts(s):
    s = str(s).strip().strip('"').replace("Z", "")
    m = re.match(r"^(\d{4})-(\d{3})T(\d\d):(\d\d):(\d\d)", s)
    if m:
        y, j, hh, mm, ss = map(int, m.groups())
        return calendar.timegm((y, 1, 1, hh, mm, ss)) + (j - 1) * 86400
    for fmt, n in (("%Y-%m-%dT%H:%M:%S", 19), ("%Y-%m-%d %H:%M:%S", 19), ("%Y-%m-%d", 10)):
        try:
            return calendar.timegm(time.strptime(s[:n], fmt))
        except ValueError:
            pass
    return None


def day(t):
    return time.strftime("%Y-%m-%d", time.gmtime(t))


# Longest stretches with no frames, in days.
def gaps(ts_list, n=5):
    t = sorted(ts_list)
    g = sorted(((b - a) / 86400, day(a), day(b)) for a, b in zip(t, t[1:]))
    return [dict(days=round(d, 2), after=a, before=b) for d, a, b in g[-n:][::-1]]


# Names the color band of a frame from its wavelength range in microns.
def band(w1, w2):
    try:
        a, b = float(w1) * 1000, float(w2) * 1000
    except (TypeError, ValueError):
        return "?"
    if b - a > 250:
        return "pan"
    c = (a + b) / 2
    return "violet" if c < 450 else "blue" if c < 500 else "green" if c < 590 else "red" if c < 700 else "nir" if c < 1100 else "ir"


# Counts sets with blue, green and red frames from one camera taken within the window.
def color_sets(frames, window_min):
    frames = sorted(frames, key=lambda r: r["t"])
    used, sets = set(), []
    by_cam = collections.defaultdict(list)
    for r in frames:
        by_cam[r["cam"]].append(r)
    for cam, fr in by_cam.items():
        times = [r["t"] for r in fr]
        for g in fr:
            if g["band"] != "green" or g["id"] in used:
                continue
            near = {}
            for j in range(bisect.bisect_left(times, g["t"] - window_min * 60), bisect.bisect_right(times, g["t"] + window_min * 60)):
                r = fr[j]
                if r["band"] in ("blue", "red", "violet") and r["id"] not in used:
                    k = "blue" if r["band"] == "violet" else r["band"]
                    if k not in near or abs(r["t"] - g["t"]) < abs(near[k]["t"] - g["t"]):
                        near[k] = r
            if len(near) == 2:
                used.update([g["id"], near["blue"]["id"], near["red"]["id"]])
                sets.append(dict(t=g["t"], time=day(g["t"]), cam=cam, size_px=g.get("size_px")))
    return sets


def size_summary(rows):
    s = [(r["t"], r["size_px"]) for r in rows if r.get("size_px")]
    if not s:
        return None
    s.sort()
    by_day = collections.defaultdict(list)
    for t, v in s:
        by_day[day(t)].append(v)
    return dict(first=round(s[0][1], 1), last=round(s[-1][1], 1), max=round(max(v for _, v in s), 1), by_day={d: round(max(v), 1) for d, v in sorted(by_day.items())})


def opus_job(out, job):
    q = dict(instrument=job["instrument"], time1=job["time1"], time2=job["time2"])
    d = getj("data.json", **q, limit="1")
    if not d.get("page"):
        print(job["name"], "no frames", flush=True)
        return dict(name=job["name"], frames=0)
    meta = getj(f"metadata/{d['page'][0][0]}.json")
    flat = {k: c for c, vals in meta.items() if isinstance(vals, dict) for k in vals}
    fcol = [k for k, c in flat.items() if "filter" in k.lower() and "constraints" in c.lower() and "general" not in c.lower()][:2]
    ccol = [k for k, c in flat.items() if "camera" in k.lower() and "constraints" in c.lower()][:1]
    geo = []
    for b in job["bodies"]:
        geo += [f"SURFACEGEO{b}_centerresolution1", f"SURFACEGEO{b}_centerdistance1"]
    base = ["opusid", "time1", "target", "instrument", "primaryfilespec", "wavelength1", "wavelength2", "observationduration"]
    cols = base + fcol + ccol + geo
    try:
        getj("data.json", **q, cols=",".join(cols), limit="1")
    except Exception as e:
        print("geometry columns refused, trying without", e, flush=True)
        cols = base + fcol + ccol
    rows = []
    while True:
        page = getj("data.json", **q, cols=",".join(cols), limit="1000", order="time1,opusid", startobs=str(len(rows) + 1))["page"]
        if not page:
            break
        rows += [dict(zip(cols, r)) for r in page]
        print(job["name"], len(rows), flush=True)
    for r in rows:
        r["id"], r["t"] = r["opusid"], ts(r["time1"])
        r["band"] = band(r["wavelength1"], r["wavelength2"])
        r["filt"] = ",".join(str(r[c]) for c in fcol)
        r["cam"] = f"{r['instrument']}|{r[ccol[0]]}" if ccol else r["instrument"]
        r["prefix"] = os.path.basename(str(r["primaryfilespec"]))[:3].lower()
        sz = None
        for b in job["bodies"]:
            try:
                res = float(r.get(f"SURFACEGEO{b}_centerresolution1"))
                if res > 0:
                    sz = max(sz or 0, DIAM[b] / res)
            except (TypeError, ValueError):
                pass
        r["size_px"] = sz
    rows = [r for r in rows if r["t"]]
    with open(os.path.join(out, f"{job['name']}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    on_body = [r for r in rows if r["size_px"]]
    sets = color_sets(on_body or rows, 20)
    rep = dict(name=job["name"], frames=len(rows), frames_with_body_geometry=len(on_body), filter_fields=fcol, camera_fields=ccol, columns=cols,
               by_target=collections.Counter(r["target"] for r in rows).most_common(15), by_camera=collections.Counter(r["cam"] for r in rows).most_common(),
               by_band=collections.Counter(r["band"] for r in rows).most_common(), by_filter=collections.Counter(r["filt"] for r in on_body or rows).most_common(25),
               by_prefix=collections.Counter(f"{r['cam']}|{r['prefix']}|{r['band']}" for r in rows).most_common(25),
               on_body_by_day=sorted(collections.Counter(day(r["t"]) for r in on_body).items()), on_body_by_month=sorted(collections.Counter(day(r["t"])[:7] for r in on_body).items()),
               on_body_gaps=gaps([r["t"] for r in on_body]), size=size_summary(on_body), color_sets=len(sets),
               color_sets_by_month=sorted(collections.Counter(s["time"][:7] for s in sets).items()), color_sets_by_cam=collections.Counter(s["cam"] for s in sets).most_common(),
               color_set_size=size_summary([dict(t=s["t"], size_px=s["size_px"]) for s in sets]), sample_files=[r["primaryfilespec"] for r in rows[:: max(1, len(rows) // 12)]])
    return rep


def listing(url):
    html = get(url).decode("utf-8", "replace")
    if "href" not in html.lower():
        print("no links at", url, html[:600].replace("\n", " "), flush=True)
    names = []
    for h in re.findall(r'href="([^"?#]+)"', html):
        h = urllib.parse.unquote(h).rstrip()
        if h.startswith(("http", "/", "..", "mailto")):
            h = h.rstrip("/").split("/")[-1] + ("/" if h.endswith("/") else "")
        if h and h not in names and not h.startswith(".."):
            names.append(h)
    return names


# Reads the column layout of a PDS3 index label.
def parse_label(txt):
    cols, cur, table, recbytes = [], None, None, None
    for line in txt.splitlines():
        m = re.match(r"\s*([\^A-Z_0-9:]+)\s*=\s*(.*?)\s*$", line)
        if not m:
            continue
        k, v = m.group(1), m.group(2).strip().strip('"')
        if k == "RECORD_BYTES":
            recbytes = int(re.sub(r"\D", "", v) or 0)
        elif k.startswith("^") and "TABLE" in k:
            table = re.sub(r"[()\"]", "", v).split(",")[0].strip()
        elif k == "OBJECT" and v == "COLUMN":
            cur = {}
        elif k == "END_OBJECT" and v == "COLUMN" and cur is not None:
            cols.append(cur)
            cur = None
        elif cur is not None and k in ("NAME", "START_BYTE", "BYTES", "ITEMS", "ITEM_BYTES"):
            cur[k] = v if k == "NAME" else int(re.sub(r"\D", "", v) or 0)
    return table, recbytes, cols


# Walks archive directory listings, following folders whose names match FOLLOW, and collects the index labels it finds.
def find_indexes(url, follow, depth=0, found=None):
    found = [] if found is None else found
    try:
        names = listing(url)
    except Exception as e:
        print("cannot list", url, e, flush=True)
        return found
    print("listed", url, len(names), names[:60], flush=True)
    for n in names:
        b = n.rstrip("/")
        if b.lower() == "index":
            try:
                for m in listing(url + b + "/"):
                    if m.lower().endswith(".lbl") and "index" in m.lower() and "cum" not in m.lower() and "geom" not in m.lower():
                        found.append(url + b + "/" + m)
            except Exception as e:
                print("cannot list index", url, e, flush=True)
        elif depth < 2 and re.search(follow, b):
            find_indexes(url + b + "/", follow, depth + 1, found)
    return found


def index_job(out, job):
    labels = []
    for root in job["roots"]:
        labels = find_indexes(root, job["vol"])
        if labels:
            break
    print(job["name"], "index labels", labels, flush=True)
    t1, t2 = ts(job["time1"]), ts(job["time2"])
    rows, columns = [], None
    for lbl in labels:
        table, recbytes, cols = parse_label(get(lbl).decode("latin-1"))
        if not cols:
            print("no columns in", lbl, flush=True)
            continue
        columns = [c["NAME"] for c in cols]
        tab = lbl.rsplit("/", 1)[0] + "/" + (table or os.path.basename(lbl)[:-4] + ".TAB")
        tcol = next((c for c in cols if c["NAME"] in ("START_TIME", "IMAGE_TIME", "SPACECRAFT_CLOCK_START_TIME") and "CLOCK" not in c["NAME"]), None) or next(c for c in cols if "TIME" in c["NAME"])
        print("reading", tab, "time column", tcol["NAME"], flush=True)
        n = 0
        try:
            stream = get(tab, raw=True)
        except Exception:
            stream = get(tab.replace(".TAB", ".tab"), raw=True)
        for line in stream:
            n += 1
            line = line.decode("latin-1")
            rec = {c["NAME"]: line[c["START_BYTE"] - 1:c["START_BYTE"] - 1 + c["BYTES"]].strip().strip('"').strip() for c in cols if "START_BYTE" in c}
            t = ts(rec[tcol["NAME"]])
            if t and t1 <= t < t2:
                rec["t"] = t
                rows.append(rec)
        print(lbl, "lines", n, "in window", len(rows), flush=True)
    if not rows:
        return dict(name=job["name"], frames=0, labels=labels)
    keys = list(rows[0])
    fkey = next((k for k in keys if "FILTER" in k and "NAME" in k), None) or next((k for k in keys if "FILTER" in k), None)
    tkey = next((k for k in keys if k == "TARGET_NAME"), None)
    ikey = next((k for k in keys if k in ("INSTRUMENT_ID", "INSTRUMENT_NAME")), None)
    with open(os.path.join(out, f"{job['name']}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    sizes = horizons_sizes(job, sorted({day(r["t"]) for r in rows}))
    for r in rows:
        r["size_px"] = sizes.get(day(r["t"]))
    by_inst = collections.Counter(r.get(ikey, "?") for r in rows)
    rep = dict(name=job["name"], frames=len(rows), labels=labels, columns=columns, filter_key=fkey, target_key=tkey, instrument_key=ikey,
               by_instrument=by_inst.most_common(), by_target=collections.Counter(r.get(tkey, "?") for r in rows).most_common(15),
               by_filter=collections.Counter(f"{r.get(ikey, '?')}|{r.get(fkey, '?')}" for r in rows).most_common(30),
               by_day=sorted(collections.Counter(day(r["t"]) for r in rows).items()), gaps=gaps([r["t"] for r in rows]),
               size_px_by_day_for_camera=dict(camera=job["cam"], ifov_urad=IFOV[job["cam"]], by_day=sizes), sample_rows=rows[:: max(1, len(rows) // 8)][:8])
    return rep


# Target size in pixels per day from the spacecraft's range (JPL Horizons, quantity 20).
def horizons_sizes(job, days):
    out = {}
    for i in range(0, len(days), 40):
        part = days[i:i + 40]
        q = {"format": "json", "COMMAND": f"'{job['horizons']}'", "OBJ_DATA": "NO", "MAKE_EPHEM": "YES", "EPHEM_TYPE": "OBSERVER", "CENTER": f"'500@{job['sc']}'",
             "TLIST": "'" + " ".join(f"{d} 12:00" for d in part) + "'", "TIME_TYPE": "UT", "QUANTITIES": "'20'", "CSV_FORMAT": "YES"}
        try:
            txt = json.load(urllib.request.urlopen(HORIZONS + "?" + urllib.parse.urlencode(q), timeout=120))["result"]
        except Exception as e:
            print("horizons failed", job["name"], e, flush=True)
            continue
        if "$$SOE" not in txt:
            print("no ephemeris", job["name"], txt[:600], flush=True)
            continue
        for d, row in zip(part, txt.split("$$SOE")[1].split("$$EOE")[0].strip().splitlines()):
            c = [x.strip() for x in row.split(",")]
            try:
                au = float(next(x for x in c[2:] if re.match(r"^\d+\.\d+(E[+-]\d+)?$", x)))
                out[d] = round(DIAM[job["body"]] / (au * 1.495978707e8) / (IFOV[job["cam"]] * 1e-6), 1)
            except (StopIteration, ValueError):
                print("row?", row, flush=True)
        time.sleep(1)
    return out


def main(out):
    os.makedirs(out, exist_ok=True)
    report = {}
    only = set(sys.argv[2].split(",")) if len(sys.argv) > 2 and sys.argv[2] else None
    for job in OPUS_JOBS:
        if only and job["name"] not in only:
            continue
        try:
            report[job["name"]] = opus_job(out, job)
        except Exception as e:
            report[job["name"]] = dict(error=repr(e))
            print("FAILED", job["name"], e, flush=True)
        save(out, "report.json", report)
    for job in INDEX_JOBS:
        if only and job["name"] not in only:
            continue
        try:
            report[job["name"]] = index_job(out, job)
        except Exception as e:
            report[job["name"]] = dict(error=repr(e))
            print("FAILED", job["name"], e, flush=True)
        save(out, "report.json", report)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("frames", "frames_with_body_geometry", "color_sets", "by_band", "by_filter", "size", "error")} for k, v in report.items()}, indent=1, default=str)[:20000], flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
