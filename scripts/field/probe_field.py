# Lists the public Chandra, Hubble and Webb observations around one sky position without downloading pictures; usage: probe_field.py OUTDIR NAME RA_DEG DEC_DEG RADIUS_DEG
import collections, json, math, os, re, sys, time, urllib.parse, urllib.request
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mast"))
import find_rows

TAP = "https://cda.cfa.harvard.edu/cxctap/sync"
MAST_DOWNLOAD = "https://mast.stsci.edu/api/v0.1/Download/file?uri="
UA = find_rows.UA


# Reads a URL with up to 4 tries; returns the response body as text.
def get_text(url, timeout=120):
    last = None
    for attempt in range(1, 5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=timeout) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(5 * attempt)
    raise RuntimeError(f"{url[:120]}: {str(last)[:160]}")


# One TAP query in the archive's `text` format: `#` lines describe the columns, then a tab-separated header and rows.
def tap(adql):
    text = get_text(f"{TAP}?REQUEST=doQuery&LANG=ADQL&FORMAT=text&QUERY={urllib.parse.quote(adql)}")
    if re.search(r"<\s*(VOTABLE|html)\b", text[:500], re.I):
        raise RuntimeError("archive error: " + re.sub(r"\s+", " ", text[:300]))
    lines = [x.rstrip("\r") for x in text.split("\n") if x.strip() and not x.startswith("#")]
    if not lines:
        return []
    head = lines[0].split("\t")
    return [dict(zip(head, (c.strip() for c in x.split("\t")))) for x in lines[1:]]


# Total size in bytes of a file behind a URL, from a one-byte ranged request; -1 when it cannot be read.
def size_of(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Range": "bytes=0-0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            m = re.search(r"/(\d+)$", r.headers.get("Content-Range") or "")
            total = int(m.group(1)) if m else int(r.headers.get("Content-Length") or 0)
            r.close()
            return total
    except Exception:  # noqa: BLE001
        return -1


def sep_deg(ra1, dec1, ra2, dec2):
    a = math.radians
    c = math.sin(a(dec1)) * math.sin(a(dec2)) + math.cos(a(dec1)) * math.cos(a(dec2)) * math.cos(a(ra1 - ra2))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def write(out, lines):
    open(os.path.join(out, "report.txt"), "w").write("\n".join(lines) + "\n")


# Chandra: public observations whose pointing is within the radius, with their image products and the event file that goes with each.
def chandra(out, ra, dec, rad, lines):
    lines.append("== Chandra (Chandra Data Archive)")
    dra = rad / max(math.cos(math.radians(dec)), 0.05)
    where = (f"status = 'archived' AND ra BETWEEN {ra - dra} AND {ra + dra} AND dec BETWEEN {dec - rad} AND {dec + rad}")
    obs = tap("SELECT obsid, target_name, ra, dec, instrument, grating, proposal_number, start_date, public_avail_date, exposure_time, exposure_mode, event_count "
              f"FROM cxc.observation WHERE {where} ORDER BY obsid")
    obs = [o for o in obs if sep_deg(float(o["ra"]), float(o["dec"]), ra, dec) <= rad]
    lines.append(f"{len(obs)} public observations within {rad} deg")
    imgs = collections.defaultdict(list)
    if obs:
        ids = ",".join(o["obsid"] for o in obs)
        for r in tap(f"SELECT obsid, name, accref, imgfmt, filesize, bandpass, bandlo, bandhi, exptime FROM cxc.image WHERE obsid IN ({ids}) ORDER BY obsid, accref"):
            imgs[r["obsid"]].append(r)
    total_ks = 0.0
    for o in obs:
        ks = float(o["exposure_time"] or 0)
        total_ks += ks if o["grating"].upper() == "NONE" else 0
        line = (f"obsid {o['obsid']} | {o['target_name']} | {o['instrument']} | grating {o['grating']} | {o['exposure_mode']} | {ks:.1f} ks | "
                f"events {o['event_count']} | proposal {o['proposal_number']} | released {o['public_avail_date'][:10]} | {float(o['ra']):.4f} {float(o['dec']):.4f}")
        # The event file sits beside the images; its name carries the same pipeline version (N002 and so on).
        evt = ""
        for im in imgs.get(o["obsid"], []):
            m = re.search(r"filename=([^&]*?)(acisf|hrcf)(\d+)(N\d+)_(?:cntr|full)_img\d*\.fits", im["accref"])
            if m:
                base = im["accref"].split("filename=")[0] + "filename=" + m.group(1)
                evt = f"{base}{m.group(2)}{m.group(3)}{m.group(4)}_evt2.fits.gz"
                break
        if evt:
            sz = size_of(evt)
            line += f" | evt2 {sz / 1e6:.0f} MB {evt}" if sz >= 0 else f" | evt2 not found at {evt}"
        else:
            line += " | evt2 name unknown"
        lines.append(line)
        for im in imgs.get(o["obsid"], []):
            if "fits" in im["imgfmt"]:
                lines.append(f"    image {im['accref'].split('/')[-1]} | {im['bandlo']}-{im['bandhi']} keV | {int(float(im['filesize'] or 0)) / 1e6:.1f} MB")
        write(out, lines)
    lines.append(f"imaging exposure without a grating: {total_ks:.1f} ks in total")


# Hubble and Webb: public single-filter level-3 image mosaics within the radius, grouped by program, instrument and filter, with the largest file of each.
def mast(out, ra, dec, rad, lines):
    lines.append("== Hubble and Webb (MAST)")
    rows, page = [], 1
    while True:
        req = {"service": "Mast.Caom.Cone", "format": "json", "params": {"ra": ra, "dec": dec, "radius": rad}, "pagesize": 2000, "page": page}
        url = find_rows.API + "?request=" + urllib.parse.quote(json.dumps(req, separators=(",", ":")))
        d = json.loads(get_text(url, 180))
        if d.get("status") != "COMPLETE":
            time.sleep(10)
            continue
        rows += d["data"]
        pg = d.get("paging", {})
        if page >= pg.get("pagesFiltered", 1) or not d["data"]:
            break
        page += 1
    keep = [r for r in rows if r.get("obs_collection") in ("HST", "JWST") and r.get("dataproduct_type") == "image" and r.get("calib_level") == 3
            and r.get("intentType") == "science" and r.get("dataRights") == "PUBLIC" and (r.get("dataURL") or "").endswith(".fits")
            and r.get("filters") and ";" not in str(r["filters"])]
    lines.append(f"{len(rows)} rows from MAST, {len(keep)} public single-filter level-3 image mosaics")
    groups = collections.defaultdict(dict)
    for r in keep:
        key = (r["obs_collection"], str(r["proposal_id"]), r["instrument_name"], r["target_name"])
        f = r["filters"]
        if f not in groups[key] or (r.get("t_exptime") or 0) > (groups[key][f].get("t_exptime") or 0):
            groups[key][f] = r
    for key, flt in sorted(groups.items(), key=lambda kv: (kv[0][0], -len(kv[1]))):
        parts, total = [], 0
        for f, r in sorted(flt.items(), key=lambda kv: ((kv[1].get("em_min") or 0) + (kv[1].get("em_max") or 0))):
            sz = size_of(MAST_DOWNLOAD + urllib.parse.quote(r["dataURL"], safe=":/"))
            total += max(sz, 0)
            parts.append(f"{f} {sz / 1e9:.2f} GB {float(r.get('t_exptime') or 0):.0f} s" if sz >= 0 else f"{f} ?")
        lines.append(f"{key[0]}|{key[1]} | {key[3]} | {key[2]} | {len(flt)} filters | total {total / 1e9:.1f} GB | {', '.join(parts)}")
        write(out, lines)


def main():
    out, name = sys.argv[1], sys.argv[2]
    ra, dec, rad = (float(x) for x in sys.argv[3:6])
    os.makedirs(out, exist_ok=True)
    lines = [f"field {name}: ra {ra} dec {dec} radius {rad} deg"]
    for step in (chandra, mast):
        try:
            step(out, ra, dec, rad, lines)
        except Exception as e:  # noqa: BLE001
            lines.append(f"{step.__name__} failed: {e}")
        write(out, lines)
    print("\n".join(lines))


main()
