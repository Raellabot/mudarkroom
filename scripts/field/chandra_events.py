# Chandra Data Archive helpers: find a target's public event files, download them, and bin the photons onto a sky grid.
import collections, math, os, re, shutil, time, urllib.parse, urllib.request
from astropy.io import fits
import numpy as np
from scipy.ndimage import gaussian_filter

TAP = "https://cda.cfa.harvard.edu/cxctap/sync"
# Your name and contact URL, sent to the archives as the User-Agent: edit this line or set the CONTACT_INFO environment variable.
UA = os.environ.get("CONTACT_INFO") or "CONTACT-INFO"


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


# The event file's address from the address of one of the observation's images (same folder, same pipeline version).
def event_url(image_accref):
    m = re.search(r"filename=([^&]*?)(acisf|hrcf)(\d+)(N\d+)_(?:cntr|full)_img\d*\.fits", image_accref)
    if not m:
        return None
    return f"{image_accref.split('filename=')[0]}filename={m.group(1)}{m.group(2)}{m.group(3)}{m.group(4)}_evt2.fits.gz"


# The public imaging observations of a target (no grating), each with its exposure in ks and event file address, in observation-id order.
def event_files(target, instrument):
    obs = tap("SELECT obsid, target_name, instrument, grating, exposure_time FROM cxc.observation "
              f"WHERE status = 'archived' AND target_name = '{target}' AND instrument = '{instrument}' ORDER BY obsid")
    obs = [o for o in obs if o["grating"].upper() == "NONE"]
    if not obs:
        return []
    ids = ",".join(o["obsid"] for o in obs)
    imgs = collections.defaultdict(list)
    for r in tap(f"SELECT obsid, accref FROM cxc.image WHERE obsid IN ({ids}) ORDER BY obsid, accref"):
        imgs[r["obsid"]].append(r["accref"])
    out = []
    for o in obs:
        urls = [u for u in (event_url(a) for a in imgs.get(o["obsid"], [])) if u]
        if urls:
            out.append({"obsid": o["obsid"], "exposure_ks": float(o["exposure_time"] or 0), "url": urls[0]})
    return out


# Streams a file to `dest` (a `.part` file until complete), asking again a few times when the connection drops.
def fetch(url, dest):
    last = None
    for attempt in range(1, 5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=300) as r, open(dest + ".part", "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            with open(dest + ".part", "rb") as f:
                head = f.read(6)
            if not (head[:2] == b"\x1f\x8b" or head == b"SIMPLE"):
                raise RuntimeError(f"not a FITS file ({os.path.getsize(dest + '.part')} bytes, starts {head!r})")
            os.replace(dest + ".part", dest)
            return
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(10 * attempt)
    raise RuntimeError(f"download failed for {url[:120]}: {str(last)[:160]}")


# Sky positions (degrees) of events from their tangent-plane pixel positions, using the table's own WCS keywords (TCTYPn, TCRVLn, TCRPXn, TCDLTn).
def sky_to_radec(x, y, hdr, xcol, ycol):
    ra0, dec0 = math.radians(hdr[f"TCRVL{xcol}"]), math.radians(hdr[f"TCRVL{ycol}"])
    xi = np.radians((x - hdr[f"TCRPX{xcol}"]) * hdr[f"TCDLT{xcol}"])
    eta = np.radians((y - hdr[f"TCRPX{ycol}"]) * hdr[f"TCDLT{ycol}"])
    rho = np.hypot(xi, eta)
    c = np.arctan(rho)
    sc, cc = np.sin(c), np.cos(c)
    with np.errstate(invalid="ignore", divide="ignore"):
        dec = np.arcsin(np.where(rho > 0, cc * math.sin(dec0) + eta * sc * math.cos(dec0) / np.where(rho > 0, rho, 1), math.sin(dec0)))
    ra = ra0 + np.arctan2(xi * sc, rho * math.cos(dec0) * cc - eta * math.sin(dec0) * sc)
    return np.degrees(ra) % 360.0, np.degrees(dec)


# Sky positions of the good events of one event file inside an energy band (eV), and the file's exposure time in seconds.
def read_events(path, lo_ev=500.0, hi_ev=7000.0):
    with fits.open(path, memmap=False) as hdul:
        hdu = hdul["EVENTS"]
        hdr = hdu.header
        names = [str(hdr.get(f"TTYPE{i}", "")).lower() for i in range(1, hdr["TFIELDS"] + 1)]
        xcol, ycol = names.index("x") + 1, names.index("y") + 1
        data = hdu.data
        keep = (data["energy"] >= lo_ev) & (data["energy"] <= hi_ev)
        ra, dec = sky_to_radec(np.asarray(data["x"][keep], dtype=np.float64), np.asarray(data["y"][keep], dtype=np.float64), hdr, xcol, ycol)
        return ra, dec, float(hdr.get("EXPOSURE", 0.0))


# Counts per pixel of a grid (the grid's own WCS gives each event's pixel).
def bin_events(ra, dec, wcs, shape):
    px, py = wcs.world_to_pixel_values(ra, dec)
    ix, iy = np.rint(px).astype(np.int64), np.rint(py).astype(np.int64)
    ok = (ix >= 0) & (ix < shape[1]) & (iy >= 0) & (iy < shape[0])
    counts = np.zeros(shape, dtype=np.float32)
    np.add.at(counts, (iy[ok], ix[ok]), 1.0)
    return counts


# Adaptive smoothing: each pixel takes the narrowest Gaussian (sigmas in pixels) whose kernel holds at least `min_counts` photons, else the widest.
def smooth_adaptive(counts, sigmas=(1.5, 3.0, 6.0, 12.0, 24.0), min_counts=12.0):
    out = np.full(counts.shape, np.nan, dtype=np.float32)
    for s in sigmas:
        sm = gaussian_filter(counts.astype(np.float32), s, mode="reflect")
        held = sm * (2.0 * math.pi * s * s)
        take = np.isnan(out) & ((held >= min_counts) | (s == sigmas[-1]))
        out[take] = sm[take]
    return out
