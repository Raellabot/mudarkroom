# Makes ghoul pictures of the Witch Head Nebula (IC 2118) from the four WISE bands in IRSA's AllWISE atlas (usage: wise.py OUTDIR [RA DEC SIZE_DEG]).
import csv, io, os, sys, time, urllib.parse, urllib.request
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from reproject import reproject_interp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render

UA = os.environ.get("CONTACT_INFO") or "CONTACT-INFO"
BLACKS = [float(v) for v in os.environ.get("BLACKS", "").split(",") if v.strip()]
BLACK = float(os.environ.get("WISE_BLACK", "40"))
PIX = float(os.environ.get("WISE_ARCSEC_PER_PX", "3.0"))
BASE = "https://irsa.ipac.caltech.edu/ibe"
BANDS = [(1, "W1 3.4 µm"), (2, "W2 4.6 µm"), (3, "W3 12 µm"), (4, "W4 22 µm")]
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def get(url, tries=3):
    for attempt in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=300).read()
        except Exception as e:  # noqa: BLE001
            log(f"{url[-90:]}: attempt {attempt + 1} failed: {e}")
            time.sleep(5)
    return None


outdir = sys.argv[1]
ra, dec, size = (float(v) for v in sys.argv[2:5]) if len(sys.argv) >= 5 else (76.3, -6.9, 3.2)
os.makedirs(outdir, exist_ok=True)
# Which atlas tiles cover the field.
q = f"{BASE}/search/wise/allwise/p3am_cdd?" + urllib.parse.urlencode({"POS": f"{ra},{dec}", "SIZE": size, "INTERSECT": "OVERLAPS", "ct": "csv"})
raw = get(q)
if raw is None:
    sys.exit("the IRSA tile search failed")
text = raw.decode("utf-8", "replace")
log("search reply starts: " + text[:300].replace("\n", " | "))
rows = [r for r in csv.DictReader(io.StringIO(text))]
tiles = sorted({r.get("coadd_id") or r.get("COADD_ID") for r in rows if (r.get("coadd_id") or r.get("COADD_ID"))})
log(f"{len(tiles)} tiles: {tiles}")
# The common grid: north up, east left, tangent projection around the field center.
n = int(round(size * 3600.0 / PIX))
w = WCS(naxis=2)
w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
w.wcs.crval = [ra, dec]
w.wcs.crpix = [n / 2.0 + 0.5, n / 2.0 + 0.5]
w.wcs.cdelt = [-PIX / 3600.0, PIX / 3600.0]
arrays, names = [], []
for band, name in BANDS:
    acc, cnt = np.zeros((n, n), np.float32), np.zeros((n, n), np.float32)
    for cid in tiles:
        url = f"{BASE}/data/wise/allwise/p3am_cdd/{cid[:2]}/{cid[:4]}/{cid}/{cid}-w{band}-int-3.fits"
        blob = get(url)
        if blob is None:
            log(f"{cid} W{band}: not found, skipped")
            continue
        with fits.open(__import__("io").BytesIO(blob)) as h:
            data, hdr = h[0].data.astype(np.float32), h[0].header
        reproj, foot = reproject_interp((data, WCS(hdr)), w, shape_out=(n, n))
        ok = np.isfinite(reproj) & (foot > 0.5)
        acc[ok] += reproj[ok]
        cnt[ok] += 1
        log(f"W{band} {cid}: {int(ok.sum())} px on the grid")
    out = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan).astype(np.float32)
    arrays.append(out)
    names.append(name)
# Fill the few pixels no tile covers with the median so the stretch is not thrown off.
arrays = [np.where(np.isfinite(a), a, np.nanmedian(a)) for a in arrays]
made = render.render(outdir, "witchhead", names, arrays, f"WISE AllWISE atlas, Witch Head Nebula (IC 2118), {size:g}° field", BLACK, BLACKS, flip=True)
log(f"{len(made)} pictures")
