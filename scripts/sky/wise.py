# Makes ghoul pictures of the Witch Head Nebula (IC 2118) from the four WISE bands in IRSA's AllWISE atlas (usage: wise.py OUTDIR [RA DEC SIZE_DEG]).
import concurrent.futures, csv, io, os, sys, time, urllib.parse, urllib.request
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from reproject import reproject_interp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render

UA = os.environ.get("CONTACT_INFO") or "CONTACT-INFO"
BLACKS = [float(v) for v in os.environ.get("BLACKS", "").split(",") if v.strip()]
BLACK = float(os.environ.get("WISE_BLACK", "50"))
PIX = float(os.environ.get("WISE_ARCSEC_PER_PX", "3.0"))
BASE = "https://irsa.ipac.caltech.edu/ibe"
BANDS = [(1, "W1 3.4 µm"), (2, "W2 4.6 µm"), (3, "W3 12 µm"), (4, "W4 22 µm")]
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def get(url, tries=4):
    for attempt in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=90).read()
        except Exception as e:  # noqa: BLE001
            log(f"{url[-90:]}: attempt {attempt + 1} failed: {e}")
            time.sleep(5)
    return None


# Each tile has its own sky level: the offsets that make the overlapping tiles agree (least squares over the pairs that overlap), with the offsets summing to zero.
def solve_offsets(tiles_px):
    k = len(tiles_px)
    rows, rhs = [], []
    for i in range(k):
        for j in range(i + 1, k):
            a, b = tiles_px[i][::4, ::4], tiles_px[j][::4, ::4]
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() > 5000:
                r = np.zeros(k)
                r[i], r[j] = 1.0, -1.0
                rows.append(r)
                rhs.append(float(np.median(a[m] - b[m])))
    if not rows:
        return np.zeros(k)
    rows.append(np.ones(k))
    rhs.append(0.0)
    return np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]


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
WORK = os.environ.get("WORK_DIR", "work")
os.makedirs(WORK, exist_ok=True)
cache = os.path.join(WORK, f"witchhead-{n}.npz")
if os.path.exists(cache):
    z = np.load(cache)
    arrays, names = [z[f"a{i}"] for i in range(len(BANDS))], [name for _, name in BANDS]
    log("mosaics found in the cache; no download")
else:
    arrays, names = [], []
    for band, name in BANDS:
        tl = []
        urls = {cid: f"{BASE}/data/wise/allwise/p3am_cdd/{cid[:2]}/{cid[:4]}/{cid}/{cid}-w{band}-int-3.fits" for cid in tiles}
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:   # the tiles of one band, four at a time
            blobs = dict(zip(urls, pool.map(get, urls.values())))
        for cid, blob in blobs.items():
            if blob is None:
                log(f"{cid} W{band}: not found, skipped")
                continue
            with fits.open(io.BytesIO(blob)) as h:
                data, hdr = h[0].data.astype(np.float32), h[0].header
            reproj, foot = reproject_interp((data, WCS(hdr)), w, shape_out=(n, n))
            tl.append(np.where(np.isfinite(reproj) & (foot > 0.5), reproj, np.nan).astype(np.float32))
            log(f"W{band} {cid}: {int(np.isfinite(tl[-1]).sum())} px on the grid")
        del blobs
        off = solve_offsets(tl)
        log(f"W{band} tile offsets: " + ", ".join(f"{o:.2f}" for o in off))
        acc, cnt = np.zeros((n, n), np.float32), np.zeros((n, n), np.float32)
        for t, o in zip(tl, off):
            ok = np.isfinite(t)
            acc[ok] += t[ok] - o
            cnt[ok] += 1
        arrays.append(np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan).astype(np.float32))
        names.append(name)
        del tl
    np.savez_compressed(cache, **{f"a{i}": a for i, a in enumerate(arrays)})
# Fill the few pixels no tile covers with the median so the stretch is not thrown off.
arrays = [np.where(np.isfinite(a), a, np.nanmedian(a)) for a in arrays]
y0, y1, x0, x1 = (float(v) for v in os.environ.get("WISE_BOX", "0.06,0.80,0.285,0.82").split(","))   # rows run south to north, columns east to west
arrays = [a[int(y0 * n):int(y1 * n), int(x0 * n):int(x1 * n)] for a in arrays]
made = render.render(outdir, "witchhead", names, arrays, f"WISE AllWISE atlas, Witch Head Nebula (IC 2118), {size:g}° field", BLACK, BLACKS, flip=False, rot180=True)
log(f"{len(made)} pictures")
