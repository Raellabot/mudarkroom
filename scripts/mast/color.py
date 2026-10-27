# Makes color pictures of Hubble and Webb fields from several filters of one target (usage and settings are in README.md).
import colorsys, gzip, json, os, sys, time, urllib.error, urllib.request, warnings
from astropy.io import fits
from astropy.visualization import make_lupton_rgb
from astropy.wcs import WCS
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter, uniform_filter1d

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "field"))
import chandra_events

# Your name and contact URL, sent to MAST as the User-Agent: edit this line or set the CONTACT_INFO environment variable.
UA = os.environ.get("CONTACT_INFO") or "CONTACT-INFO"
DOWNLOAD = "https://mast.stsci.edu/api/v0.1/Download/file?uri="
MAX_SIDE = int(os.environ.get("MAX_SIDE", "2600"))
MAX_FILTERS = int(os.environ.get("MAX_FILTERS", "6"))
WORK = os.environ.get("WORK_DIR", "work")
STRETCH = float(os.environ.get("STRETCH", "0.25"))
Q = float(os.environ.get("Q", "8"))
SIZE_CAP = float(os.environ.get("SIZE_CAP_GB", "6")) * 1e9
FLATTEN = os.environ.get("FLATTEN", "1") == "1"
NO_DOWNLOAD = os.environ.get("NO_DOWNLOAD", "0") == "1"
NORM_EXPS = [float(v) for v in os.environ.get("NORM_EXPS", "0.5").split(",") if v.strip()]
FLIP = os.environ.get("FLIP", "1") == "1"
CACHE_ONLY = {v.strip() for v in os.environ.get("CACHE_ONLY", "").split(",") if v.strip()}  # sets that must never be downloaded
VARIANTS = [v.strip() for v in os.environ.get("VARIANTS", "base").split(",") if v.strip()]
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


# ---- palettes: t in [0, 1] is the place of the filter in the order of wavelengths --------------------------------------------------------
# Blue (shortest) to red (longest) through cyan, green, yellow and orange.
def chromatic(t):
    r, g, b = colorsys.hsv_to_rgb(240.0 / 360.0 * (1.0 - t), 1.0, 1.0)
    return np.array([r, g, b])


CREATIVE = [(0.00, (0.10, 0.25, 1.00)), (0.25, (0.00, 0.80, 0.90)), (0.50, (1.00, 0.85, 0.40)), (0.75, (1.00, 0.40, 0.25)), (1.00, (0.90, 0.20, 0.60))]


def creative(t):
    for (a, ca), (b, cb) in zip(CREATIVE, CREATIVE[1:]):
        if t <= b:
            k = (t - a) / (b - a)
            return np.array(ca) * (1 - k) + np.array(cb) * k
    return np.array(CREATIVE[-1][1])


# Creative palette 2: no dark blue; aqua for the shortest wavelength, then green, yellow, orange and a red-pink for the longest.
CREATIVE2 = [(0.00, (0.20, 0.80, 0.95)), (0.20, (0.20, 0.90, 0.55)), (0.45, (0.75, 0.95, 0.30)), (0.65, (1.00, 0.85, 0.30)), (0.85, (1.00, 0.50, 0.25)), (1.00, (0.95, 0.25, 0.45))]


def creative2(t):
    for (a, ca), (b, cb) in zip(CREATIVE2, CREATIVE2[1:]):
        if t <= b:
            k = (t - a) / (b - a)
            return np.array(ca) * (1 - k) + np.array(cb) * k
    return np.array(CREATIVE2[-1][1])


# Creative palette 3: palette 2 with the warm end turned pink (coral, hot pink, magenta-pink) so the long filters read pink instead of orange.
CREATIVE3 = [(0.00, (0.20, 0.80, 0.95)), (0.18, (0.30, 0.90, 0.60)), (0.38, (0.80, 0.95, 0.40)), (0.55, (1.00, 0.85, 0.40)), (0.72, (1.00, 0.52, 0.52)), (0.88, (1.00, 0.30, 0.66)), (1.00, (0.95, 0.38, 0.82))]


def creative3(t):
    for (a, ca), (b, cb) in zip(CREATIVE3, CREATIVE3[1:]):
        if t <= b:
            k = (t - a) / (b - a)
            return np.array(ca) * (1 - k) + np.array(cb) * k
    return np.array(CREATIVE3[-1][1])


# Ranked palettes: a filter's hue depends on its rank among the filters, not its wavelength; "stsci" copies the hues of STScI's color keys.
STSCI_HUES = {"blue": (0.10, 0.30, 1.00), "cyan": (0.00, 0.90, 1.00), "green": (0.10, 1.00, 0.20), "yellow": (1.00, 0.95, 0.10), "orange": (1.00, 0.50, 0.05), "red": (1.00, 0.10, 0.10)}
STSCI_BY_COUNT = {
    2: ["blue", "red"], 3: ["blue", "green", "red"], 4: ["blue", "green", "orange", "red"], 5: ["blue", "cyan", "green", "orange", "red"],
    6: ["blue", "cyan", "green", "yellow", "orange", "red"],
}


def stsci_colors(n):
    names = STSCI_BY_COUNT.get(n)
    if names:
        return [np.array(STSCI_HUES[h]) for h in names]
    ramp = [np.array(STSCI_HUES[h]) for h in STSCI_BY_COUNT[6]]  # more than six filters: spread along the six hues
    out = []
    for i in range(n):
        x = i * 5 / (n - 1)
        a = min(int(x), 4)
        out.append(ramp[a] * (1 - (x - a)) + ramp[a + 1] * (x - a))
    return out


def creativerank_colors(n):
    return [creative3(i / (n - 1) if n > 1 else 0.0) for i in range(n)]


# Creative palette 5: NASA-style blue and cyan for the short wavelengths, a soft yellow in the middle, the pinks of palette 4 for the long ones.
CREATIVE5 = [(0.00, (0.10, 0.30, 1.00)), (0.25, (0.00, 0.90, 1.00)), (0.50, (0.95, 0.90, 0.45)), (0.75, (1.00, 0.45, 0.62)), (1.00, (0.95, 0.38, 0.82))]


def creative5_colors(n):
    out = []
    for i in range(n):
        t = i / (n - 1) if n > 1 else 0.0
        for (a, ca), (b, cb) in zip(CREATIVE5, CREATIVE5[1:]):
            if t <= b:
                k = (t - a) / (b - a)
                out.append(np.array(ca) * (1 - k) + np.array(cb) * k)
                break
    return out


RANKED = {"stsci": stsci_colors, "creativerank": creativerank_colors, "creative5": creative5_colors}
ALL_PALETTES = {"chromatic": chromatic, "creative": creative, "creative2": creative2, "creative3": creative3, "stsci": None, "creativerank": None, "creative5": None}
PALETTE_NAMES = {"chromatic": "chromatic palette", "creative": "creative palette", "creative2": "creative palette 2", "creative3": "creative palette 3", "stsci": "NASA-style hues, by rank", "creativerank": "creative palette 4, by rank", "creative5": "creative palette 5, by rank"}
PALETTES = {k: ALL_PALETTES[k] for k in os.environ.get("PALETTES", "chromatic,creative").split(",") if k in ALL_PALETTES}


def hexof(c):
    return "#%02x%02x%02x" % tuple(int(round(255 * v)) for v in np.clip(c, 0, 1))


# ---- choosing the filters ------------------------------------------------------------------------------------------------------------------
def pick(rows, collection, target, proposal, instrument, limit=None):
    best = {}
    for r in rows:
        if (r["obs_collection"], r["target_name"], r["proposal_id"], r["instrument_name"]) != (collection, target, proposal, instrument):
            continue
        f = r.get("filters") or ""
        if r.get("dataproduct_type") != "image" or r.get("calib_level") != 3 or not (r.get("dataURL") or "").endswith(".fits"):
            continue
        if ";" in f or not f or "detection" in f.lower() or r.get("em_min") is None or r.get("em_max") is None:
            continue
        if f not in best or (r.get("t_exptime") or 0) > (best[f].get("t_exptime") or 0):
            best[f] = r
    chosen = sorted(best.values(), key=lambda r: (r["em_min"] + r["em_max"]) / 2)
    return thin(chosen, limit or MAX_FILTERS)


# An even spread of wavelengths when there are more filters than the limit.
def thin(chosen, limit):
    if len(chosen) > limit:
        idx = sorted({round(i * (len(chosen) - 1) / (limit - 1)) for i in range(limit)})
        chosen = [chosen[i] for i in idx]
    return chosen


# Several sources in one set: the spec fields hold one value per source joined by "+" (HST+JWST, 10248+2662, ...); the filters of all sources are merged by wavelength.
def pick_all(rows, collection, target, proposal, instrument):
    fields = [collection.split("+"), target.split("+"), proposal.split("+"), instrument.split("+")]
    if len({len(f) for f in fields}) != 1:
        raise ValueError("the first four spec fields need the same number of '+' separated values")
    groups = list(zip(*fields))
    if len(groups) == 1:
        return pick(rows, *groups[0])
    chosen = sorted((r for g in groups for r in pick(rows, *g, limit=10 ** 6)), key=lambda r: (r["em_min"] + r["em_max"]) / 2)
    names = [r["filters"] for r in chosen]
    if len(set(names)) != len(names):
        raise ValueError(f"the same filter name comes from two sources: {sorted({n for n in names if names.count(n) > 1})}")
    return thin(chosen, MAX_FILTERS)


# ---- download and reduce -------------------------------------------------------------------------------------------------------------------
def download(uri, dest):
    url = DOWNLOAD + urllib.request.quote(uri, safe=":/")
    for attempt in range(1, 4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=180) as resp, open(dest + ".part", "wb") as out:
                size = int(resp.headers.get("Content-Length") or 0)
                if size > SIZE_CAP:
                    raise RuntimeError(f"file is {size / 1e9:.1f} GB, over the {SIZE_CAP / 1e9:.1f} GB cap")
                got = 0
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    got += len(chunk)
            os.replace(dest + ".part", dest)
            log(f"downloaded {os.path.basename(dest)}: {got / 1e6:.0f} MB")
            return
        except RuntimeError:
            raise
        except Exception as e:  # noqa: BLE001
            log(f"download attempt {attempt} failed: {e}")
            time.sleep(5 * attempt)
    raise RuntimeError(f"could not download {uri}")


def sci_hdu(hdul):
    for h in hdul:
        if h.name == "SCI" and h.data is not None and h.data.ndim == 2:
            return h
    for h in hdul:
        if h.data is not None and h.data.ndim == 2:
            return h
    raise RuntimeError("no 2-D image in the file")


# The WCS of the image after averaging g x g blocks (pixel centers move by half a block).
def rebinned_wcs(w, g):
    n = w.deepcopy()
    n.wcs.crpix = (np.array(w.wcs.crpix) - 0.5) / g + 0.5
    if n.wcs.has_cd():
        n.wcs.cd = np.array(w.wcs.cd) * g
    else:
        n.wcs.cdelt = np.array(w.wcs.cdelt) * g
    return n


def scale_of(w):
    from astropy.wcs.utils import proj_plane_pixel_scales
    return float(np.mean(proj_plane_pixel_scales(w))) * 3600.0  # arcsec per pixel


# The header of the SCI (or first 2-D) image within the first bytes of a FITS file, or None if it is not there.
def parse_image_header(blob):
    pos, first2d = 0, None
    for _ in range(12):
        end = -1
        for i in range(pos, len(blob) - 2879, 2880):
            if any(blob[i + j:i + j + 80].rstrip() == b"END" for j in range(0, 2880, 80)):
                end = i + 2880
                break
        if end < 0:
            break
        hdr = fits.Header.fromstring(blob[pos:end].decode("ascii", "replace"))
        n = hdr.get("NAXIS", 0)
        if n == 2 and (first2d is None):
            first2d = hdr
        if n == 2 and hdr.get("EXTNAME") == "SCI":
            return hdr
        size = abs(hdr.get("BITPIX", 8)) // 8 * (int(np.prod([hdr[f"NAXIS{k}"] for k in range(1, n + 1)])) if n else 0)
        pos = end + ((size + 2879) // 2880) * 2880
        if pos >= len(blob):
            break
    return first2d


# (header, file size) of the image extension from an HTTP range request on the first bytes; (None, 0) on failure.
def header_of(uri):
    url = DOWNLOAD + urllib.request.quote(uri, safe=":/")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Range": "bytes=0-799999"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            blob = resp.read(800000)
            total = resp.headers.get("Content-Range", "").rsplit("/", 1)[-1] or resp.headers.get("Content-Length") or "0"
    except Exception as e:  # noqa: BLE001
        log(f"range request failed: {e}")
        return None, 0
    return parse_image_header(blob), int(total) if str(total).isdigit() else 0


# Averages g x g blocks of the SCI image a strip of rows at a time, so a 3 GB mosaic never sits in memory twice.
def block_mean_file(path, g, blank_zero=False):
    with fits.open(path, memmap=True) as hdul:
        h = sci_hdu(hdul)
        w = WCS(h.header)
        ny, nx = h.data.shape
        oy, ox = ny // g, nx // g
        out = np.full((oy, ox), np.nan, dtype=np.float32)
        step = max(1, 256)  # output rows per strip
        for r0 in range(0, oy, step):
            r1 = min(oy, r0 + step)
            strip = np.asarray(h.data[r0 * g:r1 * g, :ox * g], dtype=np.float32)
            if blank_zero:  # drizzled Hubble products hold exactly 0 where nothing was observed
                strip = np.where(strip == 0, np.nan, strip)
            v = np.where(np.isfinite(strip), strip, np.nan).reshape(r1 - r0, g, ox, g)
            with np.errstate(all="ignore"):
                cnt = np.isfinite(v).sum(axis=(1, 3))
                sm = np.nansum(v, axis=(1, 3))
            out[r0:r1] = np.where(cnt >= (g * g) / 2, sm / np.maximum(cnt, 1), np.nan)
    return out, w


# Returns (arrays on the common grid, grid info, the filters actually used). NaN outside each filter's coverage.
def align(chosen, work, label=None):
    heads = {}
    for r in chosen:
        hdr, size = header_of(r["dataURL"])
        if hdr is None:
            raise RuntimeError(f"could not read the header of {r['dataURL']} without downloading it")
        if size > SIZE_CAP:
            log(f"{r['filters']}: file is {size / 1e9:.1f} GB, over the {SIZE_CAP / 1e9:.1f} GB cap; this filter is left out")
            continue
        w = WCS(hdr)
        heads[r["filters"]] = (r, w, (int(hdr["NAXIS2"]), int(hdr["NAXIS1"])), size)
        log(f"{r['filters']}: {hdr['NAXIS1']}x{hdr['NAXIS2']} px at {scale_of(w):.4f} arcsec/px, {size / 1e6:.0f} MB")
    if len(heads) < 3:
        raise RuntimeError("fewer than three filters can be used")
    used = [heads[r["filters"]][0] for r in chosen if r["filters"] in heads]
    # The grid: the footprint of the filter that covers the most sky, reduced so the longest side is MAX_SIDE.
    ref_r, ref_w, ref_shape, _ = max(heads.values(), key=lambda v: v[2][0] * v[2][1] * scale_of(v[1]) ** 2)
    f = max(1, int(np.ceil(max(ref_shape) / MAX_SIDE)))
    out_w = rebinned_wcs(ref_w, f)
    ny, nx = ref_shape[0] // f, ref_shape[1] // f
    log(f"grid from {ref_r['filters']}: {nx}x{ny} px (reduced {f}x), {scale_of(out_w):.3f} arcsec/px")
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float64)
    world = out_w.pixel_to_world_values(xx, yy)
    del yy, xx
    out = []
    # Each filter is saved as soon as it is aligned, so a stopped run keeps what it finished (the grid depends only on the headers).
    sig = f"{nx}x{ny}x{f}x{ref_r['filters']}x{len(used)}"
    sigfile = os.path.join(work, f"{label}-part-grid.txt") if label else None
    if sigfile and (not os.path.exists(sigfile) or open(sigfile).read() != sig):
        for fn in os.listdir(work):
            if fn.startswith(f"{label}-part-"):
                os.remove(os.path.join(work, fn))
        open(sigfile, "w").write(sig)
    for r in used:
        _, w, shape, _ = heads[r["filters"]]
        partfile = os.path.join(work, f"{label}-part-{r['filters']}.npy") if label else None
        if partfile and os.path.exists(partfile):
            res = np.load(partfile)
            out.append(res)
            log(f"{r['filters']}: found aligned in the cache from an earlier run; no download")
            continue
        path = os.path.join(work, os.path.basename(r["dataURL"]))
        if os.path.exists(path + ".part"):
            os.remove(path + ".part")
        if not os.path.exists(path):
            download(r["dataURL"], path)
        g = max(1, int(np.floor(scale_of(out_w) / scale_of(w))))
        red, w2 = block_mean_file(path, g, blank_zero=r["obs_collection"] == "HST")
        os.remove(path)
        rw = rebinned_wcs(w2, g)
        px, py = rw.world_to_pixel_values(*world)
        valid = np.isfinite(red)
        filled = np.where(valid, red, 0.0).astype(np.float32)
        val = ndi.map_coordinates(filled, [py, px], order=1, mode="constant", cval=0.0, prefilter=False)
        cov = ndi.map_coordinates(valid.astype(np.float32), [py, px], order=1, mode="constant", cval=0.0, prefilter=False)
        res = np.where(cov > 0.5, val / np.maximum(cov, 1e-6), np.nan).astype(np.float32)
        out.append(res)
        if partfile:
            np.save(partfile, res)
        log(f"{r['filters']}: aligned (reduced {g}x), covers {np.isfinite(res).mean() * 100:.0f}% of the grid")
    return out, {"width": nx, "height": ny, "reduction": f, "arcsec_per_pixel": scale_of(out_w), "reference_filter": ref_r["filters"], "wcs": out_w.to_header_string(relax=True)}, used


# Take off the slowly changing sky background: a low percentile of each block, smoothed and interpolated, subtracted.
def flatten(a, blocks=40):
    ny, nx = a.shape
    by, bx = max(16, ny // blocks), max(16, nx // blocks)
    gy, gx = max(2, ny // by), max(2, nx // bx)
    grid = np.full((gy, gx), np.nan, dtype=np.float32)
    for i in range(gy):
        for j in range(gx):
            v = a[i * by:(i + 1) * by if i < gy - 1 else ny, j * bx:(j + 1) * bx if j < gx - 1 else nx]
            v = v[np.isfinite(v)]
            if v.size > (by * bx) // 4:
                grid[i, j] = np.percentile(v, 30)
    if not np.isfinite(grid).any():
        return a
    fill = np.where(np.isfinite(grid), grid, np.nanmedian(grid))
    sm = ndi.gaussian_filter(fill, 1.2, mode="nearest")
    cy = (np.arange(ny) + 0.5) / by - 0.5
    cx = (np.arange(nx) + 0.5) / bx - 0.5
    bg = ndi.map_coordinates(sm, np.meshgrid(cy.clip(0, gy - 1), cx.clip(0, gx - 1), indexing="ij"), order=1, mode="nearest")
    return (a - bg).astype(np.float32)


# ---- color ------------------------------------------------------------------------------------------------------------------------------
# ---- finishing steps: a table per set label, and sets not in FINISH are made without them ----

# Removes row stripes in column segments of `seg` px: each row's faint-sky level minus its smooth trend over `win` rows is subtracted.
def row_destripe(a, seg=96, win=25, pct=25, axis=1):
    a = a.astype(np.float32).copy()
    if axis == 0: a = a.T
    H, W = a.shape
    n = max(1, int(round(W / seg)))
    edges = np.linspace(0, W, n + 1).astype(int)
    for i in range(n):
        s = a[:, edges[i]:edges[i+1]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prof = np.nanpercentile(s, pct, axis=1)
        ok = np.isfinite(prof)
        if ok.sum() < 3 * win: continue
        idx = np.arange(H)
        prof = np.interp(idx, idx[ok], prof[ok])
        hp = prof - uniform_filter1d(prof, size=win, mode="nearest")
        a[:, edges[i]:edges[i+1]] -= hp[:, None]
    return a.T if axis == 0 else a



def dense_crop_box(arrs, thr=0.97):
    cover = np.all([np.isfinite(a) for a in arrs], axis=0)
    ys, xs = np.where(cover)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    box0 = (y0, y1, x0, x1)
    while True:
        if y1 - y0 < 2 or x1 - x0 < 2: return box0
        c = cover[y0:y1, x0:x1]
        rows, cols = c.mean(axis=1), c.mean(axis=0)
        moved = False
        if rows[0] < thr: y0 += 1; moved = True
        if rows[-1] < thr: y1 -= 1; moved = True
        if cols[0] < thr: x0 += 1; moved = True
        if cols[-1] < thr: x1 -= 1; moved = True
        if not moved: return y0, y1, x0, x1


# Fill NaN holes (gaps between tiles, dead pixels) from their neighbors by normalized smoothing; the data around them are untouched.
def fill_holes(a, sigma=2.0, passes=3):
    a = a.astype(np.float32).copy()
    for _ in range(passes):
        bad = ~np.isfinite(a)
        if not bad.any(): break
        v = np.where(bad, 0, a); wgt = (~bad).astype(np.float32)
        num = gaussian_filter(v, sigma); den = gaussian_filter(wgt, sigma)
        fill = np.where(den > 0.05, num / np.maximum(den, 1e-6), np.nan)
        a = np.where(bad, fill, a)
        sigma *= 1.6
    return a

# Like fill_holes, but only inside the footprint: gaps narrower than 2 x reach px are closed, the outside edge stays empty.
def fill_inside(a, sigma=2.0, passes=3, reach=60):
    from scipy.ndimage import binary_closing
    pad = int(reach) + 2
    inside = binary_closing(np.pad(np.isfinite(a), pad), iterations=int(reach))[pad:-pad, pad:-pad]
    return np.where(inside, fill_holes(a, sigma, passes), np.nan).astype(np.float32)

def shift_img(a, dy, dx):
    from scipy.ndimage import shift
    return shift(np.nan_to_num(a, nan=0.0), (dy, dx), order=1, mode="nearest")

# (dy, dx) in pixels by which `a` is displaced from `ref` (cross-correlation of square-root images, sub-pixel by a parabola).
def phase_shift(a, ref):
    a = np.sqrt(np.clip(np.nan_to_num(a), 0, None)); b = np.sqrt(np.clip(np.nan_to_num(ref), 0, None))
    a = a / max(a.max(), 1e-9); b = b / max(b.max(), 1e-9)
    c = np.fft.fftshift(np.fft.ifft2(np.fft.fft2(a) * np.conj(np.fft.fft2(b))).real)
    pk = np.unravel_index(np.argmax(c), c.shape)
    def sub(i, lo, mid, hi):
        d = lo - 2 * mid + hi
        return 0.0 if d == 0 else 0.5 * (lo - hi) / d
    dy = pk[0] - c.shape[0] // 2; dx = pk[1] - c.shape[1] // 2
    if 0 < pk[0] < c.shape[0] - 1: dy += sub(0, c[pk[0]-1, pk[1]], c[pk], c[pk[0]+1, pk[1]])
    if 0 < pk[1] < c.shape[1] - 1: dx += sub(0, c[pk[0], pk[1]-1], c[pk], c[pk[0], pk[1]+1])
    return float(dy), float(dx)


# Per-subject finishing in order: crop to a box, crop to dense data, fill holes, crop and align on a planet, remove row stripes.
def finish_arrays(arrs, fin, scale=1.0):
    a = list(arrs)
    if fin.get("box"):
        fy0, fy1, fx0, fx1 = fin["box"]
        H, W = a[0].shape
        a = [x[int(fy0 * H):int(fy1 * H), int(fx0 * W):int(fx1 * W)] for x in a]
    if fin.get("dense"):
        y0, y1, x0, x1 = dense_crop_box(a, fin["dense"])
        a = [x[y0:y1, x0:x1] for x in a]
    if not fin.get("nofill"):
        if fin.get("fill"):   # "fill": (first width in px, passes, reach in px): wide gaps are filled but not the sky outside the footprint
            a = [fill_inside(x, *fin["fill"]) for x in a]
        else:
            a = [fill_holes(x) for x in a]
    if fin.get("planet"):
        (cy, cx), rf = fin["planet"]
        H, W = a[0].shape
        r = int(rf * W)
        y, x0 = int(cy * H), int(cx * W)
        a = [x[max(0, y - r):y + r, max(0, x0 - r):x0 + r] for x in a]
        shifts = [(0.0, 0.0)] + [phase_shift(x, a[0]) for x in a[1:]]
        log("planet offsets between filters (px): " + ", ".join(f"({dy:.1f},{dx:.1f})" for dy, dx in shifts))
        a = [shift_img(x, -dy, -dx) for x, (dy, dx) in zip(a, shifts)]
    if fin.get("destripe"):
        seg, win = fin["destripe"]
        a = [row_destripe(x, seg=max(8, int(seg * scale)), win=max(5, int(win * scale))) for x in a]
    return a


FINISH = {
    "saturn": {"planet": ((0.501, 0.501), 0.155), "black": 40, "stretch": 0.4, "upscale": 3},
    "uranus": {"planet": ((0.554, 0.573), 0.08), "destripe": (180, 45), "black": 40, "stretch": 0.4, "upscale": 3},
    "neptune": {"planet": ((0.520, 0.548), 0.10), "destripe": (180, 45), "black": 40, "stretch": 0.4, "upscale": 3},
    "ldn1527": {"destripe": (288, 75), "black": 70, "stretch": 0.4},
    "crab-miri": {"dense": 0.9, "destripe": (288, 75), "black": 45, "stretch": 0.4},
    "ngc602-miri": {"dense": 0.9, "black": 15, "stretch": 0.25},
    "jades-goods-n": {"nofill": True, "destripe": (288, 75), "black": 50},
    "ceers-miri1": {"nofill": True, "black": 75},
    "hudf-acs": {"nofill": True, "black": 50},
    "abell2744-glass": {"nofill": True, "destripe": (288, 75), "black": 60},
    "cosmos-web-tile": {"nofill": True, "destripe": (288, 75), "black": 65},
    "ngdeep": {"nofill": True, "black": 70},
    "jems-udf": {"nofill": True, "black": 70},
    "jades-goods-s": {"nofill": True, "destripe": (288, 75), "black": 70},
    "primer-uds": {"nofill": True, "destripe": (288, 75), "black": 90},
    "smacs0723": {"box": (0.0, 1.0, 0.55, 1.0), "dense": 0.97},   # the cluster half; the other module's field is left out
    "ngc602-multi": {"nofill": True, "destripe": (288, 75), "black": 50, "xray_dim": (0.55, 40)},   # the brightest X-ray source is dimmed by 55% over a 40 px radius
    "sn1987a-1726": {"nofill": True, "black": 20, "upscale": 4, "sky": (83.86679, -69.26975, 7.0)},   # a 25 arcsecond close-up of the remnant
    "elgordo-nircam": {"box": (0.0, 1.0, 0.565, 1.0), "black": 55, "fill": (8.0, 4, 130)},   # the right pointing holds the cluster; the left one is a different field
    "ngc346-nircam": {"box": (0.012, 0.965, 0.58, 0.99), "black": 30, "fill": (8.0, 4, 130), "destripe": (288, 75)},   # the right pointing holds the nebula; the left one is star field; gaps between detectors are filled
    "antennae-nircam": {"black": 30},   # a darker sky: the default black point leaves a milky halo over the whole frame
    "westerlund1-nircam": {"black": 1.0},   # fills the masked cores of saturated stars and the gap strip
    "ngc3324-multi": {"nofill": True, "black": 50},   # the Cosmic Cliffs; black point and X-ray dimming are tuned on the exported samples
    "hltau-hst-narrow": {"nofill": True, "black": 30, "sky": (67.9102, 18.2327, 14.0), "upscale": 3},   # HL Tau on the WFPC2 PC chip
    "hltau-hst-wide": {"nofill": True, "black": 30, "sky": (67.9102, 18.2327, 14.0), "upscale": 3},
    "hltau-webb": {"nofill": True, "black": 10, "upscale": 8},
    "ngc3324-nircam": {"nofill": True, "black": 50, "xray_sig": 3.0},   # the Cosmic Cliffs, NIRCam only; the X-ray layer keeps only sources above S/N 3 (a 69 ks exposure)
}

# Sets that get a Chandra X-ray layer: the target and detector of the public imaging observations, and the photon energy band in eV.
XRAY = {
    "ngc602-multi": {"target": "NGC602", "instrument": "ACIS-I", "band": (500.0, 7000.0)},
    "ngc3324-multi": {"target": "NGC3324", "instrument": "ACIS-I", "band": (500.0, 7000.0)},
    "ngc3324-nircam": {"target": "NGC3324", "instrument": "ACIS-I", "band": (500.0, 7000.0)},
}
XRAY_COLOR = np.array([0.55, 1.00, 0.10])   # lime green
FINISH_ON = os.environ.get("FINISH", "1") == "1"


# Background off, then scaled so the 99.7th percentile of the signal is 1. NaN (no coverage) becomes 0.
def normalize(a):
    v = a[np.isfinite(a)]
    if v.size < 100:
        return np.zeros_like(a), 0.0
    bg = np.median(v[v <= np.percentile(v, 60)]) if v.size else 0.0
    x = a - bg
    top = np.percentile(x[np.isfinite(x)], 99.7)
    if not np.isfinite(top) or top <= 0:
        top = float(np.nanmax(x)) or 1.0
    x = np.clip(x / top, 0, None)
    return np.nan_to_num(x, nan=0.0).astype(np.float32), float(top)


# Black point at the 1st percentile, each filter divided by its own 99.7th percentile, then weighted by (top / largest top) ** (1 - exp).
def normalize_keep_haze(arrays, exp, black=1.0):
    out, tops = [], []
    for a in arrays:
        v = a[np.isfinite(a)]
        if v.size < 100:
            out.append(np.zeros_like(a)); tops.append(0.0)
            continue
        x = a - np.percentile(v, black)
        x = np.clip(x, 0, None)
        top = float(np.percentile(x[np.isfinite(x)], 99.7)) or float(np.nanmax(x)) or 1.0
        out.append(np.nan_to_num(x / top, nan=0.0).astype(np.float32)); tops.append(top)
    big = max(tops) or 1.0
    weights = [((t / big) ** (1.0 - exp)) if t > 0 else 0.0 for t in tops]
    return [o * w for o, w in zip(out, weights)], tops, weights


# The Chandra photons of a set binned onto the set's grid: counts per pixel, cached beside the aligned arrays, and a note of the observations used.
def xray_counts(label, grid, cfg):
    cache = os.path.join(WORK, f"{label}-xray.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        if str(z["wcs"]) == grid["wcs"] and z["counts"].shape == (grid["height"], grid["width"]):
            log(f"{label}: X-ray counts found in the cache; no download")
            return z["counts"].astype(np.float32), json.loads(str(z["info"]))
        log(f"{label}: the cached X-ray counts are for a different grid")
    if NO_DOWNLOAD or label in CACHE_ONLY:
        raise RuntimeError("no cached X-ray counts for this grid and downloads are off for this set")
    if "-SIP" in grid["wcs"]:
        raise RuntimeError("the grid header has SIP distortion terms, which the X-ray binning does not follow")
    wcs = WCS(fits.Header.fromstring(grid["wcs"]))
    files = chandra_events.event_files(cfg["target"], cfg["instrument"])
    if not files:
        raise RuntimeError(f"no public {cfg['instrument']} observations of {cfg['target']}")
    counts = np.zeros((grid["height"], grid["width"]), dtype=np.float32)
    info = {"observations": [], "exposure_ks": 0.0, "photons": 0, "band_ev": list(cfg["band"])}
    for f in files:
        path = os.path.join(WORK, f"evt-{f['obsid']}.fits.gz")
        chandra_events.fetch(f["url"], path)
        ra, dec, expo = chandra_events.read_events(path, *cfg["band"])
        os.remove(path)
        counts += chandra_events.bin_events(ra, dec, wcs, counts.shape)
        info["observations"].append(f["obsid"])
        info["exposure_ks"] += expo / 1000.0
        info["photons"] += int(ra.size)
        log(f"Chandra obsid {f['obsid']}: {ra.size} photons in the band, {expo / 1000:.1f} ks")
    log(f"{label}: {len(files)} Chandra observations, {info['photons']} photons, {info['exposure_ks']:.0f} ks, {counts.sum():.0f} on the grid")
    np.savez_compressed(cache, counts=counts, info=json.dumps(info), wcs=grid["wcs"])
    return counts, info


# The X-ray layer's brightness from 0 to 1: significance-based smoothing of the counts, asinh stretch relative to the background, and an optional dimming around the brightest source; `scale` is the grid size relative to the full one.
def xray_layer(counts, fin, scale=1.0):
    sigmas = tuple(1.5 * scale * 1.6 ** i for i in range(7))
    d, bg = chandra_events.smooth_significant(counts, sigmas, fin.get("xray_sig", 1.5))
    d = gaussian_filter(d, 0.6 * scale, mode="reflect")
    top = float(np.percentile(d[d > 0], 99.9)) if (d > 0).any() else 1.0
    soft = fin.get("xray_soft", 1.0) * bg
    layer = np.clip(np.arcsinh(d / soft) / np.arcsinh(top / soft), 0, 1)
    if fin.get("xray_dim"):
        strength, radius = fin["xray_dim"]
        y, x = np.unravel_index(d.argmax(), d.shape)
        yy, xx = np.mgrid[0:d.shape[0], 0:d.shape[1]]
        layer = layer * (1 - strength * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * (radius * scale) ** 2)))
    return layer.astype(np.float32)


# Tweaks of a ranked palette keyed by filter name; "whiten" turns bright star cores and glare toward white (0 = off, 1 = fully).
TWEAKS = {
    "base": {"gain": {}, "soften": {}, "whiten": 0.0},
    "deep": {"gain": {"F187N": 0.55, "F090W": 1.35}, "soften": {}, "whiten": 0.0},
    "soft": {"gain": {}, "soften": {"F335M": 0.35, "F444W": 0.30}, "whiten": 0.0},
    "deepsoft": {"gain": {"F187N": 0.55, "F090W": 1.35}, "soften": {"F335M": 0.35, "F444W": 0.30}, "whiten": 0.0},
    "white": {"gain": {}, "soften": {}, "whiten": 0.55},
    "whiter": {"gain": {}, "soften": {}, "whiten": 0.9},
    "gentle": {"gain": {}, "soften": {}, "whiten": 0.55, "stretch": 0.5},    # for big bright objects: a gentler stretch so the inner structure is not blown out
    "gentler": {"gain": {}, "soften": {}, "whiten": 0.55, "stretch": 1.0},
}
SOFT_PINK = np.array([1.00, 0.78, 0.86])  # what a softened color is mixed toward


# Blends bright pixels toward white: the weight grows from 0 at 45% brightness to k at 95%.
def whiten_bright(img, k):
    f = img.astype(np.float32) / 255.0
    w = np.clip((f.max(axis=2) - 0.45) / 0.5, 0.0, 1.0) * k
    f = f + w[..., None] * (1.0 - f)
    return np.clip(np.rint(f * 255.0), 0, 255).astype(np.uint8)


def compose(stack, waves, palette, ranked=None, colors=None, whiten=0.0, stretch=None, xray=None):
    lw = np.log(np.array(waves, dtype=float))
    t = (lw - lw.min()) / (lw.max() - lw.min()) if lw.max() > lw.min() else np.zeros_like(lw)
    if colors is None:
        colors = ranked(len(stack)) if ranked else [palette(float(k)) for k in t]
    rgb = np.zeros(stack[0].shape + (3,), dtype=np.float32)
    for x, c in zip(stack, colors):
        for ch in range(3):
            rgb[..., ch] += x * c[ch]
    rgb *= 3.0 / max(len(stack), 3)
    img = make_lupton_rgb(rgb[..., 0], rgb[..., 1], rgb[..., 2], minimum=0.0, stretch=stretch or STRETCH, Q=Q, output_dtype=np.uint8)
    if whiten > 0:
        img = whiten_bright(img, whiten)
    if xray is not None:  # screened on top, so the X-rays add light and never darken the picture
        f = img.astype(np.float32) / 255.0
        img = np.clip(np.rint(255.0 * (1.0 - (1.0 - f) * (1.0 - XRAY_COLOR * xray[..., None]))), 0, 255).astype(np.uint8)
    if FLIP:
        img = np.ascontiguousarray(img[::-1])
    return img, colors


def legend(img, names, colors, waves, title, extra=()):
    h, w = img.shape[:2]
    bar = max(70, h // 22)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", max(14, bar // 4))
    except OSError:
        font = ImageFont.load_default()
    probe = ImageDraw.Draw(Image.new("RGB", (4, 4)))
    sw, line = bar // 3, max(14, bar // 4) + 8
    items = []
    for n, c, wv in zip(names, colors, waves):
        items.append((f"{n} {wv / 1000:.2f} µm" if wv >= 1000 else f"{n} {wv:.0f} nm", c))
    items += list(extra)
    # The title wraps by words when it is wider than the picture; the swatches wrap onto more rows.
    head = [title]
    if probe.textlength(title, font=font) > w - 24:
        head, cur = [], ""
        for word in title.split(" "):
            if cur and probe.textlength(cur + " " + word, font=font) > w - 24:
                head.append(cur)
                cur = word
            else:
                cur = word if not cur else cur + " " + word
        head.append(cur)
    rows, x, row = [[]], 12, 0
    for text, c in items:
        step = sw + 24 + int(probe.textlength(text, font=font))
        if x + step > w and rows[row]:
            rows.append([])
            row += 1
            x = 12
        rows[row].append((x, text, c))
        x += step
    # One title line and one row of swatches keep the plain layout; extra title lines and swatch rows add height below.
    canvas = Image.new("RGB", (w, h + bar + line * (len(head) - 1) + line * (len(rows) - 1)), (0, 0, 0))
    canvas.paste(Image.fromarray(img), (0, 0))
    d = ImageDraw.Draw(canvas)
    for i, t in enumerate(head):
        d.text((12, h + 6 + line * i), t, fill=(235, 235, 235), font=font)
    y = h + bar // 2 + 2 + line * (len(head) - 1)
    for r in rows:
        for x, text, c in r:
            d.rectangle([x, y, x + sw, y + sw], fill=tuple(int(round(255 * v)) for v in np.clip(c, 0, 1)))
            d.text((x + sw + 8, y - 2), text, fill=(220, 220, 220), font=font)
        y += line
    return canvas


# recipes.json and the contact sheet; called after every set so a stopped run leaves them behind.
def write_outputs(outdir, recipes, made):
    json.dump(recipes, open(os.path.join(outdir, "recipes.json"), "w"), indent=1)
    # A contact sheet: each set's two palettes side by side.
    if made:
        thumbs = []
        for fn in made:
            im = Image.open(os.path.join(outdir, fn)).convert("RGB")
            im.thumbnail((900, 900))
            thumbs.append(im)
        cols = 2
        rows_n = (len(thumbs) + cols - 1) // cols
        cell = max(max(t.size[0] for t in thumbs), 1), max(max(t.size[1] for t in thumbs), 1)
        sheet = Image.new("RGB", (cols * cell[0], rows_n * cell[1]), (0, 0, 0))
        for i, t in enumerate(thumbs):
            sheet.paste(t, ((i % cols) * cell[0], (i // cols) * cell[1]))
        sheet.save(os.path.join(outdir, "sheet.jpg"), quality=85)


def main():
    outdir, rows_path, specs = sys.argv[1], sys.argv[2], sys.argv[3:]
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(WORK, exist_ok=True)
    rows = json.load(gzip.open(rows_path))["rows"]
    recipes, made = [], []
    for spec in specs:
        parts = spec.split("|")
        collection, target, proposal, instrument, label = parts[:5]
        black = float(parts[5]) if len(parts) > 5 and parts[5] else 1.0  # percentile of each filter taken as black: 1 keeps gas haze, 40 to 60 gives a black sky
        chosen = pick_all(rows, collection, target, proposal, instrument)
        log(f"== {label}: {len(chosen)} filters: {[r['filters'] for r in chosen]}")
        if len(chosen) < 3:
            log("fewer than three plain filters; skipped")
            continue
        cache = os.path.join(WORK, f"{label}-aligned.npz")
        try:
            if os.path.exists(cache):
                z = np.load(cache, allow_pickle=True)
                used_names = json.loads(str(z["names"]))
                chosen = [r for n in used_names for r in chosen if r["filters"] == n]
                aligned = [z[f"a{i}"].astype(np.float32) for i in range(len(chosen))]
                grid = json.loads(str(z["grid"]))
                log("aligned arrays found in the cache; no download")
            elif NO_DOWNLOAD or label in CACHE_ONLY:
                log(f"{label}: no cached aligned arrays and downloads are off for this set; skipped")
                continue
            else:
                aligned, grid, chosen = align(chosen, WORK, label)
                np.savez_compressed(cache, grid=json.dumps(grid), names=json.dumps([r["filters"] for r in chosen]), **{f"a{i}": a.astype(np.float32) for i, a in enumerate(aligned)})
        except Exception as e:  # noqa: BLE001
            log(f"{label} failed: {e}")
            continue
        names = [r["filters"] for r in chosen]
        waves = [(r["em_min"] + r["em_max"]) / 2 for r in chosen]
        raw = aligned
        xr, xinfo = None, None
        if label in XRAY:
            try:
                xr, xinfo = xray_counts(label, grid, XRAY[label])
            except Exception as e:  # noqa: BLE001
                log(f"{label}: X-ray layer failed: {e}; the set is skipped")
                continue
        if "export" in VARIANTS:  # no pictures: save a one-third-size copy of the aligned arrays, so the finishing steps can be worked out on this computer
            f = 3
            small = []
            for a in aligned:
                h, w = (a.shape[0] // f) * f, (a.shape[1] // f) * f
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    small.append(np.nanmean(a[:h, :w].reshape(h // f, f, w // f, f), axis=(1, 3)).astype(np.float32))
            os.makedirs(os.path.join(outdir, "samples"), exist_ok=True)
            extra = {}
            if xr is not None:
                h, w = (xr.shape[0] // f) * f, (xr.shape[1] // f) * f
                extra = {"xray": xr[:h, :w].reshape(h // f, f, w // f, f).sum(axis=(1, 3)).astype(np.float32), "xinfo": json.dumps(xinfo)}
            np.savez_compressed(os.path.join(outdir, "samples", f"{label}.npz"), names=json.dumps(names), waves=json.dumps([float(x) for x in waves]),
                                grid=json.dumps({k: v for k, v in grid.items() if k != "wcs"}), **extra, **{f"a{i}": a for i, a in enumerate(small)})
            log(f"{label}: samples saved ({small[0].shape[1]}x{small[0].shape[0]} px per filter)")
            continue
        fin = FINISH.get(label) if FINISH_ON else None
        if fin and fin.get("sky"):   # crop to a square of half-width `half` arcsec around a sky position (RA, Dec in degrees)
            ra, dec, half = fin["sky"]
            gw = WCS(fits.Header.fromstring(grid["wcs"]))
            cx, cy = gw.world_to_pixel_values(ra, dec)
            r = int(round(half / scale_of(gw)))
            cy, cx = int(round(float(cy))), int(round(float(cx)))
            y0, y1, x0, x1 = max(0, cy - r), min(raw[0].shape[0], cy + r), max(0, cx - r), min(raw[0].shape[1], cx + r)
            if xr is not None:
                log(f"{label}: the X-ray layer cannot follow a crop by sky position; the set is skipped")
                continue
            raw = aligned = [x[y0:y1, x0:x1] for x in raw]
            log(f"{label}: cropped to {x1 - x0}x{y1 - y0} px around RA {ra} Dec {dec}")
        if fin:
            raw = aligned = finish_arrays(raw, fin, float(os.environ.get("FINISH_SCALE", "1")))
            black = fin.get("black", black)
            if xr is not None and (fin.get("box") or fin.get("dense") or fin.get("planet")):
                log(f"{label}: the X-ray layer cannot follow a crop by box, density or planet; the set is skipped")
                continue
            log(f"{label}: finishing steps applied: {sorted(fin)}")
        xl = xray_layer(xr, fin or {}, float(os.environ.get("FINISH_SCALE", "1"))) if xr is not None else None   # on the whole grid, before the crop, so its edges are not smoothed against nothing
        old_wanted = any(k not in RANKED for k in PALETTES)
        aligned = [flatten(a) for a in raw] if (FLATTEN and old_wanted) else raw
        # Keep only the sky every filter saw: the bounding box of the pixels all filters cover (or all but one, if that leaves too little).
        for need in (len(aligned), len(aligned) - 1):
            cover = np.sum([np.isfinite(a) for a in aligned], axis=0) >= need
            ys, xs = np.where(cover)
            if ys.size and cover.mean() > 0.25:
                break
        if ys.size:
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            aligned = [a[y0:y1, x0:x1] for a in aligned]
            raw = [a[y0:y1, x0:x1] for a in raw]
            if xl is not None:
                xl = xl[y0:y1, x0:x1]
            log(f"cropped to {x1 - x0}x{y1 - y0} px where {need} of {len(names)} filters have data")
        norm = [normalize(a) for a in aligned] if old_wanted else []
        stack = [n[0] for n in norm]
        for name, (x, top) in zip(names, norm):
            log(f"{name}: signal scale {top:.4g}, lit pixels {(x > 0.05).mean() * 100:.1f}%")
        for pname, pal in PALETTES.items():
            if pname in RANKED:
                variants = [(e, *normalize_keep_haze(raw, e, black)) for e in NORM_EXPS]
            else:
                variants = [(None, stack, None, None)]
            for e, stk, tops, weights in variants:
              for vname in (VARIANTS if e is not None else ["base"]):
                tw = TWEAKS.get(vname, TWEAKS["base"])
                stk2 = [x * tw["gain"].get(n, 1.0) for x, n in zip(stk, names)]
                cols = None
                if e is not None:
                    base_cols = RANKED[pname](len(stk2))
                    cols = [c * (1 - tw["soften"].get(n, 0.0)) + SOFT_PINK * tw["soften"].get(n, 0.0) for c, n in zip(base_cols, names)]
                img, colors = compose(stk2, waves, pal, RANKED.get(pname), cols, tw.get("whiten", 0.0), tw.get("stretch") or (fin or {}).get("stretch"), xl)
                suffix = "" if e is None else f"-n{int(round(e * 100)):02d}" + ("" if vname == "base" else f"-{vname}")
                where = f"{chosen[0]['obs_collection']} {instrument.split('/')[0]}, {target}, program {proposal}"
                if "+" in collection:
                    where = " + ".join(f"{c} {i.split('/')[0]} {p}" for c, i, p in zip(collection.split("+"), instrument.split("+"), proposal.split("+"))) + (" + Chandra ACIS" if xl is not None else "") + f", {target.split('+')[0]}"
                made_from = "MAST mosaics" if xl is None else "MAST mosaics and Chandra Data Archive events"
                title = f"{where} — {PALETTE_NAMES[pname]}{'' if e is None else f', brightness exponent {e:g}'}{'' if vname == 'base' else f', {vname}'} — made by this project (Raellabot) from {made_from}"
                if fin and fin.get("upscale"):
                    k = fin["upscale"]
                    img = np.array(Image.fromarray(img).resize((img.shape[1] * k, img.shape[0] * k), Image.LANCZOS))
                canvas = legend(img, names, colors, waves, title, [("Chandra X-ray 0.5-7 keV", XRAY_COLOR)] if xl is not None else [])
                fn = f"{label}-{pname}{suffix}.jpg"
                canvas.save(os.path.join(outdir, fn), quality=90)
                made.append(fn)
                recipes.append({
                    "picture": fn, "palette": pname, "collection": collection, "target": target, "proposal_id": proposal, "instrument": instrument,
                    "title": chosen[0].get("obs_title"), "grid": {k: v for k, v in grid.items() if k != "wcs"}, "xray": None if xl is None else {**xinfo, "color": hexof(XRAY_COLOR), "smoothing": "significance-based, several widths blended", "dimmed_brightest_source": (fin or {}).get("xray_dim")}, "stretch": STRETCH, "q": Q, "flipped_upright": FLIP, "brightness_exponent": e, "variant": vname if e is not None else None,
                    "filters": [{"filter": n, "center_nm": round(w, 1), "color": hexof(c), "obs_id": r["obs_id"], "exposure_s": r.get("t_exptime"), "product": r["dataURL"]}
                                for n, w, c, r in zip(names, waves, colors, chosen)],
                    "black_percentile": black if e is not None else None, "finishing": fin,
                    "method": ("level-3 mosaics reduced to one grid by sky coordinates; black point at the given percentile of each filter (1 keeps sky haze), no flattening; each filter scaled to its 99.7th percentile then weighted by (top/largest top)^(1-exponent); hues by rank in wavelength order; Lupton asinh stretch"
                               if e is not None else
                               "level-3 mosaics reduced to one grid by sky coordinates; background removed and 99.7th percentile scaled per filter; colors by wavelength order; Lupton asinh stretch"),
                })
        write_outputs(outdir, recipes, made)
    write_outputs(outdir, recipes, made)
    log(f"done: {len(made)} pictures")
    if not made and "export" not in VARIANTS:
        sys.exit(1)


main()
