# Builds a Hubble + Spitzer subject: Hubble HLSP mosaics from MAST and Spitzer SEIP mosaics from IRSA, resampled onto one sky grid, then exported like any target.
# usage: export_composite.py part targets/<id>.json PARTDIR LAYER_INDEX   (one layer onto the grid, saved as PARTDIR/<i>.npz)
#        export_composite.py build targets/<id>.json PARTDIR OUTDIR      (all saved layers into a manifest)
import gzip, json, os, shutil, sys, time, urllib.parse, urllib.request, warnings
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from scipy.ndimage import binary_erosion, map_coordinates, median_filter, minimum_filter
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "mast"))
import export_target as ex
import color

warnings.simplefilter("ignore", FITSFixedWarning)
UA = os.environ.get("CONTACT_INFO") or "cosmic-darkroom builder"
WORK = os.environ.get("WORK_DIR", "work")
MAST_DL = "https://mast.stsci.edu/api/v0.1/Download/file?uri="
IRSA_SEIP = "https://irsa.ipac.caltech.edu/data/SPITZER/Enhanced/SEIP/images/{a}/{b}/{aor}/{c}/{tile}/{aor}.{tile}{short}.IRAC.{band}.mosaic.fits"


def grid_wcs(src):
    ra, dec = src["center"]
    w, h = src["size_px"]
    s = src["arcsec_per_pixel"] / 3600.0
    g = WCS(naxis=2)
    g.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    g.wcs.crval = [ra, dec]
    g.wcs.crpix = [(w + 1) / 2.0, (h + 1) / 2.0]
    g.wcs.cdelt = [-s, s]
    return g, (h, w)


def sky_of_grid(g, shape):
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    return g.pixel_to_world_values(xx.astype(np.float64), yy.astype(np.float64))


def block_reduce(data, k, zero_empty):
    # Mean over k x k blocks, read a band of rows at a time so a large memory-mapped mosaic is never loaded whole; a block needs half its pixels to count.
    H, W = data.shape
    h, w = H // k, W // k
    out = np.full((h, w), np.nan, dtype=np.float32)
    for r in range(0, h, 64):
        r1 = min(h, r + 64)
        band = np.asarray(data[r * k:r1 * k, :w * k], dtype=np.float32)
        bad = ~np.isfinite(band)
        if zero_empty:
            bad |= band == 0
        v = np.where(bad, 0, band).reshape(r1 - r, k, w, k).sum(axis=(1, 3))
        n = (~bad).reshape(r1 - r, k, w, k).sum(axis=(1, 3))
        out[r:r1] = np.where(n >= 0.5 * k * k, v / np.maximum(n, 1), np.nan)
    return out


def resample(path, ra, dec, out_scale, zero_empty):
    # Puts one mosaic onto the grid (positions given as RA, Dec of every grid pixel); returns NaN where the mosaic has no data.
    with fits.open(path, memmap=True) as hd:
        hdu = next((x for x in hd if x.data is not None and getattr(x.data, "ndim", 0) == 2), None)
        if hdu is None:
            print(f"  {os.path.basename(path)}: no 2-D image inside; skipped", flush=True)
            return None
        w = WCS(hdu.header).celestial
        data = hdu.data
        in_scale = color.scale_of(w)
        k = max(1, int(round(out_scale / in_scale)))
        red = block_reduce(data, k, zero_empty) if k > 1 else np.asarray(data, dtype=np.float32)
        if k == 1 and zero_empty:
            red = np.where(red == 0, np.nan, red)
        x, y = w.world_to_pixel_values(ra, dec)
        xr, yr = (x - (k - 1) / 2.0) / k, (y - (k - 1) / 2.0) / k
        inside = (xr >= 0) & (yr >= 0) & (xr <= red.shape[1] - 1) & (yr <= red.shape[0] - 1)
        out = np.full(ra.shape, np.nan, dtype=np.float32)
        if inside.any():
            m = np.isfinite(red)
            vals = map_coordinates(np.where(m, red, 0), [yr[inside], xr[inside]], order=1, mode="nearest")
            cover = map_coordinates(m.astype(np.float32), [yr[inside], xr[inside]], order=1, mode="nearest")
            out[inside] = np.where(cover > 0.99, vals, np.nan)
        print(f"  {os.path.basename(path)}: {data.shape[1]}x{data.shape[0]} px at {in_scale:.3f}\", block {k}, covers {np.isfinite(out).mean() * 100:.1f}% of the grid", flush=True)
        return out


def fetch(url, path):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=300) as r, open(path + ".part", "wb") as f:
                shutil.copyfileobj(r, f, 1 << 22)
            with open(path + ".part", "rb") as f:
                head = f.read(300)
            if head.startswith(b"\x1f\x8b"):
                with gzip.open(path + ".part", "rb") as g, open(path + ".unz", "wb") as f:
                    shutil.copyfileobj(g, f, 1 << 22)
                os.replace(path + ".unz", path + ".part")
                with open(path + ".part", "rb") as f:
                    head = f.read(300)
            if not head.startswith(b"SIMPLE"):
                raise RuntimeError("not a FITS file, starts with " + repr(head[:200]))
            os.replace(path + ".part", path)
            return path
        except Exception as e:
            print(f"  retry {attempt + 1} for {url}: {e}", flush=True)
            time.sleep(10 * (attempt + 1))
    raise RuntimeError("could not download " + url)


def hubble_files(src, filt):
    # The program's own calibrated, combined Hubble images (one per pointing) for this filter, from MAST's standard HST collection.
    from astroquery.mast import Observations
    h = src["hubble"]
    ra0, dec0 = src["center"]
    obs = Observations.query_criteria(obs_collection="HST", proposal_id=h["program"], instrument_name=h["instrument"], filters=filt)
    keep = []
    for o in obs:
        url = str(o["dataURL"])
        if str(o["dataproduct_type"]) != "image" or int(o["calib_level"]) != 3 or not url.endswith(".fits") or str(o["intentType"]) != "science":
            continue
        d = np.hypot((float(o["s_ra"]) - ra0) * np.cos(np.radians(dec0)), float(o["s_dec"]) - dec0)
        if d > h.get("radius_deg", 0.45):
            continue
        keep.append((os.path.basename(url), url))
    print(f"{filt}: {len(obs)} rows in MAST, {len(keep)} combined images near the grid", flush=True)
    return keep


def crop_piece(a):
    # Smallest box holding a piece's data, so hundreds of pieces fit in one file.
    m = np.isfinite(a)
    if not m.any():
        return None
    ys, xs = np.where(m.any(axis=1))[0], np.where(m.any(axis=0))[0]
    y0, y1, x0, x1 = ys[0], ys[-1] + 1, xs[0], xs[-1] + 1
    return int(y0), int(x0), a[y0:y1, x0:x1].astype(np.float32)


def collect(items, getter, ra, dec, scale, zero_empty):
    pieces, ids = [], []
    for name, ref in items:
        path = os.path.join(WORK, name)
        getter(ref, path)
        a = resample(path, ra, dec, scale, zero_empty)
        os.remove(path)
        c = None if a is None else crop_piece(a)
        if c is not None:
            pieces.append(c)
            ids.append(name.replace(".fits", ""))
    return pieces, ids


def save_pieces(path, pieces, ids, shape, extra=None):
    d = {"shape": np.array(shape), "ids": json.dumps(ids), "n": np.array(len(pieces))}
    for k, (y0, x0, c) in enumerate(pieces):
        d[f"o{k}"] = np.array([y0, x0])
        d[f"p{k}"] = c.astype(np.float16) if np.nanmax(np.abs(c)) < 6e4 else c
    d.update(extra or {})
    np.savez_compressed(path, **d)


def load_pieces(path):
    z = np.load(path)
    n = int(z["n"])
    return [(int(z[f"o{k}"][0]), int(z[f"o{k}"][1]), z[f"p{k}"].astype(np.float32)) for k in range(n)], json.loads(str(z["ids"])), tuple(int(v) for v in z["shape"]), z


def level_pieces(pieces, shape, edge=2, min_overlap=400):
    # Hubble pointings were taken at different times with different sky levels; finds one offset per pointing (least squares over every overlap) so the seams vanish, then averages.
    full = []
    for y0, x0, c in pieces:
        m = np.isfinite(c)
        if edge:
            m = binary_erosion(m, iterations=edge)
        full.append((y0, x0, np.where(m, c, np.nan)))
    n = len(full)
    rows, diffs = [], []
    for i in range(n):
        yi, xi, ci = full[i]
        for j in range(i + 1, n):
            yj, xj, cj = full[j]
            ya, yb = max(yi, yj), min(yi + ci.shape[0], yj + cj.shape[0])
            xa, xb = max(xi, xj), min(xi + ci.shape[1], xj + cj.shape[1])
            if yb - ya < 10 or xb - xa < 10:
                continue
            a = ci[ya - yi:yb - yi, xa - xi:xb - xi]
            b = cj[ya - yj:yb - yj, xa - xj:xb - xj]
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() < min_overlap:
                continue
            rows.append((i, j))
            diffs.append(float(np.median(a[m] - b[m])))
    off = np.zeros(n)
    if rows:
        A = lil_matrix((len(rows) + 1, n))
        for k, (i, j) in enumerate(rows):
            A[k, i], A[k, j] = 1, -1
        A[len(rows), :] = 1
        off = lsqr(A.tocsr(), np.array(diffs + [0.0]))[0]
    print(f"  leveled {n} pieces over {len(rows)} overlaps; offsets span {np.ptp(off):.4g}", flush=True)
    tot = np.zeros(shape, np.float64)
    cnt = np.zeros(shape, np.float32)
    for (y0, x0, c), o in zip(full, off):
        m = np.isfinite(c)
        sl = (slice(y0, y0 + c.shape[0]), slice(x0, x0 + c.shape[1]))
        tot[sl][m] += c[m] - o
        cnt[sl][m] += 1
    out = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan).astype(np.float32)
    lo = np.nanpercentile(out, 0.5)
    return out - min(lo, 0)


def line_footprint(length, angle_deg):
    k = np.zeros((length, length), bool)
    t = np.radians(angle_deg)
    for s in np.linspace(-(length - 1) / 2, (length - 1) / 2, 4 * length):
        k[int(round((length - 1) / 2 - s * np.sin(t))), int(round((length - 1) / 2 + s * np.cos(t)))] = True
    return k


def destreak(a, angle_deg, length=15, k=6.0):
    # Saturated stars bleed into long thin trails along the detector columns; a pixel far above the band across the trail but just as bright along it is part of a trail, and takes the level across it.
    f = np.nan_to_num(a)
    across = median_filter(f, footprint=line_footprint(9, angle_deg + 90))
    r = f - across
    sig = 1.4826 * np.median(np.abs(r[np.isfinite(a)] - np.median(r[np.isfinite(a)]))) or 1e-6
    along = minimum_filter(r, footprint=line_footprint(length, angle_deg))
    m = along > k * sig
    m = np.maximum.reduce([m, np.roll(m, 1, 0), np.roll(m, -1, 0), np.roll(m, 1, 1), np.roll(m, -1, 1)])
    print(f"  removed trails from {m.mean() * 100:.2f}% of the pixels", flush=True)
    return np.where(m & np.isfinite(a), across, a)


def mast_get(uri, path):
    print(f"  fetching {uri}", flush=True)
    fetch(MAST_DL + urllib.parse.quote(uri, safe=":/"), path)


def spitzer_band(src, band, ra, dec, scale):
    # Long- and short-exposure (HDR) mosaics of each tile, kept apart so the build decides how to join them.
    s = src["spitzer"]
    pieces, ids = [], []
    for tile in s["tiles"]:
        aor, sub = tile.split("-")
        for short in ("", "-short"):
            url = IRSA_SEIP.format(a=aor[0], b=aor[1:5], aor=aor, c=sub[-1], tile=tile, short=short, band=band)
            path = os.path.join(WORK, os.path.basename(url))
            fetch(url, path)
            a = resample(path, ra, dec, scale, False)
            os.remove(path)
            c = crop_piece(a)
            if c is not None:
                pieces.append(c)
                ids.append(f"SEIP {tile}{short} IRAC{band}")
    return pieces, ids


def spitzer_join(pieces, ids, shape, mode):
    # mode "short": the short exposures alone, which stay unsaturated over the bright nebula; tiles are leveled to each other like Hubble pointings.
    keep = [p for p, i in zip(pieces, ids) if ("-short" in i) == (mode == "short")]
    return level_pieces(keep, shape, edge=3, min_overlap=2000)


def part(cfg_path, partdir, i):
    cfg = json.load(open(cfg_path))
    src = cfg["source"]
    lay = src["layers"][i]
    os.makedirs(WORK, exist_ok=True)
    os.makedirs(partdir, exist_ok=True)
    g, shape = grid_wcs(src)
    ra, dec = sky_of_grid(g, shape)
    scale = src["arcsec_per_pixel"]
    print(f"== layer {i}: {lay['mission']} {lay['label']} onto a {shape[1]}x{shape[0]} grid at {scale}\"", flush=True)
    if lay["mission"] == "HST":
        pieces, ids = collect(hubble_files(src, lay["filter"]), mast_get, ra, dec, scale, True)
    else:
        pieces, ids = spitzer_band(src, lay["band"], ra, dec, scale)
    print(f"layer {i}: {len(pieces)} pieces kept", flush=True)
    save_pieces(os.path.join(partdir, f"{i}.npz"), pieces, ids, shape)


def combine(src, i, partdir):
    lay = src["layers"][i]
    pieces, ids, shape, _ = load_pieces(os.path.join(partdir, f"{i}.npz"))
    print(f"== layer {i}: {lay['label']} from {len(pieces)} pieces", flush=True)
    if lay["mission"] == "HST":
        a = level_pieces(pieces, shape)
        if src.get("hubble", {}).get("trail_angle") is not None:
            a = destreak(a, src["hubble"]["trail_angle"])
        ids = [x for x in ids]
    else:
        a = spitzer_join(pieces, ids, shape, src["spitzer"].get("join", "short"))
        ids = [x for x in ids if ("-short" in x) == (src["spitzer"].get("join", "short") == "short")]
    return a, ids


def best_box(cover):
    # Largest upright rectangle where every layer has data, found on a coarse grid (histogram method).
    f = 8
    h, w = cover.shape[0] // f, cover.shape[1] // f
    c = cover[:h * f, :w * f].reshape(h, f, w, f).mean(axis=(1, 3)) >= 0.995
    heights = np.zeros(w, int)
    best = (0, 0, 0, 0, 0)
    for y in range(h):
        heights = np.where(c[y], heights + 1, 0)
        stack = []
        for x in range(w + 1):
            hh = heights[x] if x < w else 0
            start = x
            while stack and stack[-1][1] >= hh:
                sx, sh = stack.pop()
                area = sh * (x - sx)
                if area > best[0]:
                    best = (area, y - sh + 1, y + 1, sx, x)
                start = sx
            stack.append((start, hh))
    _, y0, y1, x0, x1 = best
    return y0 * f, y1 * f, x0 * f, x1 * f


def finish_manifest(cfg, outdir):
    src = cfg["source"]
    target_dir = os.path.join(outdir, cfg["id"])
    mpath = os.path.join(target_dir, "manifest.json")
    m = json.load(open(mpath))
    m["credits"] = {"line": src["credit"], "made_by": "RRR + Claude"}
    m["sources"] = src["sources"]
    for layer, lay in zip(m["layers"], src["layers"]):
        layer["primary_source"] = lay["primary_source"]
        if layer.get("exposure_s") is None:
            layer.pop("exposure_s", None)
    if src.get("extra_preset"):
        p = dict(m["presets"][0])
        ep = src["extra_preset"]
        by_label = {l["label"]: l["id"] for l in m["layers"]}
        p["id"], p["name"] = ep["id"], ep["name"]
        p["colors"] = {by_label[k]: v for k, v in ep["colors"].items()}
        p["weights"] = {lid: (round(m["presets"][0]["weights"][lid] * ep.get("weights", {}).get(lab, 1.0), 4)) for lab, lid in by_label.items()}
        m["presets"].append(p)
    with open(mpath, "w") as f:
        json.dump(m, f, indent=1)


def build(cfg_path, partdir, outdir):
    cfg = json.load(open(cfg_path))
    src = cfg["source"]
    g, shape = grid_wcs(src)
    raw, chosen = [], []
    for i, lay in enumerate(src["layers"]):
        a, ids = combine(src, i, partdir)
        raw.append(a)
        c = lay["center_nm"]
        chosen.append({"obs_collection": lay["mission"], "filters": lay["label"], "instrument_name": lay["instrument"], "em_min": c, "em_max": c,
                       "obs_id": ids[0] if ids else lay["label"], "tiles": [{"obs_id": t} for t in ids], "t_exptime": None})
    cover = np.all([np.isfinite(a) for a in raw], axis=0)
    y0, y1, x0, x1 = best_box(cover)
    print(f"every layer covers the box x {x0}-{x1}, y {y0}-{y1} ({(x1 - x0) * src['arcsec_per_pixel'] / 60:.1f} x {(y1 - y0) * src['arcsec_per_pixel'] / 60:.1f} arcmin)", flush=True)
    raw = [a[y0:y1, x0:x1] for a in raw]
    g.wcs.crpix = [g.wcs.crpix[0] - x0, g.wcs.crpix[1] - y0]
    if os.environ.get("SAVE_COMBINED"):
        np.savez_compressed(os.environ["SAVE_COMBINED"], *[a.astype(np.float16) for a in raw])
    grid = {"arcsec_per_pixel": src["arcsec_per_pixel"], "wcs": g.to_header().tostring()}
    fin = cfg.get("finish") or None
    manifest = ex.build(cfg, chosen, raw, grid, None, None, outdir, fin)
    finish_manifest(cfg, outdir)
    print(f"done: {len(manifest['layers'])} layers for {cfg['id']}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "part":
        part(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    else:
        build(sys.argv[2], sys.argv[3], sys.argv[4])
