# Builds a Hubble + Spitzer subject: Hubble HLSP mosaics from MAST and Spitzer SEIP mosaics from IRSA, resampled onto one sky grid, then exported like any target.
# usage: export_composite.py part targets/<id>.json PARTDIR LAYER_INDEX   (one layer onto the grid, saved as PARTDIR/<i>.npz)
#        export_composite.py build targets/<id>.json PARTDIR OUTDIR      (all saved layers into a manifest)
import gzip, json, os, shutil, sys, time, urllib.parse, urllib.request, warnings
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from scipy.ndimage import map_coordinates

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


def accumulate(paths_or_urls, getter, ra, dec, scale, zero_empty):
    tot = np.zeros(ra.shape, np.float64)
    n = np.zeros(ra.shape, np.float32)
    ids = []
    for name, ref in paths_or_urls:
        path = os.path.join(WORK, name)
        getter(ref, path)
        a = resample(path, ra, dec, scale, zero_empty)
        os.remove(path)
        if a is None:
            continue
        m = np.isfinite(a)
        if m.any():
            tot[m] += a[m]
            n[m] += 1
            ids.append(name.replace(".fits", ""))
    return np.where(n > 0, tot / np.maximum(n, 1), np.nan).astype(np.float32), ids


def mast_get(uri, path):
    print(f"  fetching {uri}", flush=True)
    fetch(MAST_DL + urllib.parse.quote(uri, safe=":/"), path)


def spitzer_band(src, band, ra, dec, scale):
    # Long-exposure mosaic, with the saturated core filled from the short-exposure (HDR) mosaic of the same tiles.
    s = src["spitzer"]
    tot = np.zeros(ra.shape, np.float64)
    n = np.zeros(ra.shape, np.float32)
    ids = []
    for tile in s["tiles"]:
        aor, sub = tile.split("-")
        layers = {}
        for short in ("", "-short"):
            url = IRSA_SEIP.format(a=aor[0], b=aor[1:5], aor=aor, c=sub[-1], tile=tile, short=short, band=band)
            path = os.path.join(WORK, os.path.basename(url))
            fetch(url, path)
            layers[short] = resample(path, ra, dec, scale, False)
            os.remove(path)
        L, S = layers[""], layers["-short"]
        both = np.isfinite(L) & np.isfinite(S)
        fill = np.isfinite(S) & ~np.isfinite(L)
        if both.any():
            sv = S[both]
            bright = both & (S > np.percentile(sv, 95))
            fill |= bright & (L < 0.85 * S)
        a = np.where(fill, S, L)
        print(f"  tile {tile} band {band}: {fill.sum()} grid pixels taken from the short exposures", flush=True)
        m = np.isfinite(a)
        tot[m] += a[m]
        n[m] += 1
        if m.any():
            ids.append(f"SEIP {tile} IRAC{band}")
    return np.where(n > 0, tot / np.maximum(n, 1), np.nan).astype(np.float32), ids


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
        files = hubble_files(src, lay["filter"])
        a, ids = accumulate(files, mast_get, ra, dec, scale, True)
    else:
        a, ids = spitzer_band(src, lay["band"], ra, dec, scale)
    print(f"layer {i} covers {np.isfinite(a).mean() * 100:.1f}% of the grid from {len(ids)} files", flush=True)
    np.savez_compressed(os.path.join(partdir, f"{i}.npz"), a=a, ids=json.dumps(ids))


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
        z = np.load(os.path.join(partdir, f"{i}.npz"))
        raw.append(z["a"].astype(np.float32))
        ids = json.loads(str(z["ids"]))
        c = lay["center_nm"]
        chosen.append({"obs_collection": lay["mission"], "filters": lay["label"], "instrument_name": lay["instrument"], "em_min": c, "em_max": c,
                       "obs_id": ids[0] if ids else lay["label"], "tiles": [{"obs_id": t} for t in ids], "t_exptime": None})
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
