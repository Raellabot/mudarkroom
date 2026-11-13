# Exports one target as linear half-float layers plus a manifest.json for the page (usage in README.md).
import gzip, json, math, os, re, sys, time, warnings
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "mast"))
import color

chandra_events = color.chandra_events
EXPORT_SIDE = int(os.environ.get("EXPORT_SIDE", "1600"))
PREVIEW_SIDE = 512
BLACK_FLOOR_PCT = 1.0   # stored layers keep the sky haze; the page raises the black point itself
NORM_EXP = 0.5
F16_MAX = 60000.0
PERCENTILES = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
PRESET_PALETTES = ["stsci", "creativerank", "creative5", "ghoul", "slime", "ghost", "pumpkin", "tide", "blood"]
STRIP_VALUES = [1, 25, 50, 75]
XRAY_ID = "chandra-acis-i"


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")


# Block mean down to at most `side` pixels on the long side; returns the array and the reduction factor.
def shrink(a, side):
    f = max(1, int(math.ceil(max(a.shape) / side)))
    if f == 1:
        return a.astype(np.float32), 1
    h, w = (a.shape[0] // f) * f, (a.shape[1] // f) * f
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmean(a[:h, :w].reshape(h // f, f, w // f, f), axis=(1, 3)).astype(np.float32), f


def write_f16(path, a):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    h = np.clip(np.nan_to_num(a, nan=0.0), 0, F16_MAX).astype("<f2")
    with gzip.GzipFile(path, "wb", compresslevel=9, mtime=0) as f:
        f.write(np.ascontiguousarray(h).tobytes())


# The X-ray photons smoothed as in mudarkroom's xray_layer, but before the asinh stretch so the page can stretch them; also the softening level and the brightest source.
def xray_linear(counts, fin, scale=1.0):
    sigmas = tuple(1.5 * scale * 1.6 ** i for i in range(7))
    d, bg = chandra_events.smooth_significant(counts, sigmas, fin.get("xray_sig", 1.5))
    d = gaussian_filter(d, 0.6 * scale, mode="reflect").astype(np.float32)
    top = float(np.percentile(d[d > 0], 99.9)) if (d > 0).any() else 1.0
    soft = fin.get("xray_soft", 1.0) * bg
    peak = np.unravel_index(d.argmax(), d.shape)
    return d, top, soft, (int(peak[0]), int(peak[1]))


def common_box(arrs):
    for need in (len(arrs), len(arrs) - 1):
        cover = np.sum([np.isfinite(a) for a in arrs], axis=0) >= need
        ys, xs = np.where(cover)
        if ys.size and cover.mean() > 0.25:
            return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return 0, arrs[0].shape[0], 0, arrs[0].shape[1]


def split_module(raw, side):
    # Two-module fields have a wide empty stripe between the modules: cuts there and keeps the left or the right one. The narrow chip gaps inside a module are shorter, so the widest empty run is the stripe.
    covered = np.all([np.isfinite(a) for a in raw], axis=0).mean(axis=0) > 0.05
    xs = np.where(covered)[0]
    best, start = (0, 0, 0), None
    for x in range(xs.min(), xs.max() + 2):
        empty = x <= xs.max() and not covered[x]
        if empty and start is None:
            start = x
        elif not empty and start is not None:
            best = max(best, (x - start, start, x))
            start = None
    width, g0, g1 = best
    if width < 20:
        raise RuntimeError("no empty stripe between modules was found")
    print(f"two modules: empty stripe of {width} px at x {g0}-{g1}; keeping the {side} one", flush=True)
    cut = (slice(None), slice(0, g0)) if side == "left" else (slice(None), slice(g1, None))
    raw = [a[cut] for a in raw]
    y0, y1, x0, x1 = common_box(raw)
    return [a[y0:y1, x0:x1] for a in raw]


def build(cfg, chosen, raw, grid, xr, xinfo, outdir, fin):
    names = [r["filters"] for r in chosen]
    waves = [(r["em_min"] + r["em_max"]) / 2 for r in chosen]
    fscale = float(os.environ.get("FINISH_SCALE", "1"))
    if fin and fin.get("module") and xr is None:
        # cut to one module first so a box crop in the finishing table lands inside that module
        raw = split_module(raw, fin["module"])
        fin = {k: v for k, v in fin.items() if k != "module"}
    if fin:
        if xr is not None and (fin.get("box") or fin.get("dense") or fin.get("planet")):
            raise RuntimeError("the X-ray layer cannot follow a crop by box, density or planet")
        raw = color.finish_arrays(raw, fin, fscale)
    d, xtop, xsoft, peak = (None, None, None, None)
    if xr is not None:
        d, xtop, xsoft, peak = xray_linear(xr, fin or {}, fscale)
    y0, y1, x0, x1 = common_box(raw)
    raw = [a[y0:y1, x0:x1] for a in raw]
    if fin and fin.get("module"):
        raw = split_module(raw, fin["module"])
    if fin and fin.get("cbox") and xr is None:
        # crop by fractions of the picture left after the common-coverage crop (rows run bottom to top)
        cy0, cy1, cx0, cx1 = fin["cbox"]
        h, w = raw[0].shape
        raw = [a[int(cy0 * h):int(cy1 * h), int(cx0 * w):int(cx1 * w)] for a in raw]
    if d is not None:
        d = d[y0:y1, x0:x1]
        peak = (peak[0] - y0, peak[1] - x0)
    H, W = raw[0].shape
    print(f"cropped to {W}x{H} px; {len(raw)} optical layers" + (" + X-ray" if d is not None else ""), flush=True)
    tops, stored = [], []
    for a in raw:
        v = a[np.isfinite(a)]
        x = np.clip(a - np.percentile(v, BLACK_FLOOR_PCT), 0, None)
        top = float(np.percentile(x[np.isfinite(x)], 99.7)) or float(np.nanmax(x)) or 1.0
        tops.append(top)
        stored.append((x / top)[::-1])   # upright, like the pictures made by mudarkroom
    big = max(tops) or 1.0
    weights = [round((t / big) ** (1.0 - NORM_EXP), 4) for t in tops]
    target_dir = os.path.join(outdir, cfg["id"])
    layers, full_shape, prev_shape = [], None, None
    ids = []
    full_arrays, prev_arrays = [], []
    for r, a, top, wgt in zip(chosen, stored, tops, weights):
        lid = slug(f"{r['obs_collection']}-{r['filters']}")
        ids.append(lid)
        full, f = shrink(a, EXPORT_SIDE)
        prev, pf = shrink(full, PREVIEW_SIDE)
        full_shape, prev_shape = full.shape, prev.shape
        vals = full[np.isfinite(full)]
        pct = {str(p): round(float(np.percentile(vals, p)), 6) for p in PERCENTILES}
        write_f16(os.path.join(target_dir, "layers", f"{lid}.f16.gz"), full)
        write_f16(os.path.join(target_dir, "layers", f"{lid}.preview.f16.gz"), prev)
        full_arrays.append(np.nan_to_num(full, nan=0.0))
        prev_arrays.append(np.nan_to_num(prev, nan=0.0))
        layers.append({
            "id": lid, "mission": r["obs_collection"], "instrument": r["instrument_name"], "label": r["filters"], "kind": "optical-ir",
            "center_nm": round((r["em_min"] + r["em_max"]) / 2, 1),
            "file": f"layers/{lid}.f16.gz", "preview_file": f"layers/{lid}.preview.f16.gz",
            "encoding": {"format": "float16-le-gzip", "scale": top, "percentiles": pct},
            "default_color": "#ffffff", "default_weight": wgt,
            "obs_ids": [str(t["obs_id"]) for t in r.get("tiles", [r])], "exposure_s": r.get("t_exptime"),
        })
        r_scale = f
    xl_full = None
    if d is not None:
        xs = (d / xtop)[::-1]
        full, f = shrink(xs, EXPORT_SIDE)
        prev, pf = shrink(full, PREVIEW_SIDE)
        write_f16(os.path.join(target_dir, "layers", f"{XRAY_ID}.f16.gz"), full)
        write_f16(os.path.join(target_dir, "layers", f"{XRAY_ID}.preview.f16.gz"), prev)
        soft_rel = xsoft / xtop
        dim = None
        if fin and fin.get("xray_dim"):
            strength, radius = fin["xray_dim"]
            dim = {"strength": strength, "radius_px": round(radius * fscale / f, 2), "x": round(peak[1] / f, 2), "y": round((H - 1 - peak[0]) / f, 2)}
        layers.append({
            "id": XRAY_ID, "mission": "Chandra", "instrument": "ACIS-I", "label": "X-ray 0.5-7 keV", "kind": "xray",
            "file": f"layers/{XRAY_ID}.f16.gz", "preview_file": f"layers/{XRAY_ID}.preview.f16.gz",
            "encoding": {"format": "float16-le-gzip", "scale": xtop, "noise_floor": round(soft_rel, 8)},
            "blend": "screen", "default_color": color.hexof(color.XRAY_COLOR), "default_weight": 1.0,
            "obs_ids": [str(o) for o in (xinfo or {}).get("observations", [])],
            "exposure_s": round(((xinfo or {}).get("exposure_ks") or 0.0) * 1000.0, 1),
            **({"dim": dim} if dim else {}),
        })
        xl_full = full
    stretch = (fin or {}).get("stretch") or color.STRETCH
    black = (fin or {}).get("black", BLACK_FLOOR_PCT)
    presets = []
    for pname in PRESET_PALETTES:
        cols = color.RANKED[pname](len(names))
        colors = {lid: color.hexof(c) for lid, c in zip(ids, cols)}
        wts = {lid: w for lid, w in zip(ids, weights)}
        if d is not None:
            colors[XRAY_ID] = color.hexof(color.XRAY_COLOR)
            wts[XRAY_ID] = 1.0
        presets.append({"id": pname, "name": color.PALETTE_NAMES[pname], "colors": colors, "weights": wts, "stretch": stretch, "softening": color.Q,
                        "black_point": black, "white_point": 100, "saturation": 1.0})
    first = presets[0]
    for layer in layers:
        layer["default_color"] = first["colors"][layer["id"]]
    manifest = {
        "version": 1, "build": os.environ.get("GITHUB_RUN_ID") or time.strftime("%Y%m%d%H%M%S", time.gmtime()), "id": cfg["id"], "name": cfg["name"], "blurb": cfg.get("blurb", ""),
        "grid": {"width": int(full_shape[1]), "height": int(full_shape[0]), "arcsec_per_pixel": round(grid["arcsec_per_pixel"] * r_scale, 6),
                 "preview_width": int(prev_shape[1]), "preview_height": int(prev_shape[0])},
        "layers": layers, "presets": presets, "black_point_strip": {"values": STRIP_VALUES},
        "credits": {"line": "Data: NASA, ESA, CSA, STScI (Hubble, Webb); NASA/CXC (Chandra). Developed by RRR + Claude. Not made or endorsed by NASA, ESA or STScI.", "made_by": "RRR + Claude"},
        "sources": [{"name": "MAST, the Mikulski Archive for Space Telescopes (Hubble and Webb)", "url": "https://mast.stsci.edu"},
                    {"name": "Chandra Data Archive", "url": "https://cxc.harvard.edu/cda/"}],
    }
    with open(os.path.join(target_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    check(target_dir, ids, waves, prev_arrays, full_arrays, weights, xl_full, first, stretch)
    return manifest


# Pictures to look at: each preview layer in gray, and a composite of the first preset made the way mudarkroom does it (this checks the stored layers end to end).
def check(target_dir, ids, waves, prev_arrays, full_arrays, weights, xl_full, preset, stretch):
    out = os.path.join(target_dir, "check")
    os.makedirs(out, exist_ok=True)
    for lid, a in zip(ids, prev_arrays):
        g = np.clip(np.arcsinh(a * 8) / np.arcsinh(8), 0, 1)
        Image.fromarray((g * 255).astype(np.uint8)).save(os.path.join(out, f"{lid}.png"))
    stk = [a * w for a, w in zip(full_arrays, weights)]
    cols = [np.array([int(preset["colors"][lid][i:i + 2], 16) / 255 for i in (1, 3, 5)]) for lid in ids]
    xlayer = None
    if xl_full is not None:
        # The X-ray layer as the page will draw it: asinh of the stored value over the softening level, then the dimming.
        xl = json.load(open(os.path.join(target_dir, "manifest.json")))["layers"][-1]
        noise = xl["encoding"]["noise_floor"]
        xlayer = np.clip(np.arcsinh(xl_full / noise) / np.arcsinh(1.0 / noise), 0, 1).astype(np.float32)
        if xl.get("dim"):
            dim = xl["dim"]
            yy, xx = np.mgrid[0:xlayer.shape[0], 0:xlayer.shape[1]]
            xlayer = xlayer * (1 - dim["strength"] * np.exp(-((yy - dim["y"]) ** 2 + (xx - dim["x"]) ** 2) / (2 * dim["radius_px"] ** 2)))
    img, _ = color.compose(stk, waves, None, None, cols, 0.0, stretch, xlayer)
    Image.fromarray(np.ascontiguousarray(img[::-1])).save(os.path.join(out, f"composite-{preset['id']}.jpg"), quality=88)
    # A small picture for the page's subject list.
    thumb = Image.fromarray(np.ascontiguousarray(img[::-1]))
    thumb.thumbnail((288, 288))
    thumb.save(os.path.join(target_dir, "thumb.jpg"), quality=80)


def load_aligned(label, chosen):
    cache = os.path.join(color.WORK, f"{label}-aligned.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        used = json.loads(str(z["names"]))
        chosen = [r for n in used for r in chosen if r["filters"] == n]
        return chosen, [z[f"a{i}"].astype(np.float32) for i in range(len(chosen))], json.loads(str(z["grid"]))
    if color.NO_DOWNLOAD:
        raise RuntimeError(f"{label}: no cached aligned arrays and downloads are off")
    aligned, grid, chosen = color.align(chosen, color.WORK, label)
    np.savez_compressed(cache, grid=json.dumps(grid), names=json.dumps([r["filters"] for r in chosen]), **{f"a{i}": a.astype(np.float32) for i, a in enumerate(aligned)})
    return chosen, aligned, grid


def upscale(a, k):
    # Smooth enlargement by a whole number, keeping the empty (NaN) edges empty.
    from scipy.ndimage import zoom
    m = np.isfinite(a)
    z = zoom(np.where(m, a, 0).astype(np.float32), k, order=3)
    return np.where(zoom(m.astype(np.float32), k, order=1) > 0.99, z, np.nan).astype(np.float32)


def sky_crop(raw, grid, ra, dec, half):
    # Crops every layer to a square of half-width `half` arcsec around a sky position (RA, Dec in degrees), as mudarkroom does.
    gw = color.WCS(color.fits.Header.fromstring(grid["wcs"]))
    cx, cy = gw.world_to_pixel_values(ra, dec)
    r = int(round(half / color.scale_of(gw)))
    cy, cx = int(round(float(cy))), int(round(float(cx)))
    y0, y1, x0, x1 = max(0, cy - r), min(raw[0].shape[0], cy + r), max(0, cx - r), min(raw[0].shape[1], cx + r)
    print(f"cropped to {x1 - x0}x{y1 - y0} px around RA {ra} Dec {dec}", flush=True)
    return [x[y0:y1, x0:x1] for x in raw]


def main():
    cfg_path, outdir, rows_path = sys.argv[1:4]
    cfg = json.load(open(cfg_path))
    os.makedirs(color.WORK, exist_ok=True)
    rows = json.load(gzip.open(rows_path))["rows"]
    collection, target, proposal, instrument, label = cfg["spec"].split("|")[:5]
    chosen = color.pick_all(rows, collection, target, proposal, instrument)
    print(f"== {label}: {len(chosen)} filters: {[r['filters'] for r in chosen]}", flush=True)
    if len(chosen) < 3:
        sys.exit("fewer than three plain filters")
    chosen, raw, grid = load_aligned(label, chosen)
    xr, xinfo = None, None
    if label in color.XRAY:
        xr, xinfo = color.xray_counts(label, grid, color.XRAY[label])
    # The finishing table is shared with mudarkroom; a target's own "finish" block adds to it or overrides it.
    fin = {**((color.FINISH.get(label) if color.FINISH_ON else None) or {}), **cfg.get("finish", {})} or None
    if fin and fin.get("sky"):
        if xr is not None:
            raise RuntimeError("the X-ray layer cannot follow a crop by sky position")
        raw = sky_crop(raw, grid, *fin["sky"])
    if fin and fin.get("upscale"):
        raw = [upscale(a, int(fin["upscale"])) for a in raw]
    manifest = build(cfg, chosen, raw, grid, xr, xinfo, outdir, fin)
    print(f"done: {len(manifest['layers'])} layers for {cfg['id']}", flush=True)


if __name__ == "__main__":
    main()
