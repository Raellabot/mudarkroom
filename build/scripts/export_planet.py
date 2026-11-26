# Exports one planet or moon target (Cassini, JunoCam or Voyager colour frames) as linear half-float layers plus a manifest.json for the page.
# usage: export_planet.py targets/<id>.json OUTDIR
import json, os, sys, time, urllib.request
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("", "mast", "planets"):
    sys.path.insert(0, os.path.join(HERE, sub))
import color
import export_target as ex

STRIP_VALUES = [1, 25, 50, 75]
PRESET_PALETTES = ["stsci", "creativerank", "creative5", "ghoul", "slime", "ghost", "pumpkin", "tide", "blood"]

# Band centres in nm (blue, green, red). JunoCam's filters are broad: 420-520, 500-600 and 600-800 nm.
JUNO_NM = {"Blue": 470.0, "Green": 550.0, "Red": 700.0}
CASSINI_NM = {"Blue": 440.0, "Green": 568.0, "Red": 650.0}
# Voyager ISS filter centers in nm (PDS Ring-Moon Systems Node instrument pages, after Danielson et al. 1981); the same for both spacecraft and both cameras.
VOYAGER_NM = {"Violet": 400.0, "Blue": 480.0, "Green": 585.0, "Orange": 615.0}

# Defaults for how a planet picture is developed at first: black point and stretch for the page, saturation, and a gentle sharpening of the frames (sigma in pixels, amount). A target's "finish" block overrides these.
FINISH = {
    "junocam": {"black": 5, "stretch": 0.4, "saturation": 1.4, "unsharp": [2.0, 0.9], "planet_percentiles": True},
    "cassini": {},
    "voyager": {"black": 2, "saturation": 1.3},
}

CREDITS = {
    "cassini": {
        "line": "Data: NASA/JPL-Caltech/Space Science Institute (Cassini). Developed by RRR + Claude. Not made or endorsed by NASA, JPL or the Space Science Institute.",
        "sources": [{"name": "Cassini ISS images, PDS Ring-Moon Systems Node (SETI Institute)", "url": "https://opus.pds-rings.seti.org/opus/"}],
    },
    "voyager": {
        "line": "Data: NASA/JPL-Caltech (Voyager). Developed by RRR + Claude. Not made or endorsed by NASA or JPL.",
        "sources": [{"name": "Voyager ISS images, PDS Ring-Moon Systems Node (SETI Institute)", "url": "https://opus.pds-rings.seti.org/opus/"}],
    },
    "junocam": {
        "line": "Data: Courtesy NASA/JPL-Caltech (JunoCam). Developed by RRR + Claude. Not made or endorsed by NASA or JPL.",
        "sources": [{"name": "JunoCam images, Planetary Data System (NASA JPL)", "url": "https://planetarydata.jpl.nasa.gov/img/data/juno/"}],
    },
}


def get(url, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "cosmic-darkroom (github.com/Raellabot/cosmic-darkroom)"})
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.read()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(3 * (i + 1))


def cassini_layers(src):
    import cassini
    chans, info = cassini.layers(src["set"], src["start"], src["opus_target"], src.get("align_box"))
    cam = info["camera"]
    ids = dict(zip(("red", "grn", "blu"), info["opus_ids"]))
    out = []
    for name, key in (("Blue", "blu"), ("Green", "grn"), ("Red", "red")):
        out.append(dict(
            name=name, array=chans[key], center=CASSINI_NM[name], mission="Cassini", instrument=f"ISS {cam}",
            obs_id=ids[key], exposure=info["exposure_s"][key],
            link=f"https://opus.pds-rings.seti.org/opus/#/opusid={ids[key]}",
        ))
    meta = {k: info[k] for k in ("shift_red", "shift_blue", "peak_red", "peak_blue")}
    return out, meta, 1.0


def junocam_layers(src, workdir):
    import junocam
    pid, orbit, vol = src["product"], src["orbit"], src["volume"]
    url = f"https://planetarydata.jpl.nasa.gov/img/data/juno/{vol}/EXTRAS/FULL/RDR/JUPITER/ORBIT_{orbit}/{pid}.PNG"
    path = os.path.join(workdir, f"{pid}.png")
    if not os.path.exists(path):
        open(path, "wb").write(get(url))
    rgb, stats = junocam.process(path)
    if rgb is None:
        raise RuntimeError(f"{pid}: {stats}")
    print("junocam stats:", stats, flush=True)
    out = []
    for name, ch in (("Blue", 2), ("Green", 1), ("Red", 0)):
        out.append(dict(
            name=name, array=rgb[..., ch], center=JUNO_NM[name], mission="Juno", instrument="JunoCam",
            obs_id=pid, exposure=None,
            link=f"https://planetarydata.jpl.nasa.gov/img/data/juno/{vol}/DATA/RDR/JUPITER/ORBIT_{orbit}/{pid}.LBL",
        ))
    return out, {k: stats[k] for k in ("framelets", "rows", "repaired_frac") if k in stats}, 1.0


def voyager_layers(src):
    import voyager
    chans, info = voyager.layers(src)
    cam = src["camera"]
    sc = src["spacecraft"]
    out = []
    for name in ("Violet", "Blue", "Green", "Orange"):
        key = name.lower()
        if key not in chans:
            continue
        out.append(dict(
            name=name, array=chans[key], center=VOYAGER_NM[name], mission="Voyager", instrument=f"ISS {cam}",
            obs_id=src[key], exposure=None, link=f"https://opus.pds-rings.seti.org/opus/#/opusid={src[key]}",
        ))
    print("voyager", sc, cam, "align:", json.dumps(info["align"]), "trim:", json.dumps(info.get("trim")), flush=True)
    return out, {"align": info["align"]}, 1.0


def sharpen(chans, sigma, amount):
    # Unsharp mask that stays inside the planet: blurs with a weight of 1 on planet pixels and 0 on sky, so the limb gets no halo. Every channel gets the same kernel, so the alignment holds.
    from scipy.ndimage import gaussian_filter
    m = (np.max(np.stack(chans), axis=0) > 0).astype(np.float32)
    wm = gaussian_filter(m, sigma)
    out = []
    for x in chans:
        blur = gaussian_filter(x * m, sigma) / np.maximum(wm, 1e-4)
        out.append(np.clip(x + amount * (x - blur), 0, None) * m)
    return out


def build(cfg, layers_in, outdir, credits, kind=None):
    target_dir = os.path.join(outdir, cfg["id"])
    fin = {**FINISH.get(kind, {}), **cfg.get("finish", {})}
    chans = [np.clip(np.nan_to_num(L["array"].astype(np.float32), nan=0.0), 0, None) for L in layers_in]   # the dark sky is already taken off; unlike telescope mosaics, the 1st percentile here is not haze
    if fin.get("unsharp"):
        chans = sharpen(chans, *fin["unsharp"])
    tops, stored = [], []
    for L, x in zip(layers_in, chans):
        top = float(np.percentile(x, 99.7)) or float(x.max()) or 1.0
        tops.append(top)
        stored.append(x / top)
    big = max(tops)
    weights = [round(t / big, 4) for t in tops]   # linear: keeps the colour balance of the frames (red, green and blue as measured)
    layers, ids, full_arrays, prev_arrays = [], [], [], []
    full_shape = prev_shape = None
    for L, a, top, wgt in zip(layers_in, stored, tops, weights):
        lid = f"{L['mission'].lower()}-{L['name'].lower()}"
        ids.append(lid)
        full, f = ex.shrink(a, ex.EXPORT_SIDE)
        prev, pf = ex.shrink(full, ex.PREVIEW_SIDE)
        full_shape, prev_shape = full.shape, prev.shape
        vals = full[np.isfinite(full)]
        if fin.get("planet_percentiles"):
            vals = vals[vals > 0]   # percentiles over the planet only, so the black point slider means the same thing whether the frame is mostly sky or mostly planet
        pct = {str(p): round(float(np.percentile(vals, p)), 6) for p in ex.PERCENTILES}
        ex.write_f16(os.path.join(target_dir, "layers", f"{lid}.f16.gz"), full)
        ex.write_f16(os.path.join(target_dir, "layers", f"{lid}.preview.f16.gz"), prev)
        full_arrays.append(np.nan_to_num(full, nan=0.0))
        prev_arrays.append(np.nan_to_num(prev, nan=0.0))
        layer = {
            "id": lid, "mission": L["mission"], "instrument": L["instrument"], "label": L["name"], "kind": "optical-ir", "center_nm": L["center"],
            "file": f"layers/{lid}.f16.gz", "preview_file": f"layers/{lid}.preview.f16.gz",
            "encoding": {"format": "float16-le-gzip", "scale": top, "percentiles": pct},
            "default_color": "#ffffff", "default_weight": wgt, "obs_ids": [L["obs_id"]], "primary_source": L["link"],
        }
        if L.get("exposure"):
            layer["exposure_s"] = L["exposure"]
        layers.append(layer)
    names = [L["name"] for L in layers_in]
    waves = [L["center"] for L in layers_in]
    stretch = cfg.get("stretch") or fin.get("stretch") or color.STRETCH
    black = cfg.get("black", fin.get("black", 1.0))
    sat = fin.get("saturation", 1.0)
    true_cols = {"Violet": "#0000ff", "Blue": "#0000ff", "Green": "#00ff00", "Orange": "#ff0000", "Red": "#ff0000"}
    true_w = list(weights)
    if "Blue" in names and "Violet" in names:   # Voyager blue and violet frames share the blue channel half and half
        true_w = [round(w / 2, 4) if n in ("Blue", "Violet") else w for w, n in zip(weights, names)]
    presets = [{
        "id": "true-colour", "name": "True colour", "colors": {lid: true_cols[n] for lid, n in zip(ids, names)},
        "weights": {lid: w for lid, w in zip(ids, true_w)}, "stretch": stretch, "softening": color.Q, "black_point": black, "white_point": 100, "saturation": sat,
    }]
    flat = {lid: 1.0 for lid in ids}
    for pname in PRESET_PALETTES:
        cols = color.RANKED[pname](len(ids))
        presets.append({"id": pname, "name": color.PALETTE_NAMES[pname], "colors": {lid: color.hexof(c) for lid, c in zip(ids, cols)},
                        "weights": flat, "stretch": stretch, "softening": color.Q, "black_point": black, "white_point": 100, "saturation": sat})
    for layer in layers:
        layer["default_color"] = presets[0]["colors"][layer["id"]]
    manifest = {
        "version": 1, "build": os.environ.get("GITHUB_RUN_ID") or time.strftime("%Y%m%d%H%M%S", time.gmtime()), "id": cfg["id"], "name": cfg["name"],
        "blurb": cfg.get("blurb", ""),
        "grid": {"width": int(full_shape[1]), "height": int(full_shape[0]), "arcsec_per_pixel": 1.0,
                 "preview_width": int(prev_shape[1]), "preview_height": int(prev_shape[0])},
        "layers": layers, "presets": presets, "black_point_strip": {"values": STRIP_VALUES},
        "credits": {"line": credits["line"], "made_by": "RRR + Claude"}, "sources": credits["sources"],
    }
    with open(os.path.join(target_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    # The check picture shows the first look the page will give: the black point taken off, then the saturation.
    chk = full_arrays
    if black > 0:
        pcts = [np.percentile(a[a > 0], black) if (a > 0).any() else 0.0 for a in full_arrays]
        chk = [np.clip((a - b) / max(1 - b, 1e-6), 0, None) * (a > 0) for a, b in zip(full_arrays, pcts)]
    ex.check(target_dir, ids, waves, prev_arrays, chk, true_w, None, presets[0], stretch)
    if sat != 1.0:
        from PIL import Image
        path = os.path.join(target_dir, "check", f"composite-{presets[0]['id']}.jpg")
        img = np.asarray(Image.open(path)).astype(np.float32)
        luma = img @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        Image.fromarray(np.clip(luma[..., None] + (img - luma[..., None]) * sat, 0, 255).astype(np.uint8)).save(path, quality=88)
    return manifest


def main():
    cfg_path, outdir = sys.argv[1:3]
    cfg = json.load(open(cfg_path))
    src = cfg["source"]
    work = os.environ.get("WORK_DIR", "work")
    os.makedirs(work, exist_ok=True)
    if src["kind"] == "cassini":
        layers_in, meta, _ = cassini_layers(src)
    elif src["kind"] == "junocam":
        layers_in, meta, _ = junocam_layers(src, work)
    elif src["kind"] == "voyager":
        layers_in, meta, _ = voyager_layers(src)
    else:
        sys.exit(f"unknown source kind {src['kind']}")
    print(f"== {cfg['id']}: {[L['name'] for L in layers_in]} {layers_in[0]['array'].shape} {meta}", flush=True)
    manifest = build(cfg, layers_in, outdir, CREDITS[src["kind"]], src["kind"])
    print(f"done: {len(manifest['layers'])} layers for {cfg['id']}", flush=True)


if __name__ == "__main__":
    main()
