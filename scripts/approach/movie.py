# Approach movies from any mission's frames: reads a set list (one row per color set or single gray frame), downloads and reads each frame (VICAR, PDS3 or FITS), aligns the channels, centers the body, renders on one fixed brightness scale and writes three movies with ffmpeg.
# Usage: movie.py SETS.csv OUTDIR ROTATION_HOURS [white]  (white: balance color on the brightest tenth of the disk, for frames without calibrated colors)
import csv, gzip, json, os, re, subprocess, sys, time, urllib.error, urllib.parse, urllib.request
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "voyager"))
from approach import CANVAS, FPS, MAX_LOST, MAX_TINT, SPIN_DIAMETER, clean_sky, font, movie, render, repair, tinted_rows
from screen import align

UA = "mudarkroom (github.com/Raellabot/mudarkroom)"
EDGE = 4
CHANS = ("red", "green", "blue")


def get(url, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=300) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404 or i == tries - 1:
                raise
            time.sleep(5 * (i + 1))
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))


# Fetches a frame; on a 404 looks in the folder for a file with the same stem (the part before the first underscore) and the same ending.
def fetch(url):
    try:
        return url, get(url)
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        folder, name = url.rsplit("/", 1)
        stem, ext = name.split("_")[0].upper(), name.rsplit(".", 1)[-1].upper()
        links = re.findall(r'href="([^"?#]+)"', get(folder + "/").decode("latin1"))
        hits = sorted({l.rsplit("/", 1)[-1] for l in links if l.rsplit("/", 1)[-1].upper().startswith(stem) and l.upper().rsplit(".", 1)[-1] in ({ext, ext + "S"} if ext == "FIT" else {ext})})
        if not hits:
            raise
        u = folder + "/" + hits[-1]
        return u, get(u)


def vicar(b):
    lbl = int(re.match(rb"LBLSIZE=(\d+)", b).group(1))
    items = {}
    for k, v in re.findall(r"(\w+)=('(?:[^']|'')*'|\([^)]*\)|[^\s]+)", b[:lbl].decode("latin1")):
        items.setdefault(k, v.strip("'"))
    ns, nl, nbb, nlb = int(items["NS"]), int(items["NL"]), int(items.get("NBB", 0)), int(items.get("NLB", 0))
    dt = {"REAL": "<f4", "HALF": "<i2", "BYTE": "u1", "FULL": "<i4"}[items["FORMAT"]]
    size = np.dtype(dt).itemsize
    rec = int(items.get("RECSIZE", nbb + ns * size))
    raw = np.frombuffer(b, dtype="u1", count=nl * rec, offset=lbl + nlb * rec).reshape(nl, rec)
    return np.ascontiguousarray(raw[:, nbb:nbb + ns * size]).view(dt).reshape(nl, ns)


# PDS3 image with an attached label: finds the IMAGE object and its sample type.
def pds3(b):
    text = b[:min(len(b), 200000)].decode("latin1")
    end = re.search(r"\nEND[ \t]*\r?\n", text)
    text = text[:end.end()] if end else text
    rec = int(re.search(r"RECORD_BYTES\s*=\s*(\d+)", text).group(1))
    m = re.search(r"\^IMAGE\s*=\s*(\d+)(\s*<BYTES>)?", text)
    off = int(m.group(1)) - 1 if m.group(2) else (int(m.group(1)) - 1) * rec
    obj = re.search(r"OBJECT\s*=\s*IMAGE\b(.*?)END_OBJECT\s*=\s*IMAGE\b", text, re.S).group(1)
    val = lambda k, d=None: (re.search(rf"\b{k}\s*=\s*\"?([^\s\"]+)", obj) or [None, d])[1]
    nl, ns, bits = int(val("LINES")), int(val("LINE_SAMPLES")), int(val("SAMPLE_BITS"))
    st = val("SAMPLE_TYPE", "UNSIGNED_INTEGER").upper()
    pre = int(val("LINE_PREFIX_BYTES", 0)) + int(val("LINE_SUFFIX_BYTES", 0))
    msb = st.startswith(("MSB", "IEEE", "SUN", "MAC")) or st in ("INTEGER", "UNSIGNED_INTEGER", "REAL")
    kind = "f" if "REAL" in st else ("u" if "UNSIGNED" in st else "i")
    dt = np.dtype((">" if msb else "<") + kind + str(bits // 8))
    row = ns * dt.itemsize + pre
    raw = np.frombuffer(b, dtype="u1", count=nl * row, offset=off).reshape(nl, row)
    x = np.ascontiguousarray(raw[:, int(val("LINE_PREFIX_BYTES", 0)):int(val("LINE_PREFIX_BYTES", 0)) + ns * dt.itemsize]).view(dt).reshape(nl, ns)
    return x * float(val("SCALING_FACTOR", 1)) + float(val("OFFSET", 0))


# First image in a FITS file (primary or first image extension).
def fits(b):
    pos = 0
    while pos < len(b):
        hdr, cards = {}, 0
        while True:
            card = b[pos + cards * 80:pos + cards * 80 + 80].decode("latin1")
            cards += 1
            k, v = card[:8].strip(), card[10:].split("/")[0].strip().strip("'").strip()
            if k == "END":
                break
            if card[8:10] == "= ":
                hdr[k] = v
        pos += -(-cards * 80 // 2880) * 2880
        bitpix, naxis = int(hdr["BITPIX"]), int(hdr.get("NAXIS", 0))
        dims = [int(hdr[f"NAXIS{i}"]) for i in range(1, naxis + 1)]
        n = int(np.prod(dims)) if dims else 0
        if naxis >= 2:
            dt = np.dtype({8: "u1", 16: ">i2", 32: ">i4", -32: ">f4", -64: ">f8"}[bitpix])
            x = np.frombuffer(b, dtype=dt, count=n, offset=pos).reshape(dims[::-1])
            while x.ndim > 2:
                x = x[0]
            return x.astype(np.float64) * float(hdr.get("BSCALE", 1)) + float(hdr.get("BZERO", 0))
        pos += -(-n * abs(bitpix) // 8 // 2880) * 2880


def read(url, kind, scale):
    u, b = fetch(url)
    if u.endswith(".gz"):
        b = gzip.decompress(b)
    a = {"vicar": vicar, "pds3": pds3, "fits": fits}[kind](b).astype(np.float32)
    bad = ~np.isfinite(a) | (a <= -1e30) | (a >= 1e30)
    if kind == "vicar" and a.max() <= 255:
        bad |= a == 0  # raw byte frames mark lost data with zero
    x = np.where(bad, 0, a) * float(scale)
    return x[EDGE:-EDGE, EDGE:-EDGE], bad[EDGE:-EDGE, EDGE:-EDGE], u


# The body: everything brighter than a small fraction of its brightest part, holes filled, largest piece kept (low threshold so dark oceans and night-side edges still count).
def body_mask(x, frac):
    s = ndimage.gaussian_filter(x, 2)
    top = np.percentile(s, 99.5)
    m = ndimage.binary_fill_holes(s > frac * top)
    lab, n = ndimage.label(m)
    if n == 0:
        return m, 0
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    keep = 1 + int(np.argmax(sizes))
    return lab == keep, int(sizes.max())


def prepare(args):
    row, out, frac = args
    try:
        gray = row["red"] == row["green"] == row["blue"]
        frames, used = {}, {}
        for f in (("green",) if gray else CHANS):
            x, bad, u = read(row[f], row["kind"], row.get(f + "_scale") or 1)
            frames[f] = (x, bad)
            used[f] = u
        if gray:
            frames["red"] = frames["blue"] = frames["green"]
        if len({frames[f][0].shape for f in CHANS}) > 1:
            return dict(label=row["label"], time=row["time"], error="sizes differ")
        g, gbad = frames["green"]
        chans, bads, shifts = {"green": g}, {"green": gbad}, {}
        for f in ("red", "blue"):
            x, bad = frames[f]
            if gray:
                chans[f], bads[f] = x, bad
                continue
            (dy, dx), how, corr, _, _ = align(g, x, gbad, bad)
            chans[f] = np.roll(x, (dy, dx), axis=(0, 1))
            bads[f] = np.roll(bad, (dy, dx), axis=(0, 1))
            shifts[f] = [dy, dx, how, corr]
        sky0 = ~ndimage.binary_dilation(body_mask(np.maximum.reduce([chans[f] for f in CHANS]), frac)[0], iterations=25)
        for f in CHANS:
            good = sky0 & ~bads[f]
            if good.sum() > 1000:
                chans[f] = chans[f] - float(np.median(chans[f][good]))  # sky to zero (also takes out a raw frame's dark level)
        m, area = body_mask(np.maximum.reduce([chans[f] for f in CHANS]), frac)
        sky = ~ndimage.binary_dilation(m, iterations=25)
        core = ndimage.binary_erosion(m, iterations=3)
        if core.sum() < 50:
            core = m
        lost, ratio, white = {}, {}, {}
        for f in CHANS:
            chans[f] = clean_sky(chans[f], sky & ~bads[f])
            chans[f], lost[f] = repair(chans[f], bads[f], core)
            ratio[f] = float(np.median(chans[f][m]) / max(np.median(chans["green"][m]), 1e-9))
        bright = chans["green"] >= np.percentile(chans["green"][core], 90)
        for f in CHANS:
            white[f] = float(np.median(chans["green"][core & bright]) / max(np.median(chans[f][core & bright]), 1e-9))
        lost["rows"] = 0.0 if gray else tinted_rows({"orange": chans["red"], "green": chans["green"], "blue": chans["blue"]}, core)
        for f in CHANS:
            chans[f] = np.where(sky & bads[f], 0, chans[f])
        lo = np.maximum(np.minimum.reduce([chans[f] for f in CHANS]), 0)
        for f in CHANS:
            chans[f] = np.where(sky, lo, chans[f])
        rgb = np.stack([chans[f] for f in CHANS], -1).astype(np.float32)
        ys, xs = np.where(m)
        h, w = g.shape
        whole = bool(area > 150 and ys.min() > 3 and xs.min() > 3 and ys.max() < h - 4 and xs.max() < w - 4)
        cy, cx = ndimage.center_of_mass(m)
        diam = float(max(ys.max() - ys.min(), xs.max() - xs.min()) + 1)
        np.save(os.path.join(out, "cubes", row["label"] + ".npy"), rgb.astype(np.float16))
        return dict(label=row["label"], time=row["time"], caption=row["caption"], gray=gray, cy=float(cy), cx=float(cx), diam=diam, whole=whole, size=[h, w],
                    top=float(np.percentile(rgb.max(-1)[m], 99.5)), shifts=shifts, lost=lost, ratio=ratio, white=white, files=used)
    except Exception as e:
        return dict(label=row["label"], time=row["time"], error=repr(e)[:300])


def compose(info, out, top, spin, gain, note):
    rgb = np.load(os.path.join(out, "cubes", info["label"] + ".npy")).astype(np.float32)
    pic = Image.fromarray(render(rgb, top, np.ones(3, np.float32) if info["gray"] else gain))
    cy, cx = info["cy"], info["cx"]
    s = SPIN_DIAMETER / info["diam"] if spin else min(1.0, CANVAS / max(pic.width, pic.height))  # a frame bigger than the canvas is shrunk to fit
    if s != 1.0:
        pic = pic.resize((max(1, round(pic.width * s)), max(1, round(pic.height * s))), Image.LANCZOS)
        cy, cx = cy * s, cx * s
    canvas = Image.new("RGB", (CANVAS, CANVAS), "black")
    canvas.paste(pic, (round(CANVAS / 2 - cx), round(CANVAS / 2 - cy)))
    when = datetime.fromisoformat(info["time"][:19])
    ImageDraw.Draw(canvas).text((28, CANVAS - 56), info["caption"] + "  ·  " + when.strftime("%B %-d, %Y  %H:%M UTC") + note, fill=(235, 235, 235), font=font(26))
    return canvas


def write(infos, out, name, top, spin, fps, gain, note):
    d = os.path.join(out, "frames-" + name)
    os.makedirs(d, exist_ok=True)
    for i, info in enumerate(infos):
        compose(info, out, top, spin, gain, note).save(os.path.join(d, f"{i:04d}.png"))
    movie(d, os.path.join(out, name + ".mp4"), fps)
    subprocess.run(["rm", "-rf", d])
    return len(infos)


# One set per turn of the body: the set nearest each whole rotation after the first, within a tenth of a turn.
def one_per_turn(infos, rotation):
    times = [datetime.fromisoformat(i["time"][:19]) for i in infos]
    picked, k = [], 0
    while times[0] + k * rotation <= times[-1]:
        want = times[0] + k * rotation
        j = min(range(len(times)), key=lambda n: abs(times[n] - want))
        if abs(times[j] - want) < rotation / 10 and (not picked or picked[-1] is not infos[j]):
            picked.append(infos[j])
        k += 1
    return picked


def main(sets, out, rotation_hours, mode="-", frac="0.05", min_diam="0"):
    os.makedirs(os.path.join(out, "cubes"), exist_ok=True)
    rows = list(csv.DictReader(open(sets)))
    print("sets", len(rows), flush=True)
    with ProcessPoolExecutor(max(8, 2 * (os.cpu_count() or 2))) as ex:
        infos = list(ex.map(prepare, [(r, out, float(frac)) for r in rows], chunksize=4))
    with open(os.path.join(out, "frames.jsonl"), "w") as f:
        for i in infos:
            f.write(json.dumps(i) + "\n")
    errs = [i["error"] for i in infos if "error" in i]
    print("failed", len(errs), errs[:5], flush=True)
    ok = [i for i in infos if "error" not in i and i["whole"] and i["diam"] >= float(min_diam)]
    color = [i for i in ok if not i["gray"]]
    gain = np.ones(3, np.float32)
    if color:
        if mode == "white":
            gain = np.array([float(np.median([i["white"][f] for i in color])) for f in CHANS], np.float32)
        for i in color:
            i["ratio_w"] = {f: i["ratio"][f] * gain[n] for n, f in enumerate(CHANS)}
        med = {f: float(np.median([i["ratio_w"][f] for i in color])) for f in ("red", "blue")}
        bad = {id(i) for i in color if any(abs(i["ratio_w"][f] / med[f] - 1) > MAX_TINT for f in med)}
    else:
        bad, med = set(), {}
    good = [i for i in ok if max(i["lost"].values()) <= MAX_LOST and id(i) not in bad]
    good.sort(key=lambda i: i["time"])
    top = float(np.median([i["top"] for i in good]))
    note = ""
    counts = dict(sets=len(rows), failed=len(errs), cut_off=sum("error" not in i and not i["whole"] for i in infos), small=sum("error" not in i and i["whole"] and i["diam"] < float(min_diam) for i in infos),
                  dropouts=len(ok) - len(good), used=len(good), top=round(top, 6), first=good[0]["time"], last=good[-1]["time"], diam_first=good[0]["diam"], diam_last=good[-1]["diam"],
                  gain=[round(float(v), 4) for v in gain], ratios=med, mode=mode)
    print(json.dumps(counts), flush=True)
    rot = timedelta(hours=float(rotation_hours))
    counts["approach"] = write(good, out, "approach", top, False, FPS, gain, note)
    counts["spin"] = write(good, out, "spin", top, True, FPS, gain, note)
    counts["one_per_turn"] = write(one_per_turn(good, rot), out, "one-per-turn", top, False, 6, gain, note)
    json.dump(counts, open(os.path.join(out, "summary.json"), "w"), indent=1)
    print(json.dumps(counts, indent=1))
    subprocess.run(["rm", "-rf", os.path.join(out, "cubes")])


if __name__ == "__main__":
    main(*sys.argv[1:])
