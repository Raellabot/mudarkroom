# Jupiter movies from Cassini narrow-angle color sets (2000-01 flyby): downloads the calibrated I/F frames, aligns each set, centers the planet, renders on one fixed brightness scale and writes the movies with ffmpeg.
# Usage: jupiter_approach.py SETS.csv OUTDIR [RED_RATIO]  (RED_RATIO: scales the red channel so its disk level over green's matches this, for sets whose red frame is the 752 nm filter)
import csv, json, os, re, subprocess, sys, time, urllib.error, urllib.request
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "voyager"))
from approach import CANVAS, FPS, MAX_LOST, MAX_TINT, SPIN_DIAMETER, clean_sky, font, movie, one_per_turn, render, repair, tinted_rows
from screen import align, lit_mask, strings

OPUS = "https://opus.pds-rings.seti.org/opus/api"
CAL = "https://opus.pds-rings.seti.org/holdings/calibrated/COISS_1xxx/"
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


# Reads a VICAR image of REAL, HALF or BYTE samples.
def parse_vicar(b):
    lbl = int(re.match(rb"LBLSIZE=(\d+)", b).group(1))
    items = {}
    for k, v in re.findall(r"(\w+)=('(?:[^']|'')*'|\([^)]*\)|[^\s]+)", b[:lbl].decode("latin1")):
        items.setdefault(k, v.strip("'"))
    ns, nl, nbb, nlb = int(items["NS"]), int(items["NL"]), int(items.get("NBB", 0)), int(items.get("NLB", 0))
    dt = {"REAL": "<f4", "HALF": "<i2", "BYTE": "u1"}[items["FORMAT"]]
    size = np.dtype(dt).itemsize
    rec = int(items.get("RECSIZE", nbb + ns * size))
    raw = np.frombuffer(b, dtype="u1", count=nl * rec, offset=lbl + nlb * rec).reshape(nl, rec)
    return items, np.ascontiguousarray(raw[:, nbb:nbb + ns * size]).view(dt).reshape(nl, ns)


# Calibrated frame for one opusid: tries the green frame's folder first, then asks OPUS for the frame's files.
def calib(opusid, green_file):
    vol, _, folder = green_file.split("/")[:3]
    num = opusid.split("-")[-1].upper()
    ver = green_file.split("_")[-1].split(".")[0]
    try:
        b = get(f"{CAL}{vol}/data/{folder}/{num}_{ver}_CALIB.IMG")
    except urllib.error.HTTPError:
        urls = [u for u in strings(json.loads(get(f"{OPUS}/files/{opusid}.json"))) if u.endswith("_CALIB.IMG")]
        b = get(urls[0])
    items, a = parse_vicar(b)
    x = a.astype(np.float32)
    bad = ~np.isfinite(x) | (x == 0) | (x < -0.5)
    x = np.where(bad, 0, np.clip(x, 0, None))
    return x[EDGE:-EDGE, EDGE:-EDGE], bad[EDGE:-EDGE, EDGE:-EDGE]


# Downloads and aligns one set; saves the aligned linear I/F cube and returns where the planet sits.
def prepare(args):
    row, out = args
    try:
        frames = {f: calib(row[f], row["green_file"]) for f in CHANS}
        shapes = {frames[f][0].shape for f in CHANS}
        if len(shapes) > 1:
            return dict(label=row["label"], time=row["time"], error=f"sizes differ {shapes}")
        g, gbad = frames["green"]
        chans, bads, shifts = {"green": g}, {"green": gbad}, {}
        for f in ("red", "blue"):
            x, bad = frames[f]
            (dy, dx), how, corr, _, _ = align(g, x, gbad, bad)
            chans[f] = np.roll(x, (dy, dx), axis=(0, 1))
            bads[f] = np.roll(bad, (dy, dx), axis=(0, 1))
            shifts[f] = [dy, dx, how, corr]
        m, area = lit_mask(np.maximum.reduce([chans[f] for f in CHANS]))
        sky = ~ndimage.binary_dilation(m, iterations=25)
        core = ndimage.binary_erosion(m, iterations=3)
        if core.sum() < 50:
            core = m
        lost, ratio = {}, {}
        for f in CHANS:
            good = sky & ~bads[f]
            if good.sum() > 1000:
                chans[f] = chans[f] - float(np.median(chans[f][good]))
            chans[f] = clean_sky(chans[f], sky & ~bads[f])
            chans[f], lost[f] = repair(chans[f], bads[f], core)
            ratio[f] = float(np.median(chans[f][m]) / max(np.median(chans["green"][m]), 1e-6))
        lost["rows"] = tinted_rows({"orange": chans["red"], "green": chans["green"], "blue": chans["blue"]}, core)
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
        return dict(label=row["label"], time=row["time"], target="Jupiter", cy=float(cy), cx=float(cx), diam=diam, whole=whole, size=h,
                    top=float(np.percentile(rgb.max(-1)[m], 99.5)), shifts=shifts, lost=lost, ratio=ratio)
    except Exception as e:
        return dict(label=row["label"], time=row["time"], error=repr(e)[:300])


# Places one frame on the canvas with the planet at the center, optionally scaled so the disk is SPIN_DIAMETER across.
def compose(info, out, top, spin, gain, note):
    rgb = np.load(os.path.join(out, "cubes", info["label"] + ".npy")).astype(np.float32)
    pic = Image.fromarray(render(rgb, top, gain))
    cy, cx = info["cy"], info["cx"]
    if spin:
        s = SPIN_DIAMETER / info["diam"]
        pic = pic.resize((round(pic.width * s), round(pic.height * s)), Image.LANCZOS)
        cy, cx = cy * s, cx * s
    canvas = Image.new("RGB", (CANVAS, CANVAS), "black")
    canvas.paste(pic, (round(CANVAS / 2 - cx), round(CANVAS / 2 - cy)))
    when = datetime.fromisoformat(info["time"][:19])
    ImageDraw.Draw(canvas).text((28, CANVAS - 56), "Cassini  ·  " + when.strftime("%B %-d, %Y  %H:%M UTC") + note, fill=(235, 235, 235), font=font(26))
    return canvas


def write(infos, out, name, top, spin, fps, gain, note):
    d = os.path.join(out, "frames-" + name)
    os.makedirs(d, exist_ok=True)
    for i, info in enumerate(infos):
        compose(info, out, top, spin, gain, note).save(os.path.join(d, f"{i:04d}.png"))
    movie(d, os.path.join(out, name + ".mp4"), fps)
    subprocess.run(["rm", "-rf", d])
    return len(infos)


def main(sets, out, red_ratio=None):
    os.makedirs(os.path.join(out, "cubes"), exist_ok=True)
    rows = list(csv.DictReader(open(sets)))
    with ProcessPoolExecutor(max(8, 2 * (os.cpu_count() or 2))) as ex:
        infos = list(ex.map(prepare, [(r, out) for r in rows]))
    with open(os.path.join(out, "frames.jsonl"), "w") as f:
        for i in infos:
            f.write(json.dumps(i) + "\n")
    ok = [i for i in infos if "error" not in i and i["whole"]]
    med = {f: float(np.median([i["ratio"][f] for i in ok])) for f in ("red", "blue")}
    good = [i for i in ok if max(i["lost"].values()) <= MAX_LOST and all(abs(i["ratio"][f] / med[f] - 1) <= MAX_TINT for f in med)]
    good.sort(key=lambda i: i["time"])
    rgain = float(red_ratio) / med["red"] if red_ratio else 1.0
    gain = np.array([rgain, 1.0, 1.0], np.float32)
    top = float(np.median([i["top"] for i in good]))
    note = "  ·  approximate color" if red_ratio else ""
    counts = dict(sets=len(rows), failed=sum("error" in i for i in infos), cut_off=sum("error" not in i and not i["whole"] for i in infos), dropouts=len(ok) - len(good),
                  used=len(good), top_if=round(top, 4), first=good[0]["time"], last=good[-1]["time"], diam_first=good[0]["diam"], diam_last=good[-1]["diam"],
                  red_ratio=round(med["red"], 4), blue_ratio=round(med["blue"], 4), red_gain=round(rgain, 4))
    counts["approach"] = write(good, out, "approach", top, False, FPS, gain, note)
    counts["spin"] = write(good, out, "spin", top, True, FPS, gain, note)
    counts["one_per_turn"] = write(one_per_turn(good), out, "one-per-turn", top, False, 6, gain, note)
    json.dump(counts, open(os.path.join(out, "summary.json"), "w"), indent=1)
    print(json.dumps(counts, indent=1))
    subprocess.run(["rm", "-rf", os.path.join(out, "cubes")])


if __name__ == "__main__":
    main(*sys.argv[1:4])
