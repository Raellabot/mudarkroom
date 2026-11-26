# Jupiter approach animation from Voyager 1 narrow-angle color sets: aligns each set, centers the planet, renders on one fixed brightness scale, and writes the movies with ffmpeg.
# Usage: approach.py SETS.csv OUTDIR
import csv, json, os, subprocess, sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
sys.path.insert(0, os.path.dirname(__file__))
from screen import align, geomed_url, lit_mask, parse_vicar, to_if

CANVAS = 1080
SPIN_DIAMETER = 860
ROTATION = timedelta(hours=9, minutes=55, seconds=29.7)
FPS = 12
MAX_GAP = 4         # dropped lines up to this many rows are filled from their neighbors
MAX_LOST = 0.003    # a set missing more of the disk than this in any channel is left out
MAX_TINT = 0.2      # a set whose orange or blue level is this far from the run's median is left out (a blank or wrong frame)


def font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


# Downloads and aligns one set; saves the aligned linear I/F cube and returns where the planet sits.
def prepare(args):
    row, out = args
    try:
        have = [f for f in ("orange", "green", "blue") if row[f]]
        gvol = row["green_file"].split("/")[0]
        frames = {}
        for f in have:
            _, b = geomed_url(row[f], gvol)
            frames[f] = to_if(parse_vicar(b)[1])
        g, gbad = frames["green"]
        chans, bads = {"green": g}, {"green": gbad}
        shifts = {}
        for f in ("orange", "blue"):
            x, bad = frames[f]
            (dy, dx), how, corr, _, _ = align(g, x, gbad, bad)
            chans[f] = np.roll(x, (dy, dx), axis=(0, 1))
            bads[f] = np.roll(bad, (dy, dx), axis=(0, 1))
            shifts[f] = [dy, dx, how, corr]
        m, area = lit_mask(np.maximum.reduce([chans["orange"], chans["green"], chans["blue"]]))  # lit in any filter, so a patch missing from one still counts as disk
        lost, ratio = {}, {}
        for f in ("orange", "green", "blue"):
            chans[f], lost[f] = repair(chans[f], bads[f], m)
            ratio[f] = float(np.median(chans[f][m]) / max(np.median(chans["green"][m]), 1e-6))
        lost["rows"] = tinted_rows(chans, m)
        rgb = np.stack([chans["orange"], chans["green"], chans["blue"]], -1).astype(np.float32)
        ys, xs = np.where(m)
        h, w = g.shape
        whole = bool(area > 150 and ys.min() > 3 and xs.min() > 3 and ys.max() < h - 4 and xs.max() < w - 4)
        cy, cx = ndimage.center_of_mass(m)
        diam = float(max(ys.max() - ys.min(), xs.max() - xs.min()) + 1)
        np.save(os.path.join(out, "cubes", row["label"] + ".npy"), rgb.astype(np.float16))
        return dict(label=row["label"], time=row["time"], cy=float(cy), cx=float(cx), diam=diam, whole=whole,
                    top=float(np.percentile(rgb.max(-1)[m], 99.5)), shifts=shifts, lost=lost, ratio=ratio)
    except Exception as e:
        return dict(label=row["label"], time=row["time"], error=repr(e)[:300])


# Rows whose level on the lit body jumps away from the rows around them (a line dropped or garbled in transmission without being flagged).
def odd_rows(x, lit, tol=0.12):
    n = lit.sum(1)
    mean = np.where(n > 20, (x * lit).sum(1) / np.maximum(n, 1), np.nan)
    ok = ~np.isnan(mean)
    if ok.sum() < 10:
        return np.zeros(x.shape[0], bool)
    ref = mean.copy()
    ref[ok] = ndimage.median_filter(mean[ok], 9, mode="nearest")
    out = np.zeros(x.shape[0], bool)
    out[ok] = np.abs(mean[ok] / np.maximum(ref[ok], 1e-6) - 1) > tol
    return out


# Fills dropped lines on the disk from the rows above and below; returns the channel and the fraction of the disk still missing.
def repair(x, bad, disk):
    drop = (bad | (x <= 0)) & disk
    drop[odd_rows(x, disk)] |= disk[odd_rows(x, disk)]
    rows = np.where(drop.sum(1) > 0.5 * np.maximum(disk.sum(1), 1))[0]
    x = x.copy()
    runs = np.split(rows, np.where(np.diff(rows) > 1)[0] + 1) if len(rows) else []
    for r in runs:
        if len(r) <= MAX_GAP and r[0] > 0 and r[-1] < x.shape[0] - 1:
            a, b = x[r[0] - 1], x[r[-1] + 1]
            for i, y in enumerate(r):
                t = (i + 1) / (len(r) + 1)
                x[y] = a * (1 - t) + b * t
            drop[r] = False
    return x, float(drop.sum() / max(disk.sum(), 1))


# Fraction of the disk in rows whose color is off from the rest of the disk (a block lost in one filter that came back as wrong numbers rather than gaps).
def tinted_rows(chans, disk, tol=0.25):
    n = disk.sum(1)
    use = n > 20
    if use.sum() < 10:
        return 0.0
    g = (chans["green"] * disk).sum(1)[use]
    off = np.zeros(use.sum(), bool)
    for f in ("orange", "blue"):
        r = (chans[f] * disk).sum(1)[use] / np.maximum(g, 1e-6)
        off |= np.abs(r / np.median(r) - 1) > tol
    return float(n[use][off].sum() / max(disk.sum(), 1))


def render(rgb, top):
    x = np.clip(rgb / top, 0, 1)
    return (np.arcsinh(x * 3) / np.arcsinh(3) * 255).astype(np.uint8)


# Places one frame on the canvas with the planet at the center, optionally scaled so the disk is SPIN_DIAMETER across.
def compose(info, out, top, spin):
    rgb = np.load(os.path.join(out, "cubes", info["label"] + ".npy")).astype(np.float32)
    pic = Image.fromarray(render(rgb, top))
    cy, cx = info["cy"], info["cx"]
    if spin:
        s = SPIN_DIAMETER / info["diam"]
        pic = pic.resize((round(pic.width * s), round(pic.height * s)), Image.LANCZOS)
        cy, cx = cy * s, cx * s
    canvas = Image.new("RGB", (CANVAS, CANVAS), "black")
    canvas.paste(pic, (round(CANVAS / 2 - cx), round(CANVAS / 2 - cy)))
    d = ImageDraw.Draw(canvas)
    when = datetime.fromisoformat(info["time"][:19])
    d.text((28, CANVAS - 56), "Voyager 1  ·  " + when.strftime("%B %-d, 1979  %H:%M UTC"), fill=(235, 235, 235), font=font(26))
    return canvas


def movie(frames, path, fps):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(frames, "%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17", "-movflags", "+faststart", path], check=True)


def write(infos, out, name, top, spin, fps):
    d = os.path.join(out, "frames-" + name)
    os.makedirs(d, exist_ok=True)
    for i, info in enumerate(infos):
        compose(info, out, top, spin).save(os.path.join(d, f"{i:04d}.png"))
    movie(d, os.path.join(out, name + ".mp4"), fps)
    return len(infos)


# One set per Jupiter turn: the set nearest each whole rotation after the first, so the same face comes round each frame.
def one_per_turn(infos):
    t0 = datetime.fromisoformat(infos[0]["time"][:19])
    times = [datetime.fromisoformat(i["time"][:19]) for i in infos]
    picked, k = [], 0
    while t0 + k * ROTATION <= times[-1]:
        want = t0 + k * ROTATION
        j = min(range(len(times)), key=lambda n: abs(times[n] - want))
        if abs(times[j] - want) < timedelta(minutes=70) and (not picked or picked[-1] is not infos[j]):
            picked.append(infos[j])
        k += 1
    return picked


def main(sets, out):
    os.makedirs(os.path.join(out, "cubes"), exist_ok=True)
    rows = list(csv.DictReader(open(sets)))
    with ProcessPoolExecutor(max(2, os.cpu_count() or 2)) as ex:
        infos = list(ex.map(prepare, [(r, out) for r in rows]))
    with open(os.path.join(out, "frames.jsonl"), "w") as f:
        for i in infos:
            f.write(json.dumps(i) + "\n")
    ok = [i for i in infos if "error" not in i and i["whole"]]
    med = {f: float(np.median([i["ratio"][f] for i in ok])) for f in ("orange", "blue")}
    good = [i for i in ok if max(i["lost"].values()) <= MAX_LOST and all(abs(i["ratio"][f] / med[f] - 1) <= MAX_TINT for f in med)]
    good.sort(key=lambda i: i["time"])
    top = float(np.median([i["top"] for i in good]))
    counts = dict(sets=len(rows), failed=sum("error" in i for i in infos), cut_off=sum("error" not in i and not i["whole"] for i in infos), dropouts=len(ok) - len(good),
                  used=len(good), top_if=round(top, 4), first=good[0]["time"], last=good[-1]["time"],
                  diam_first=good[0]["diam"], diam_last=good[-1]["diam"])
    counts["approach"] = write(good, out, "approach", top, False, FPS)
    counts["spin"] = write(good, out, "spin", top, True, FPS)
    counts["one_per_turn"] = write(one_per_turn(good), out, "one-per-turn", top, False, 6)
    json.dump(counts, open(os.path.join(out, "summary.json"), "w"), indent=1)
    print(json.dumps(counts, indent=1))
    for name in ("approach", "spin", "one-per-turn"):
        subprocess.run(["rm", "-rf", os.path.join(out, "frames-" + name)])
    subprocess.run(["rm", "-rf", os.path.join(out, "cubes")])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
