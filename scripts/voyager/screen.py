# Screens Voyager ISS color sets: downloads the GEOMED frames (I/F x 10000), aligns them on the green frame, and saves a picture, a black-point strip and stats per set.
# usage: screen.py CANDIDATES.csv OUTDIR CHUNK NCHUNKS
import csv, json, os, re, sys, time, urllib.request
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

OPUS = "https://opus.pds-rings.seti.org/opus/api"
HOLD = "https://pds-rings.seti.org/holdings/volumes/"
UA = "mudarkroom (github.com/Raellabot/mudarkroom)"
EDGE = 12
SIDE = 700
STRIP = [0.0, 0.02, 0.05, 0.10]


def get(url, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=180) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404 or i == tries - 1:
                raise
            time.sleep(5 * (i + 1))
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))


def parse_vicar(b):
    lbl = int(re.match(rb"LBLSIZE=(\d+)", b).group(1))
    text = b[:lbl].decode("latin1")
    items = {}
    for k, v in re.findall(r"(\w+)=('(?:[^']|'')*'|\([^)]*\)|[^\s]+)", text):
        items.setdefault(k, v.strip("'"))
    ns, nl, nbb, nlb = int(items["NS"]), int(items["NL"]), int(items.get("NBB", 0)), int(items.get("NLB", 0))
    assert items["FORMAT"] == "HALF", items["FORMAT"]
    rec = int(items.get("RECSIZE", nbb + ns * 2))
    raw = np.frombuffer(b, dtype="u1", count=nl * rec, offset=lbl + nlb * rec).reshape(nl, rec)
    return items, np.ascontiguousarray(raw[:, nbb:nbb + ns * 2]).view("<i2").reshape(nl, ns)


# GEOMED url for one frame: tries the green frame's volume first, then asks OPUS for the frame's files.
def geomed_url(opusid, green_vol):
    num = "C" + opusid.split("-c")[-1]
    guess = f"{HOLD}VGISS_{green_vol[6]}xxx/{green_vol}/DATA/{num[:6]}XX/{num}_GEOMED.IMG"
    try:
        return guess, get(guess)
    except urllib.error.HTTPError:
        urls = [u for u in strings(json.loads(get(f"{OPUS}/files/{opusid}.json"))) if u.endswith("_GEOMED.IMG")]
        return urls[0], get(urls[0])


def strings(o):
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for v in o.values():
            yield from strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from strings(v)


def to_if(a):
    bad = (a <= -32000) | (a >= 32767)
    x = np.clip(a.astype(np.float32) / 10000.0, 0, None)
    x[bad] = 0
    return x[EDGE:-EDGE, EDGE:-EDGE], bad[EDGE:-EDGE, EDGE:-EDGE]


def highpass(x):
    return x - ndimage.gaussian_filter(x, 12)


def phase_shift(ref, mov, limit):
    h, w = ref.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    f = np.fft.fft2(highpass(ref) * win) * np.conj(np.fft.fft2(highpass(mov) * win))
    c = np.real(np.fft.ifft2(f / (np.abs(f) + 1e-9)))
    c = np.fft.fftshift(c)
    cy, cx = h // 2, w // 2
    sub = c[cy - limit:cy + limit + 1, cx - limit:cx + limit + 1]
    y, x = np.unravel_index(np.argmax(sub), sub.shape)
    return y - limit, x - limit, float(sub[y, x] / (np.std(c) + 1e-12))


def lit_mask(x):
    s = ndimage.gaussian_filter(x, 2)
    top = np.percentile(s, 99.5)
    m = s > 0.2 * top
    lab, n = ndimage.label(m)
    if n == 0:
        return m, 0
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    keep = 1 + int(np.argmax(sizes))
    return lab == keep, int(sizes.max())


def edges(x, bad):
    s = ndimage.gaussian_filter(x, 1.5)
    g = np.hypot(ndimage.sobel(s, 0), ndimage.sobel(s, 1))
    g[ndimage.binary_dilation(bad, iterations=4)] = 0
    return g / (np.percentile(g, 99.5) + 1e-9)


def score(ga, gb, dy, dx):
    m = np.roll(gb, (dy, dx), axis=(0, 1))
    k = max(abs(dy), abs(dx)) + 2
    a, b = ga[k:-k, k:-k], m[k:-k, k:-k]
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))


def refine(ga, gb, dy, dx, r=4):
    best = (score(ga, gb, dy, dx), dy, dx)
    for y in range(dy - r, dy + r + 1):
        for x in range(dx - r, dx + r + 1):
            v = score(ga, gb, y, x)
            if v > best[0]:
                best = (v, y, x)
    return best


# Shift that lines `mov` up with `ref`: tries the lit body's centroid (whole body in frame), phase correlation of the edge maps and of the high-passed frames, refines each, and keeps the one whose edges match best.
def align(ref, mov, rbad, mbad):
    ga, gb = edges(ref, rbad), edges(mov, mbad)
    cands = []
    mr, ar = lit_mask(ref)
    mm, am = lit_mask(mov)
    h, w = ref.shape
    whole = ar > 150 and am > 150
    for m in (mr, mm):
        ys, xs = np.where(m)
        if ys.size == 0 or ys.min() < 3 or xs.min() < 3 or ys.max() > h - 4 or xs.max() > w - 4:
            whole = False
    if whole:
        cy0, cx0 = ndimage.center_of_mass(mr)
        cy1, cx1 = ndimage.center_of_mass(mm)
        cands.append(("centroid", int(round(cy0 - cy1)), int(round(cx0 - cx1))))
    for name, a, b in (("edges", ga, gb), ("phase", ref, mov)):
        dy, dx, _ = phase_shift(a, b, 150)
        cands.append((name, dy, dx))
    best = None
    for name, dy, dx in cands:
        v, y, x = refine(ga, gb, dy, dx)
        if best is None or v > best[0]:
            best = (v, y, x, name)
    return (best[1], best[2]), best[3], round(best[0], 3), [list(c) for c in cands], whole


def render(rgb, black, top):
    x = np.clip((rgb - black * top) / max(top * (1 - black), 1e-9), 0, 1)
    return (np.arcsinh(x * 3) / np.arcsinh(3) * 255).astype(np.uint8)


def screen(row, out):
    have = [f for f in ("orange", "green", "blue", "violet") if row[f]]
    frames, urls = {}, {}
    gvol = row["green_file"].split("/")[0]
    for f in have:
        u, b = geomed_url(row[f], gvol)
        items, a = parse_vicar(b)
        frames[f], urls[f] = to_if(a), u
    g, gbad = frames["green"]
    aligned, info = {"green": g}, {}
    for f in have:
        if f == "green":
            continue
        x, bad = frames[f]
        (dy, dx), how, corr, cands, whole = align(g, x, gbad, bad)
        aligned[f] = np.roll(x, (dy, dx), axis=(0, 1))
        info[f] = dict(shift=[dy, dx], how=how, corr=corr, cands=cands, whole=whole)
    m = max(max(abs(v["shift"][0]), abs(v["shift"][1])) for v in info.values()) + 1
    sl = (slice(m, g.shape[0] - m), slice(m, g.shape[1] - m))
    blue = np.mean([aligned[k][sl] for k in ("blue", "violet") if k in aligned], axis=0)
    rgb = np.stack([aligned["orange"][sl], aligned["green"][sl], blue], -1)
    top = float(np.percentile(rgb.max(-1), 99.7)) or 1.0
    lm, area = lit_mask(rgb[..., 1])
    sat = float(np.mean(gbad))
    fill = float(lm.mean())
    lit = rgb[lm] if lm.any() else rgb.reshape(-1, 3)
    color = (lit.mean(0) / max(lit.mean(), 1e-9)).round(3).tolist()
    pic = Image.fromarray(render(rgb, 0.0, top))
    pic.thumbnail((SIDE, SIDE))
    pic.save(os.path.join(out, "pictures", row["label"] + ".jpg"), quality=90)
    small = [Image.fromarray(render(rgb, b, top)).resize((300, int(300 * rgb.shape[0] / rgb.shape[1]))) for b in STRIP]
    strip = Image.new("RGB", (300 * len(small), small[0].height + 18), "black")
    d = ImageDraw.Draw(strip)
    for i, (s, b) in enumerate(zip(small, STRIP)):
        strip.paste(s, (300 * i, 18))
        d.text((300 * i + 4, 3), f"black {int(b * 100)}%", fill="white")
    strip.save(os.path.join(out, "strips", row["label"] + ".jpg"), quality=85)
    return dict(label=row["label"], target=row["target"], time=row["time"], filters=row["filters"], fill=round(fill, 4), body_px=area,
                top_if=round(top, 4), color=color, sat=round(sat, 4), align=info, size=list(rgb.shape[:2]), urls=urls)


def main(cands, out, chunk, n):
    for d in ("pictures", "strips"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    rows = [r for i, r in enumerate(csv.DictReader(open(cands))) if i % n == chunk]
    with open(os.path.join(out, f"stats-{chunk}.jsonl"), "w") as f:
        for r in rows:
            try:
                s = screen(r, out)
                print("ok", r["label"], s["fill"], json.dumps(s["align"]), flush=True)
            except Exception as e:
                s = dict(label=r["label"], error=repr(e)[:300])
                print("fail", r["label"], s["error"], flush=True)
            f.write(json.dumps(s) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
