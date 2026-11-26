# Voyager ISS color sets for the darkroom: downloads the GEOMED frames (I/F x 10000, calibrated, distortion removed) from the PDS Ring-Moon Systems Node, aligns them on the green frame, and returns linear channels.
import json, re, time, urllib.request
import numpy as np
from scipy import ndimage

OPUS = "https://opus.pds-rings.seti.org/opus/api"
HOLD = "https://pds-rings.seti.org/holdings/volumes/"
UA = "cosmic-darkroom (github.com/Raellabot/cosmic-darkroom)"
EDGE = 12
FILTERS = ("orange", "green", "blue", "violet")


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
    return int(y - limit), int(x - limit), float(sub[y, x] / (np.std(c) + 1e-12))


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
    return (int(best[1]), int(best[2])), best[3], round(best[0], 3), [[c[0], int(c[1]), int(c[2])] for c in cands], bool(whole)


# Aligned linear I/F channels for one set: src has the opus ids of the frames ("orange", "green", "blue", "violet"; blue or violet may be missing) and "green_volume" (like VGISS_5114).
def layers(src):
    have = [f for f in FILTERS if src.get(f)]
    assert "green" in have and "orange" in have and ("blue" in have or "violet" in have), have
    frames, urls = {}, {}
    for f in have:
        u, b = geomed_url(src[f], src["green_volume"])
        frames[f], urls[f] = to_if(parse_vicar(b)[1]), u
    g, gbad = frames["green"]
    chans, info = {"green": g}, {"urls": urls, "align": {}}
    for f in have:
        if f == "green":
            continue
        x, bad = frames[f]
        (dy, dx), how, corr, cands, whole = align(g, x, gbad, bad)
        chans[f] = np.roll(x, (dy, dx), axis=(0, 1))
        info["align"][f] = dict(shift=[dy, dx], how=how, corr=corr)
    m = max(max(abs(v["shift"][0]), abs(v["shift"][1])) for v in info["align"].values()) + 1
    sl = (slice(m, g.shape[0] - m), slice(m, g.shape[1] - m))
    return {f: chans[f][sl] for f in have}, info
