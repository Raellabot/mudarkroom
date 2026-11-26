# Voyager ISS color sets for the darkroom: downloads the GEOMED frames (I/F x 10000, calibrated, distortion removed) from the PDS Ring-Moon Systems Node, aligns them on the green frame, and returns linear channels.
import json, re, time, urllib.request
import numpy as np
from scipy import ndimage

OPUS = "https://opus.pds-rings.seti.org/opus/api"
HOLD = "https://pds-rings.seti.org/holdings/volumes/"
UA = "cosmic-darkroom (github.com/Raellabot/cosmic-darkroom)"
EDGE = 12
FILTERS = ("orange", "green", "blue", "violet")
MAX_GAP = 4  # dropped lines up to this many rows are filled from their neighbors


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


# Fills missing pixels: dropped lines up to MAX_GAP rows from the rows around them, other gaps from their neighbors; then replaces single-pixel spikes (bit errors) with the local median.
def clean(x, bad):
    x = x.copy()
    lit = ndimage.gaussian_filter(x, 2) > 0.15 * np.percentile(x, 99.5)
    bad = bad.copy()
    bad[odd_rows(x, lit)] = True
    rows = np.where(bad.mean(1) > 0.5)[0]
    for r in np.split(rows, np.where(np.diff(rows) > 1)[0] + 1) if len(rows) else []:
        if len(r) <= MAX_GAP and r[0] > 0 and r[-1] < x.shape[0] - 1:
            a, b = x[r[0] - 1], x[r[-1] + 1]
            for i, y in enumerate(r):
                t = (i + 1) / (len(r) + 1)
                x[y] = a * (1 - t) + b * t
            bad[r] = False
    if bad.any():
        w = (~bad).astype(np.float32)
        fill = ndimage.uniform_filter(x * w, 5) / np.maximum(ndimage.uniform_filter(w, 5), 1e-3)
        x[bad] = fill[bad]
    med = ndimage.median_filter(x, 3)
    noise = 1.4826 * np.median(np.abs(x - med)) + 1e-6
    spike = (x - med) > np.maximum(8 * noise, 0.03 * np.percentile(med, 99.5))
    x[spike] = med[spike]
    return x


# Sub-pixel part of a shift: a parabola through the edge-match score around the integer best.
def subpix(ga, gb, dy, dx):
    def frac(m, c, p):
        d = m - 2 * c + p
        return 0.0 if d >= 0 else float(np.clip(0.5 * (m - p) / d, -0.5, 0.5))
    c = score(ga, gb, dy, dx)
    return frac(score(ga, gb, dy - 1, dx), c, score(ga, gb, dy + 1, dx)), frac(score(ga, gb, dy, dx - 1), c, score(ga, gb, dy, dx + 1))


# Takes off the dark sky: a plane fitted to each channel away from the lit body, then a smoothed row-by-row level, so a gradient or a band left by the camera doesn't tint the black; the empty margin left by the distortion fix stays at zero.
# Both fits use the darker sky with outliers dropped, so rings or a second moon don't lift them, and what is taken off is held to a few percent of the body's brightness.
def sky(x, body):
    nodata = ndimage.binary_opening(x <= 0, iterations=3)
    away = ~ndimage.binary_dilation(body, iterations=15) & ~nodata
    if away.sum() < 2000:
        return x
    cap = 0.03 * np.percentile(x, 99.5)
    rcap = 0.03 * max(np.percentile(x, 99.5), np.percentile(x[body], 90) if body.sum() >= 50 else 0)   # the row bands may go further, so a small disk in a dark frame still loses its sky gradient
    ys, xs = np.nonzero(away)
    v = x[ys, xs]
    keep = v <= np.percentile(v, 50)
    A = np.stack([np.ones(v.size), ys, xs], 1)
    for _ in range(4):
        c, *_ = np.linalg.lstsq(A[keep], v[keep], rcond=None)
        res = v - A @ c
        mad = 1.4826 * np.median(np.abs(res[keep])) + 1e-6
        keep = np.abs(res) < 3 * mad
        if keep.sum() < 500:
            break
    yy, xx = np.mgrid[:x.shape[0], :x.shape[1]]
    y = x - np.clip(c[0] + c[1] * yy + c[2] * xx, -cap, cap)
    rows = np.full(x.shape[0], np.nan)
    for r in range(x.shape[0]):
        a = y[r][away[r]]
        if a.size > 0.4 * x.shape[1]:
            rows[r] = np.percentile(a, 25)
    ok = ~np.isnan(rows)
    if ok.sum() > 50:
        level = np.interp(np.arange(x.shape[0]), np.nonzero(ok)[0], ndimage.median_filter(rows[ok], 31, mode="nearest"))
        y = y - np.clip(ndimage.uniform_filter1d(level, 15), -rcap, rcap)[:, None]
    return np.where(nodata, 0, y)


# Lit area of a set, from all its channels together.
def lit_all(chans):
    tot = sum(chans.values())
    return tot, ndimage.gaussian_filter(tot, 2) > 0.15 * np.percentile(tot, 99.5)


# Rows where one channel's share of the light jumps away from the rows around it (a line garbled in one filter), filled from the rows around them.
def color_rows(chans, tol=0.1):
    tot, lit = lit_all(chans)
    n = lit.sum(1)
    for f, x in chans.items():
        share = np.where(n > 20, (x / np.maximum(tot, 1e-6) * lit).sum(1) / np.maximum(n, 1), np.nan)
        ok = ~np.isnan(share)
        if ok.sum() < 10:
            continue
        ref = share.copy()
        ref[ok] = ndimage.median_filter(share[ok], 9, mode="nearest")
        rows = np.where(ok & (np.abs(share / np.maximum(ref, 1e-6) - 1) > tol))[0]
        for r in np.split(rows, np.where(np.diff(rows) > 1)[0] + 1) if len(rows) else []:
            if len(r) <= MAX_GAP and r[0] > 0 and r[-1] < x.shape[0] - 1:
                a, b = x[r[0] - 1].copy(), x[r[-1] + 1].copy()
                for i, y in enumerate(r):
                    t = (i + 1) / (len(r) + 1)
                    x[y] = a * (1 - t) + b * t
    return chans


# Crops off frame edges where the colors go wrong (a strip that one filter lost or garbled along the border): walks in from each side and cuts past the last line whose color share is off.
def trim(chans, tol=0.25, depth=60):
    tot, lit = lit_all(chans)
    h, w = tot.shape
    inner = lit.copy()
    inner[:depth], inner[-depth:], inner[:, :depth], inner[:, -depth:] = False, False, False, False
    if inner.sum() < 500:
        inner = lit
    if inner.sum() < 50:
        return chans, {}
    share = {f: x / np.maximum(tot, 1e-6) for f, x in chans.items()}
    ref = {f: float(np.median(s[inner])) for f, s in share.items()}
    mh, mw = min(depth, h // 4), min(depth, w // 4)  # a line's ends are left out, so a strip along one side doesn't count against the next side
    lines = {"top": lambda i: (i, slice(mw, w - mw)), "bottom": lambda i: (h - 1 - i, slice(mw, w - mw)), "left": lambda i: (slice(mh, h - mh), i), "right": lambda i: (slice(mh, h - mh), w - 1 - i)}
    cut = {}
    for side, at in lines.items():
        k = 0
        for i in range(mh if side in ("top", "bottom") else mw):
            m = lit[at(i)]
            if m.sum() < 15:
                continue
            if any(abs(np.median(s[at(i)][m]) / max(ref[f], 1e-6) - 1) > tol for f, s in share.items()):
                k = i + 1
        cut[side] = k
    if lit[cut["top"]:h - cut["bottom"], cut["left"]:w - cut["right"]].sum() < 0.7 * lit.sum():
        return chans, {"skipped": cut}  # a cut that takes off much of the subject (a small moon at the edge) costs more than the strip
    return {f: x[cut["top"]:h - cut["bottom"], cut["left"]:w - cut["right"]] for f, x in chans.items()}, cut


def layers(src):
    have = [f for f in FILTERS if src.get(f)]
    assert "green" in have and "orange" in have and ("blue" in have or "violet" in have), have
    frames, urls = {}, {}
    for f in have:
        u, b = geomed_url(src[f], src["green_volume"])
        frames[f], urls[f] = to_if(parse_vicar(b)[1]), u
    g, gbad = frames["green"]
    chans, info = {"green": clean(g, gbad)}, {"urls": urls, "align": {}}
    ga = edges(g, gbad)
    for f in have:
        if f == "green":
            continue
        x, bad = frames[f]
        (dy, dx), how, corr, cands, whole = align(g, x, gbad, bad)
        fy, fx = subpix(ga, edges(x, bad), dy, dx)
        chans[f] = ndimage.shift(clean(x, bad), (dy + fy, dx + fx), order=1, mode="constant")
        info["align"][f] = dict(shift=[round(dy + fy, 2), round(dx + fx, 2)], how=how, corr=corr)
    m = int(max(max(abs(v["shift"][0]), abs(v["shift"][1])) for v in info["align"].values())) + 2
    sl = (slice(m, g.shape[0] - m), slice(m, g.shape[1] - m))
    body, _ = lit_mask(chans["green"][sl])
    out = color_rows({f: chans[f][sl] for f in have})
    out = speckle({f: sky(x, body) for f, x in out.items()}, body)
    out, info["trim"] = trim(out)
    return out, info


# Clears colored specks from the dark sky: a pixel bright in one channel but dark in all the others is a bit error, not a star (stars show in every filter).
def speckle(chans, body):
    away = ~ndimage.binary_dilation(body, iterations=10)
    if away.sum() < 2000:
        return chans
    noise = {f: 1.4826 * np.median(np.abs(x[away] - np.median(x[away]))) + 1e-6 for f, x in chans.items()}
    hot = {f: x > 6 * noise[f] for f, x in chans.items()}
    dark = {f: x < 3 * noise[f] for f, x in chans.items()}
    for f, x in chans.items():
        lone = away & hot[f] & np.logical_and.reduce([dark[g] for g in chans if g != f])
        x[ndimage.binary_dilation(lone)] = 0
    return chans
