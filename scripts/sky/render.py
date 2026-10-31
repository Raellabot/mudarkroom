# Shared by the Sun and WISE pictures: the ghoul, ghost and slime palettes, the black point, the asinh stretch and the legend bar.
import os
from astropy.visualization import make_lupton_rgb
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PINK = (0.980, 0.098, 0.569)   # #fa1991
STOPS = {
    "ghoul": [(0.00, (0.38, 0.12, 0.70)), (0.25, (0.30, 0.95, 0.85)), (0.50, (0.55, 1.00, 0.12)), (0.75, (1.00, 0.40, 0.04)), (1.00, PINK)],
    "ghost": [(0.00, (0.35, 0.10, 0.80)), (0.25, (0.10, 0.85, 0.80)), (0.50, (0.78, 0.70, 1.00)), (0.75, (0.85, 0.05, 0.15)), (1.00, PINK)],
    "slime": [(0.00, (0.30, 0.05, 0.70)), (0.25, (0.20, 0.95, 0.35)), (0.50, (0.85, 0.95, 0.60)), (0.75, (0.75, 0.05, 0.20)), (1.00, PINK)],
}


TWO = {"ghoul": (0.75, 1.0), "ghost": (0.25, 1.0), "slime": (0.5, 1.0)}   # where a two-filter picture samples each palette, so the three do not collapse to violet and pink


def palette_colors(name, n):
    out = []
    for i in range(n):
        t = TWO[name][i] if n == 2 and name in TWO else i / (n - 1) if n > 1 else 0.0
        for (a, ca), (b, cb) in zip(STOPS[name], STOPS[name][1:]):
            if t <= b:
                k = (t - a) / (b - a)
                out.append(np.array(ca) * (1 - k) + np.array(cb) * k)
                break
    return out


# Black point at the given percentile of each filter, each filter divided by its own 99.7th percentile, then weighted by (top / largest top) ** (1 - exp).
def normalize_keep_haze(arrays, exp, black):
    out, tops = [], []
    for a in arrays:
        v = a[np.isfinite(a)]
        x = np.clip(a - np.percentile(v, black), 0, None)
        top = float(np.percentile(x[np.isfinite(x)], 99.7)) or float(np.nanmax(x)) or 1.0
        out.append(np.nan_to_num(x / top, nan=0.0).astype(np.float32))
        tops.append(top)
    big = max(tops) or 1.0
    return [o * ((t / big) ** (1.0 - exp)) for o, t in zip(out, tops)]


def legend(img, names, colors, title):
    h, w = img.shape[:2]
    bar = max(70, h // 22)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", max(14, bar // 4))
    except OSError:
        font = ImageFont.load_default()
    canvas = Image.new("RGB", (w, h + bar), (0, 0, 0))
    canvas.paste(Image.fromarray(img), (0, 0))
    d = ImageDraw.Draw(canvas)
    d.text((12, h + 6), title, fill=(235, 235, 235), font=font)
    sw, x, y = bar // 3, 12, h + bar // 2 + 2
    probe = ImageDraw.Draw(Image.new("RGB", (4, 4)))
    for n, c in zip(names, colors):
        d.rectangle([x, y, x + sw, y + sw], fill=tuple(int(round(255 * v)) for v in np.clip(c, 0, 1)))
        d.text((x + sw + 8, y - 2), n, fill=(220, 220, 220), font=font)
        x += sw + 32 + int(probe.textlength(n, font=font))
    return canvas


# Writes one picture per palette at the default black point and at each extra one (file names end -blackNN, like the Hubble and Webb ones).
def render(outdir, label, names, arrays, title, default_black, extra_blacks=(), palettes=("ghoul", "ghost", "slime"), stretch=0.25, q=8.0, exp=0.5, flip=False, rot180=False):
    made = []
    for black in [None] + list(extra_blacks):
        b = default_black if black is None else black
        stack = normalize_keep_haze(arrays, exp, b)
        for pname in palettes:
            cols = palette_colors(pname, len(stack))
            rgb = np.zeros(stack[0].shape + (3,), dtype=np.float32)
            for x, c in zip(stack, cols):
                for ch in range(3):
                    rgb[..., ch] += x * c[ch]
            rgb *= 3.0 / max(len(stack), 3)
            img = make_lupton_rgb(rgb[..., 0], rgb[..., 1], rgb[..., 2], minimum=0.0, stretch=stretch, Q=q, output_dtype=np.uint8)
            if flip:
                img = np.ascontiguousarray(img[::-1])
            if rot180:
                img = np.ascontiguousarray(img[::-1, ::-1])
            fn = f"{label}-{pname}-n{int(round(exp * 100)):02d}" + ("" if black is None else f"-black{black:g}") + ".jpg"
            legend(img, names, cols, f"{title} — {pname} palette, black point {b:g} — made by this project (Raellabot)").save(os.path.join(outdir, fn), quality=90)
            made.append(fn)
    return made
