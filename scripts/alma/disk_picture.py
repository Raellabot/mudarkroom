# Pictures of protoplanetary disks from the DSHARP continuum images in the ALMA archive, in one color (pink); usage: disk_picture.py OUTDIR [NAME ...]  (no names = all)
import io, json, os, re, sys, time, urllib.request
import numpy as np
from astropy.io import fits
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter, label

ROOT = "https://almascience.eso.org/almadata/lp/DSHARP/"
UA = os.environ.get("CONTACT_INFO") or "CONTACT-INFO"
PINK = np.array([0xFA, 0x19, 0x91]) / 255.0   # her favorite pink, #fa1991
WORK = os.environ.get("WORK_DIR", "work")
BLACK_SIGMA = float(os.environ.get("BLACK_SIGMA", "1.0"))   # noise levels taken as black
SOFT_SIGMA = float(os.environ.get("SOFT_SIGMA", "5.0"))   # noise levels where the stretch bends
WHITEN = float(os.environ.get("WHITEN", "0.35"))   # how far the brightest pixels go toward white


# The DSHARP file names with the usual spacing for titles.
PRETTY = {"AS205": "AS 205", "AS209": "AS 209", "DoAr25": "DoAr 25", "DoAr33": "DoAr 33", "Elias20": "Elias 20", "Elias24": "Elias 24", "Elias27": "Elias 2-27", "GWLup": "GW Lup",
          "HD142666": "HD 142666", "HD143006": "HD 143006", "HD163296": "HD 163296", "HTLup": "HT Lup", "IMLup": "IM Lup", "MYLup": "MY Lup", "RULup": "RU Lup", "SR4": "SR 4",
          "Sz114": "Sz 114", "Sz129": "Sz 129", "WSB52": "WSB 52", "WaOph6": "Wa Oph 6"}
# Systems with companions far from the main disk: the crop follows the main disk only.
MAIN_ONLY = {"HTLup"}


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


# Reads a URL with up to 4 tries; returns the bytes.
def get_bytes(url, timeout=300):
    last = None
    for attempt in range(1, 5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(5 * attempt)
    raise RuntimeError(f"{url}: {str(last)[:160]}")


# The disk names the DSHARP folder has a continuum image for.
def all_names():
    html = get_bytes(ROOT).decode("utf-8", "replace")
    return sorted({m for m in re.findall(r'href="images/([^"?/]+)_continuum\.fits"', html)})


# The image as a 2D array in Jy/beam with north up and east left, plus the header facts the picture needs.
def load(name):
    path = os.path.join(WORK, f"{name}_continuum.fits")
    if not os.path.exists(path):
        os.makedirs(WORK, exist_ok=True)
        open(path, "wb").write(get_bytes(f"{ROOT}images/{name}_continuum.fits"))
    with fits.open(path) as h:
        head, data = h[0].header, np.squeeze(h[0].data).astype(np.float32)
    if data.ndim != 2:
        raise RuntimeError(f"{name}: the image has shape {data.shape}, not 2D")
    pix = abs(float(head["CDELT2"])) * 3600.0   # arcsec per pixel
    if float(head["CDELT1"]) > 0:
        data = data[:, ::-1]   # east to the left
    data = data[::-1]   # first row at the bottom in FITS; north up on screen
    info = {"pixel_arcsec": pix, "bmaj_arcsec": float(head.get("BMAJ", 0)) * 3600.0, "bmin_arcsec": float(head.get("BMIN", 0)) * 3600.0, "bpa_deg": float(head.get("BPA", 0)),
            "object": str(head.get("OBJECT", name)), "frequency_hz": float(head.get("CRVAL3", 0) or 0), "bunit": str(head.get("BUNIT", ""))}
    return np.nan_to_num(data), info


# Noise level from the outer ring of the image, where there is no disk; robust against stray bright pixels.
def noise_of(a):
    h, w = a.shape
    m = np.ones(a.shape, bool)
    m[int(h * 0.15):int(h * 0.85), int(w * 0.15):int(w * 0.85)] = False
    return 1.4826 * float(np.median(np.abs(a[m] - np.median(a[m]))))


# A square box around everything clearly above the noise, with a margin.
def crop_box(a, sigma, pix, beam_px, main_only=False):
    s = gaussian_filter(a, max(1.0, beam_px / 2.0))
    ss = noise_of(s)
    mask = s > 5.0 * ss
    if main_only and mask.any():   # join the rings of the main disk, then keep only the piece that holds the brightest pixel
        grown = gaussian_filter(mask.astype(np.float32), 25.0) > 0.05
        lab, _ = label(grown)
        py, px = np.unravel_index(np.argmax(s), s.shape)
        mask = mask & (lab == lab[py, px])
    ys, xs = np.where(mask)
    if ys.size < 50:
        return 0, a.shape[0], 0, a.shape[1]
    ylo, yhi = np.percentile(ys, [0.5, 99.5])   # percentiles, so a few noise pixels cannot stretch the box
    xlo, xhi = np.percentile(xs, [0.5, 99.5])
    cy, cx = (ylo + yhi) / 2.0, (xlo + xhi) / 2.0
    half = max(yhi - ylo, xhi - xlo) / 2.0 * 1.25 + 40
    half = max(half, 200)
    y0, x0 = int(max(0, cy - half)), int(max(0, cx - half))
    y1, x1 = int(min(a.shape[0], cy + half)), int(min(a.shape[1], cx + half))
    side = min(y1 - y0, x1 - x0)
    return y0, y0 + side, x0, x0 + side


# Pink ramp: black to the pink, then the brightest part toward white.
def pink_ramp(t):
    t = np.clip(t, 0, 1)[..., None]
    low = PINK * np.clip(t / 0.75, 0, 1)
    k = np.clip((t - 0.75) / 0.25, 0, 1) * WHITEN
    rgb = low + k * (1.0 - low)
    return np.clip(rgb, 0, 1)


def stretch(a, sigma):
    x = np.clip((a - BLACK_SIGMA * sigma) / sigma, 0, None)
    soft = max(SOFT_SIGMA - BLACK_SIGMA, 0.5)
    top = float(np.percentile(x, 99.97))
    return np.clip(np.arcsinh(x / soft) / np.arcsinh(max(top, soft) / soft), 0, 1)


def nice_bar(width_arcsec):
    for v in (2.0, 1.0, 0.5, 0.25, 0.2, 0.1, 0.05):
        if v <= width_arcsec * 0.25:
            return v
    return 0.05


def font(size):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


# The picture with the beam in the lower left, a scale bar in the lower right and a title and credit bar.
def make_picture(a, info, name, sigma):
    t = stretch(a, sigma)
    img = Image.fromarray((pink_ramp(t) * 255).astype(np.uint8))
    h, w = a.shape
    k = 1000.0 / w if w < 1000 else 1.0
    if k > 1:
        img = img.resize((int(w * k), int(h * k)), Image.LANCZOS)
        w, h = img.size
    scale = k * info["pixel_arcsec"]
    d = ImageDraw.Draw(img)
    pad = max(18, w // 30)
    bmaj, bmin = info["bmaj_arcsec"] / scale, info["bmin_arcsec"] / scale
    if bmaj > 0:
        ell = Image.new("L", (int(bmaj * 2 + 6), int(bmaj * 2 + 6)), 0)
        e = ImageDraw.Draw(ell)
        c = ell.size[0] / 2.0
        e.ellipse([c - bmaj, c - bmin, c + bmaj, c + bmin], fill=255)
        ell = ell.rotate(info["bpa_deg"] - 90, resample=Image.BICUBIC)   # BPA is east of north; the drawn ellipse starts with its long axis horizontal
        img.paste((255, 255, 255), (pad, h - pad - ell.size[1]), ell)
    bar = nice_bar(w * scale)
    bx1, by = w - pad, h - pad
    bx0 = bx1 - bar / scale
    d.line([bx0, by, bx1, by], fill=(255, 255, 255), width=max(2, w // 400))
    f = font(max(12, w // 55))
    label = f"{bar:g} arcsec"
    d.text(((bx0 + bx1) / 2 - d.textlength(label, font=f) / 2, by - f.size - 10), label, fill=(255, 255, 255), font=f)
    d.text((pad, pad), name, fill=(255, 255, 255), font=font(max(16, w // 32)))
    f2 = font(max(11, w // 70))
    text = (f"ALMA continuum at {info['frequency_hz'] / 1e9:.0f} GHz (about {299792458.0 / info['frequency_hz'] * 1000:.2f} mm); one color, pink = brightness; white ellipse = beam ({info['bmaj_arcsec'] * 1000:.0f} x {info['bmin_arcsec'] * 1000:.0f} mas). "
            "DSHARP large program image (Andrews et al. 2018, ApJL 869, L41), ALMA (ESO/NAOJ/NRAO); stretched and colored by this project (Raellabot).")
    probe = ImageDraw.Draw(Image.new("RGB", (4, 4)))
    lines, cur = [], ""
    for word in text.split():
        if cur and probe.textlength(cur + " " + word, font=f2) > w - 2 * pad:
            lines.append(cur)
            cur = word
        else:
            cur = (cur + " " + word).strip()
    lines.append(cur)
    band = 16 + len(lines) * (f2.size + 6)
    canvas = Image.new("RGB", (w, h + band), (0, 0, 0))
    canvas.paste(img, (0, 0))
    dc = ImageDraw.Draw(canvas)
    for i, line in enumerate(lines):
        dc.text((pad, h + 8 + i * (f2.size + 6)), line, fill=(235, 235, 235), font=f2)
    return canvas


def main():
    out = sys.argv[1]
    os.makedirs(os.path.join(out, "samples"), exist_ok=True)
    names = sys.argv[2:] or all_names()
    log(f"{len(names)} disks: {names}")
    recipes = []
    for name in names:
        try:
            a, info = load(name)
            sigma = noise_of(a)
            beam_px = info["bmaj_arcsec"] / info["pixel_arcsec"]
            y0, y1, x0, x1 = crop_box(a, sigma, info["pixel_arcsec"], beam_px, name in MAIN_ONLY)
            crop = a[y0:y1, x0:x1]
            small = crop
            if min(crop.shape) > 1000:   # keep the saved copy small enough for the results branch
                n2 = (crop.shape[0] // 2) * 2, (crop.shape[1] // 2) * 2
                small = crop[:n2[0], :n2[1]].reshape(n2[0] // 2, 2, n2[1] // 2, 2).mean(axis=(1, 3))
            np.savez_compressed(os.path.join(out, "samples", f"{name}.npz"), data=small.astype(np.float32), binned=int(small is not crop) + 1, sigma=sigma, info=json.dumps(info))
            make_picture(crop, info, PRETTY.get(name, name), sigma).save(os.path.join(out, f"{name}-pink.jpg"), quality=92)
            recipes.append({"name": name, "sigma_jy_beam": sigma, "peak_over_sigma": float(crop.max() / sigma), "crop_px": [x1 - x0, y1 - y0], "info": info})
            log(f"{name}: peak {crop.max() / sigma:.0f} sigma, crop {x1 - x0}x{y1 - y0} px at {info['pixel_arcsec'] * 1000:.1f} mas/px")
            os.remove(os.path.join(WORK, f"{name}_continuum.fits"))
        except Exception as e:  # noqa: BLE001
            log(f"{name} failed: {e}")
    json.dump(recipes, open(os.path.join(out, "recipes.json"), "w"), indent=1)
    log(f"done: {len(recipes)} pictures")
    if not recipes:
        sys.exit(1)


T0 = time.time()
main()
