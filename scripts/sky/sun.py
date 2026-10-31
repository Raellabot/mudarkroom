# Makes ghoul pictures of the Sun from SDO/AIA extreme-ultraviolet frames served by Helioviewer (usage: sun.py OUTDIR DATE [DATE ...]; settings in the environment).
import io, os, sys, time, urllib.parse, urllib.request
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render

UA = os.environ.get("CONTACT_INFO") or "CONTACT-INFO"
BLACKS = [float(v) for v in os.environ.get("BLACKS", "").split(",") if v.strip()]
BLACK = float(os.environ.get("SUN_BLACK", "65"))
SIDE = int(os.environ.get("SUN_SIDE", "2048"))
# Helioviewer source ids of the AIA channels, in wavelength order (angstroms).
SOURCES = [("131", 9), ("171", 10), ("193", 11), ("211", 12), ("304", 13), ("335", 14)]


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def fetch(date, sid):
    url = "https://api.helioviewer.org/v2/getJP2Image/?" + urllib.parse.urlencode({"date": date, "sourceId": sid})
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            return urllib.request.urlopen(req, timeout=120).read()
        except Exception as e:  # noqa: BLE001
            log(f"sourceId {sid}: attempt {attempt + 1} failed: {e}")
            time.sleep(5)
    raise RuntimeError(f"could not fetch sourceId {sid}")


def reduce(a, side):
    f = max(1, a.shape[0] // side)
    h, w = (a.shape[0] // f) * f, (a.shape[1] // f) * f
    return a[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3)).astype(np.float32)


T0 = time.time()
outdir, dates = sys.argv[1], sys.argv[2:]
os.makedirs(outdir, exist_ok=True)
for date in dates:
    arrays, names = [], []
    for ang, sid in SOURCES:
        blob = fetch(date, sid)
        im = Image.open(io.BytesIO(blob))
        a = np.asarray(im).astype(np.float32)
        if a.ndim == 3:
            a = a[..., 0]
        log(f"{date} AIA {ang}: {im.mode}, {a.shape[1]}x{a.shape[0]}, min {a.min():.0f}, max {a.max():.0f}")
        arrays.append(reduce(a, SIDE))
        names.append(f"AIA {ang} Å")
    label = "sun-" + date.replace(":", "").replace("-", "")[:13].replace("T", "-")
    made = render.render(outdir, label, names, arrays, f"SDO AIA, {date} UTC, from Helioviewer JPEG 2000 frames", BLACK, BLACKS)
    log(f"{label}: {len(made)} pictures")
