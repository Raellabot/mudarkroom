# Renders subjects as the darkroom page shows them, from the files the live page serves; usage: render_live.py OUTDIR ID...
import gzip, json, os, sys, urllib.request
import numpy as np
from PIL import Image

BASE = "https://cosmic-darkroom.ponderakaclaude.workers.dev/data"
sys.path.insert(0, os.path.dirname(__file__))


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "mudarkroom-render"}), timeout=120).read()


def level_at(pcts, p):
    ks = sorted(pcts, key=float)
    return float(np.interp(p, [float(k) for k in ks], [pcts[k] for k in ks]))


def render(m, arrays, black, preset):
    opt = [l for l in m["layers"] if l["kind"] != "xray"]
    s = np.zeros(arrays[opt[0]["id"]].shape + (3,), np.float32)
    for l in opt:
        b = level_at(l["encoding"]["percentiles"], black)
        x = np.maximum(arrays[l["id"]] - b, 0) / max(1 - b, 1e-6)
        c = np.array([int(preset["colors"][l["id"]][i:i + 2], 16) / 255 for i in (1, 3, 5)], np.float32)
        s += x[..., None] * preset["weights"].get(l["id"], 1.0) * c
    s *= 3.0 / max(len(opt), 3)
    I = s.mean(-1)
    q, st = preset["softening"], preset["stretch"]
    slope = 0.1 / np.arcsinh(0.1 * q)
    fac = np.where(I > 0, np.arcsinh(I * q / st) * slope / np.maximum(I, 1e-12), 0)
    rgb = s * fac[..., None]
    mx = rgb.max(-1, keepdims=True)
    rgb = np.where(mx > 1, rgb / np.maximum(mx, 1e-12), rgb)
    rgb = np.clip(rgb / (preset["white_point"] / 100), 0, 1)
    luma = rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    rgb = np.clip(luma[..., None] + (rgb - luma[..., None]) * preset["saturation"], 0, 1)
    return Image.fromarray((rgb * 255 + 0.5).astype(np.uint8))


out = sys.argv[1]
os.makedirs(out, exist_ok=True)
for tid in sys.argv[2:]:
    try:
        m = json.loads(get(f"{BASE}/{tid}/manifest.json"))
        w, h = m["grid"]["width"], m["grid"]["height"]
        arrays = {}
        for l in m["layers"]:
            raw = get(f"{BASE}/{tid}/{l['file']}")
            if raw[:2] == b"\x1f\x8b":
                raw = gzip.decompress(raw)
            arrays[l["id"]] = np.frombuffer(raw, "<f2").astype(np.float32).reshape(h, w)
        p = m["presets"][0]
        render(m, arrays, p["black_point"], p).save(f"{out}/{tid}.jpg", quality=92)
        render(m, arrays, 1, p).save(f"{out}/{tid}-black1.jpg", quality=92)
        print(tid, w, h, flush=True)
    except Exception as e:
        print(tid, "failed:", e, flush=True)
