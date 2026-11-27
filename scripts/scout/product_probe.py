# Probes archive products: for an OPUS id saves its file list and the head of each image or label; for a URL saves a directory listing or the file's head.
# usage: product_probe.py OUTDIR ITEM [ITEM...]  (ITEM: an OPUS id, or a URL)
import json, os, re, sys, urllib.request

OPUS = "https://opus.pds-rings.seti.org/opus/api"
UA = "mudarkroom (github.com/Raellabot/mudarkroom)"


def get(url, rng=None):
    h = {"User-Agent": UA}
    if rng:
        h["Range"] = f"bytes=0-{rng}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=120) as r:
        return r.read()


def safe(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)[-150:]


def head(out, u):
    try:
        b = get(u, 20000)
        open(os.path.join(out, safe(u) + ".head"), "wb").write(b)
        print("saved", u, len(b), flush=True)
    except Exception as e:
        print("failed", u, e, flush=True)


def main(out, *items):
    os.makedirs(out, exist_ok=True)
    for it in items:
        if it.startswith("http"):
            if it.endswith("/"):
                try:
                    t = get(it).decode("latin1")
                    links = re.findall(r'href="([^"?#]+)"', t)
                    open(os.path.join(out, safe(it) + ".list"), "w").write("\n".join(links))
                    print("listed", it, len(links), links[:40], flush=True)
                except Exception as e:
                    print("failed", it, e, flush=True)
            else:
                head(out, it)
            continue
        try:
            files = json.loads(get(f"{OPUS}/files/{it}.json"))
        except Exception as e:
            print("failed", it, e, flush=True)
            continue
        json.dump(files, open(os.path.join(out, it + "-files.json"), "w"), indent=1)
        for kind, urls in files["data"][it].items():
            print(it, kind, urls[:4], flush=True)
            if kind in ("documentation", "rms_index", "inventory") or "geometry" in kind:
                continue
            for u in urls:
                if re.search(r"\.(IMG|LBL|lbl|fit|fits|FIT)$", u):
                    head(out, u)


if __name__ == "__main__":
    main(*sys.argv[1:])
