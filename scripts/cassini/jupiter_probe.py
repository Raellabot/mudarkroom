# Probes which products the 2000-01 Cassini Jupiter frames have: saves the OPUS file lists and the head of each image's VICAR label for a few sample frames.
# usage: jupiter_probe.py OUTDIR OPUSID [OPUSID...]
import json, os, sys, urllib.request

OPUS = "https://opus.pds-rings.seti.org/opus/api"
UA = "mudarkroom (github.com/Raellabot/mudarkroom)"


def get(url, rng=None):
    h = {"User-Agent": UA}
    if rng:
        h["Range"] = f"bytes=0-{rng}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=120) as r:
        return r.read()


def main(out, *ids):
    os.makedirs(out, exist_ok=True)
    for oid in ids:
        files = json.loads(get(f"{OPUS}/files/{oid}.json"))
        json.dump(files, open(os.path.join(out, oid + "-files.json"), "w"), indent=1)
        for kind, urls in files["data"][oid].items():
            for u in urls:
                if u.endswith(".IMG") or u.endswith(".LBL"):
                    try:
                        b = get(u, 12000)
                        open(os.path.join(out, oid + "-" + u.split("/")[-1] + ".head"), "wb").write(b)
                        print(kind, u, len(b), flush=True)
                    except Exception as e:
                        print(kind, u, "failed", e, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
