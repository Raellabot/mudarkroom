# Checks what the page says about a target (name, blurb, story) against what was actually exported, so a description can't claim something the data doesn't show.
#
#   check_text.py --manifest out/export/<id>/manifest.json targets/<id>.json     one target, from a local manifest (used by export.yml)
#   check_text.py --r2 all|ID...                                                 targets already in the R2 bucket (used by check-text.yml)
#
# Exit status 1 on any ERROR. Checks, from the layers in the manifest:
#   filter count words ("six filters"), wavelength ranges ("0.9 to 4.4 microns"), listed wavelengths ("435, 606 and 850 nanometers"),
#   telescope and instrument names (a blurb may not name Webb, MIRI, WFC3... unless a layer comes from it).
# And, with --strict (export.yml) or whenever a target has a "facts" record:
#   every quantity in the blurb or story (a distance, a size, an age) must be listed in targets/<id>.json "facts"."claims" with a source link,
#   so each number has a page someone looked at. Format:
#     "facts": {"checked": "2026-11-05", "claims": [{"text": "about 45 million light-years", "source": "https://..."}]}
# What the script cannot do is read the source for you: facts are only as good as the page behind the link, so open it when you write the claim.
import argparse, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
MISSION_WORDS = {"Hubble": {"HST"}, "Webb": {"JWST"}, "Chandra": {"Chandra"}, "ALMA": {"ALMA"}, "Cassini": {"Cassini"}, "Juno": {"Juno"}, "JunoCam": {"Juno"}}
INSTRUMENT_WORDS = {"NIRCam": "NIRCAM", "MIRI": "MIRI", "WFC3": "WFC3", "ACS": "ACS", "WFPC2": "WFPC2", "UVIS": "UVIS"}
QUANTITY = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s*(million|billion|thousand)?\s*(light-years|light years|kilometers|kilometres|km|years old|years|parsecs|solar masses)", re.I)


def quantities(text):
    out = []
    for m in QUANTITY.finditer(text):
        out.append(" ".join(p for p in (m.group(1), (m.group(2) or "").lower(), m.group(3).lower()) if p))
    return out


def norm(s):
    return re.sub(r"\s+", " ", s.lower().replace("light years", "light-years").replace("kilometres", "kilometers"))


def layer_facts(manifest):
    opt = [l for l in manifest["layers"] if l["kind"] != "xray"]
    return {
        "n": len(opt), "missions": {l["mission"] for l in manifest["layers"]},
        "instruments": {l["instrument"].upper() for l in manifest["layers"]},
        "centers_um": sorted(l["center_nm"] / 1000 for l in opt if l.get("center_nm")),
        "labels": [l["label"] for l in opt],
    }


def check(cfg, manifest, strict=False):
    """Returns (errors, notes) for one target."""
    errs, notes = [], []
    f = layer_facts(manifest)
    texts = {"name": cfg["name"], "blurb": cfg["blurb"], "story": cfg.get("story", "")}
    allt = " ".join(texts.values())
    um = f["centers_um"]
    for where in ("blurb", "story"):
        t = texts[where]
        for m in re.finditer(r"(?<!of )(?<!of the )\b(two|three|four|five|six|seven|eight|nine|ten)\b(?:\s+[\w-]+)?\s+(?:filters?|colou?rs|bands?|wavelengths?)\b", t, re.I):
            n = NUM[m.group(1).lower()]
            if n != f["n"] and "medium" not in m.group(0) and "detector" not in m.group(0):
                errs.append(f"{where}: says '{m.group(0)}' but {f['n']} optical layers were exported ({', '.join(f['labels'])})")
        for m in re.finditer(r"([\d.]+)\s*(?:to|-|–)\s*([\d.]+)\s*(microns|µm|micrometers|micrometres|nanometers|nanometres|nm)", t):
            lo, hi, unit = float(m.group(1)), float(m.group(2)), m.group(3)
            if unit in ("nanometers", "nanometres", "nm"):
                lo, hi = lo / 1000, hi / 1000
            if not um:
                continue
            tol = 0.12 * max(1, hi)  # the page rounds band centers; filters' edges differ from their centers
            if abs(lo - um[0]) > max(0.12, 0.12 * um[0]) or abs(hi - um[-1]) > max(0.12, 0.12 * um[-1]):
                errs.append(f"{where}: says '{m.group(0)}' but the layer centers run {um[0]:.2f} to {um[-1]:.2f} µm")
        for m in re.finditer(r"\(([\d, and]+)\s*nanometers\)", t):
            listed = [int(x) for x in re.findall(r"\d+", m.group(1))]
            have = [u * 1000 for u in um]
            for v in listed:
                if not any(abs(v - h) <= 0.06 * v for h in have):
                    errs.append(f"{where}: lists {v} nm but no layer is centered near it ({', '.join(f'{h:.0f}' for h in have)} nm)")
    # telescope and instrument names
    named = re.sub(r"Hubble (Ultra )?Deep Field|Hubble Ultra Deep Field", "", texts["blurb"])  # a field's name is not a claim about the data
    for word, ms in MISSION_WORDS.items():
        if re.search(rf"\b{word}\b", named) and not (ms & f["missions"]):
            errs.append(f"blurb names {word} but no layer comes from it (layers: {sorted(f['missions'])})")
    for word, key in INSTRUMENT_WORDS.items():
        if re.search(rf"\b{word}\b", texts["blurb"]) and not any(key in i for i in f["instruments"]):
            errs.append(f"blurb names {word} but no layer comes from that instrument (layers: {sorted(f['instruments'])})")
    for m, w in (("HST", "Hubble"), ("JWST", "Webb"), ("Chandra", "Chandra")):
        if m in f["missions"] and not re.search(rf"\b{w}\b", texts["blurb"]):
            notes.append(f"layers come from {w} but the blurb does not name it")
    for word, key in (("narrow-angle", "NAC"), ("wide-angle", "WAC")):
        if word in texts["blurb"] and not any(key in i for i in f["instruments"]):
            errs.append(f"blurb says {word} camera but the layers come from {sorted(f['instruments'])}")
    # numbers need a source
    facts = cfg.get("facts")
    qs = quantities(texts["blurb"] + " . " + texts["story"])
    if facts or strict:
        claims = (facts or {}).get("claims", [])
        for c in claims:
            if not str(c.get("source", "")).startswith("https://"):
                errs.append(f"fact '{c.get('text')}' has no https source link")
        have = norm(" ".join(c.get("text", "") for c in claims))
        for q in qs:
            if norm(q) not in have:
                errs.append(f"'{q}' is in the description but not in facts.claims (add it with the page you checked it against)")
        if qs and not (facts or {}).get("checked"):
            errs.append("facts.checked (date) is missing")
    elif qs:
        notes.append("has quantities with no facts record: " + "; ".join(qs))
    return errs, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest")
    ap.add_argument("--r2", nargs="+")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("configs", nargs="*")
    a = ap.parse_args()
    jobs = []
    if a.manifest:
        for c in a.configs:
            jobs.append((json.load(open(c)), json.load(open(a.manifest))))
    else:
        import boto3
        s3 = boto3.client("s3", endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com", aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
                          aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"], region_name="auto")
        bucket = os.environ.get("R2_BUCKET", "interactive-darkroom")
        ids = a.r2
        if ids == ["all"]:
            ids = sorted(os.path.basename(p)[:-5] for p in os.listdir(os.path.join(HERE, "..", "targets")) if p.endswith(".json"))
        for tid in ids:
            cfg = json.load(open(os.path.join(HERE, "..", "targets", f"{tid}.json")))
            try:
                m = json.loads(s3.get_object(Bucket=bucket, Key=f"{cfg['id']}/manifest.json")["Body"].read())
            except Exception as e:
                print(f"{tid}: no manifest in R2 ({type(e).__name__})")
                continue
            if "{light}" not in cfg["blurb"] and (m.get("blurb") != cfg["blurb"] or m.get("name") != cfg["name"]):
                print(f"{tid}: ERROR the manifest in R2 has an older name or blurb than targets/{tid}.json (run update-presets)")
            jobs.append((cfg, m))
    bad = 0
    for cfg, m in jobs:
        errs, notes = check(cfg, m, a.strict)
        f = layer_facts(m)
        um = f["centers_um"]
        rng = f"{um[0]:.2f}-{um[-1]:.2f}um" if um else "-"
        print(f"{cfg['id']}: {f['n']} optical [{', '.join(f['labels'])}] {rng} {sorted(f['instruments'])}")
        for e in errs:
            print(f"   ERROR {e}")
        for n in notes:
            print(f"   note  {n}")
        bad += bool(errs)
    print(f"{len(jobs)} targets checked, {bad} with errors")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
