# Writes build/targets/<id>.json for each Voyager set in a build list, from the OPUS metadata and checked body facts.
# Usage: targets.py BUILD.csv META.jsonl FACTS.json OUTDIR LIST.txt METRICS.json
# The OPUS target is what the camera was pointed for, not always what is in the frame; RETARGET and DROP hold the ones checked by eye.
RETARGET = {"amalthea-v1n-c1607925": "Jupiter"}   # pointed for Amalthea; the frame shows Jupiter's clouds and the Great Red Spot
DROP = {"saturn-v2w-c4392542": "a green patch over the left of Saturn that the sky fix cannot take out",
        "metis-v1w-c1637337": "Jupiter's clouds, not Metis, with the colors badly out of register",
        "jupiter-rings-v2n-c2063044": "a plain field of Jupiter's haze; no ring is visible",
        "jupiter-v1n-c1601459": "Jupiter's limb with the colors torn apart (the clouds moved between filters)",
        "jupiter-v1n-c1630959": "a close-up of Jupiter's clouds that comes out in false pink and green",
        "jupiter-v1n-c1635403": "a close-up of Jupiter's clouds that comes out in false green",
        "jupiter-v1w-c1638806": "a nearly blank gray field of Jupiter's clouds",
        "io-v1w-c1639055": "Io with the colors badly out of register",
        "jupiter-v2n-c2054005": "a close-up of Jupiter's clouds that comes out in false pink and green",
        "jupiter-v2n-c2058623": "a close-up of Jupiter's clouds that comes out in false colors",
        "jupiter-v2n-c2062914": "a close-up of Jupiter's clouds that comes out in false colors",
        "jupiter-v2w-c2064424": "a close-up of Jupiter's clouds that comes out in false pink and green",
        "jupiter-v2w-c2064636": "Jupiter's limb in false colors",
        "uranus-v2w-c2678240": "Uranus out of register, a purple disk with a yellow rim",
        "uranus-v2w-c2681001": "Uranus with a magenta streak across it",
        "uranus-v2w-c2681037": "a close view of Uranus in false purple",
        "uranus-v2w-c2682623": "Uranus with wide yellow and blue rims (out of register)",
        "jupiter-v1n-c1574431": "Jupiter with a colored line torn across it",
        "jupiter-v1n-c1577424": "Jupiter with colored dashes across it"}
import csv, json, math, os, sys
from collections import Counter
from datetime import datetime

CHECKED = "2026-11-26"
OPUS_API = "https://opus.pds-rings.seti.org/opus/api/metadata/{}.json"
NSSDC = "https://nssdc.gsfc.nasa.gov/planetary/voyager-periapse.html"
FOV = {"NAC": 0.0074, "WAC": 0.0553}  # radians, roughly; only used to tell a whole disk from a close-up
PERIAPSE = {(1, "Jupiter"): "1979-03-05T12:05", (2, "Jupiter"): "1979-07-09T22:29", (1, "Saturn"): "1980-11-12T23:46",
            (2, "Saturn"): "1981-08-26T03:24", (2, "Uranus"): "1986-01-24T17:59", (2, "Neptune"): "1989-08-25T03:56"}
SYSTEM = {"Io": "Jupiter", "Europa": "Jupiter", "Ganymede": "Jupiter", "Callisto": "Jupiter", "Amalthea": "Jupiter", "Metis": "Jupiter",
          "Jupiter Rings": "Jupiter", "Titan": "Saturn", "Rhea": "Saturn", "Enceladus": "Saturn", "Tethys": "Saturn", "Dione": "Saturn",
          "Iapetus": "Saturn", "Saturn Rings": "Saturn", "Triton": "Neptune"}
SHOWN = {"Saturn Rings": "Saturn's Rings", "Jupiter Rings": "Jupiter's Rings"}
GEOKEY = {"Saturn Rings": "saturn", "Jupiter Rings": "jupiter"}


def month_day(t):
    return t.strftime("%B %-d, %Y")


def round_km(d):
    if d >= 1e6:
        v = d / 1e6
        s = f"{v:.0f}" if v >= 10 else f"{v:.1f}".rstrip("0").rstrip(".")
        return f"{s} million"
    p = 10 ** (int(math.log10(d)) - 1)
    return f"{round(d / p) * p:,.0f}"


def geometry(meta, target):
    key = GEOKEY.get(target, target.split()[0].lower())
    blk = next((v for k, v in meta.items() if k.lower().startswith(key + " surface geometry")), None)
    if not blk:
        return None, None
    d, ph = blk.get(f"SURFACEGEO{key}_centerdistance1"), blk.get(f"SURFACEGEO{key}_centerphaseangle1")
    return (float(d) if d else None), (float(ph) if ph else None)


def view(target, f, cam, d, ph, radius, lf=0.0):
    shown = SHOWN.get(target, target)
    if target in SHOWN:
        if ph is not None and ph > 120:
            return f"{shown} lit from behind"
        return f"{shown} in sunlight"
    if d is None:
        return f"A view of {shown}"
    size = 2 * radius / d / FOV[cam]
    light = ""
    if ph is not None:
        light = (" fully lit" if ph < 25 else " nearly fully lit" if ph < 50 else " partly in shadow" if ph < 95
                 else " half in night" if ph < 120 else " as a crescent" if ph < 150 else " as a thin crescent")
    expect = min(1.0, math.pi * (size / 2) ** 2)
    if lf >= 0.55 and size >= 0.06:
        return f"A close view of part of {shown}{',' + light if light else ''}"
    if lf > 2.5 * expect + 0.02 and lf > 0.15:
        return f"{shown} filling much of the frame{',' + light if light else ''}"
    if size < 0.06:
        return f"{shown}, a small point of light{', seen' + light if light.strip() else ''}"
    if size < 1.0:
        return f"The whole of {shown}{',' + light if light else ''}"
    return f"A close view of part of {shown}{',' + light if light else ''}"


def when_line(sc, system, t):
    p = datetime.fromisoformat(PERIAPSE[(sc, system)])
    days = (t - p).total_seconds() / 86400
    near = f"Voyager {sc}'s closest approach to {system} on {month_day(p)}"
    if abs(days) < 1:
        h = round(abs(days) * 24)
        if h == 0:
            return f"within the hour of {near}"
        return f"{h} hour{'s' if h != 1 else ''} {'before' if days < 0 else 'after'} {near}"
    n = round(abs(days))
    return f"{n} day{'s' if n != 1 else ''} {'before' if days < 0 else 'after'} {near}"


def main(build, metafile, factsfile, outdir, listfile, metricsfile):
    rows = [dict(r, target=RETARGET.get(r["label"], r["target"])) for r in csv.DictReader(open(build)) if r["label"] not in DROP]
    lit = {m[1]: m[3] for m in json.load(open(metricsfile))}
    meta = {json.loads(l)["label"]: json.loads(l)["meta"] for l in open(metafile)}
    facts = json.load(open(factsfile))
    base = []
    for r in rows:
        t = datetime.fromisoformat(r["time"][:19])
        sc = int(r["spacecraft"])
        base.append((r, t, sc, f"{SHOWN.get(r['target'], r['target'])} from Voyager {sc}, {month_day(t)}"))
    dup = Counter(b[3] for b in base)
    ids = []
    for r, t, sc, name in base:
        target = r["target"]
        fx = facts[target]
        cam = "NAC" if r["camera"].startswith("Narrow") else "WAC"
        system = SYSTEM.get(target, target)
        if dup[name] > 1:
            name += t.strftime(", %H:%M UTC")
        cnum = r["label"].split("-")[-1]
        tid = f"{target.lower().replace(' ', '-')}-voyager-{sc}-{cnum}"
        d, ph = geometry(meta[r["label"]], target)
        claims = list(fx["claims"])
        lead = view(target, fx, cam, d, ph, fx.get("radius_km", 1), lit.get(r["label"], 0.0))
        when = when_line(sc, system, t)
        claims.append({"text": f"Voyager {sc} closest approach to {system}: {datetime.fromisoformat(PERIAPSE[(sc, system)]).strftime('%B %-d, %Y %H:%M')} UTC", "source": NSSDC})
        far = ""
        if d is not None and target not in SHOWN:
            km = round_km(d)
            rel = "away" if d > 50 * fx["radius_km"] else f"from {SHOWN.get(target, target)}'s center"
            far = f", from about {km} kilometers {rel}"
            claims.append({"text": f"{km} kilometers from {target}'s center when the green frame was taken (OPUS center distance {d:,.0f} km)", "source": OPUS_API.format(r["green"])})
        if ph is not None:
            claims.append({"text": f"phase angle {ph:.0f} degrees (OPUS center phase angle)", "source": OPUS_API.format(r["green"])})
        story = f"{lead}{far}, {when}. {fx['story']}"
        cols = [c for c, k in (("orange", "orange"), ("green", "green"), ("blue", "blue"), ("violet", "violet")) if r[k]]
        names = ", ".join(cols[:-1]) + " and " + cols[-1]
        lo = "0.40" if r["violet"] else "0.48"
        camword = "narrow-angle" if cam == "NAC" else "wide-angle"
        blurb = f"{fx['lead']} Seen by Voyager {sc}'s {camword} camera through {names} filters: visible light only, {lo} to 0.62 µm."
        cfg = {
            "id": tid, "name": name, "blurb": blurb, "type": fx["type"],
            "aliases": fx["aliases"] + ["Voyager", f"Voyager {sc}"], "missions": ["Voyager"], "story": story,
            "source": {"kind": "voyager", "spacecraft": f"Voyager {sc}", "camera": cam, "orange": r["orange"], "green": r["green"],
                       "blue": r["blue"] or None, "violet": r["violet"] or None, "green_volume": r["green_file"].split("/")[0], "time": r["time"]},
            "facts": {"checked": CHECKED, "claims": claims},
        }
        cfg["source"] = {k: v for k, v in cfg["source"].items() if v}
        json.dump(cfg, open(os.path.join(outdir, tid + ".json"), "w"), indent=2, ensure_ascii=False)
        open(os.path.join(outdir, tid + ".json"), "a").write("\n")
        ids.append(tid)
    assert len(ids) == len(set(ids))
    open(listfile, "w").write("\n".join(ids) + "\n")
    print(len(ids), "targets")


if __name__ == "__main__":
    main(*sys.argv[1:])
