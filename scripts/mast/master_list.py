# Cleans the sweep list into a master list of picture candidates; usage: master_list.py SWEEP_CSV OUT_CSV [TARGETS_DIR]
import collections, csv, glob, json, math, os, re, sys

SKIP = re.compile(r"^(GD-?\d+|GRW|G191|P330|P177|LDS|BD[+-]|HD-?\d|WD-?\d|VB-?\d|LHS|2M\d|2MASS|FLAT|.*POINTED.*|.*CAL($|-)|.*ASTROM.*|PAR(-|\d|$)|PARALLEL|MULTIPLE|APT_|SNAP-|BACKGROUND|SKY|DARK|OFFSET|EARTH|MOON|JUPITER|SATURN|URANUS|NEPTUNE|MARS|PLUTO|CHARON|TITAN|EUROPA|GANYMEDE|CALLISTO|IO$|TROJAN|COMET|\d{4}[A-Z]{2}\d*|.*-MAPS|.*FLAT.*|.*STANDARD.*|.*ASTEROID.*)", re.I)
NAMED = re.compile(r"^(NGC|IC|M-?\d|ARP|ABELL|A\d|MACS|SMACS|UGC|HCG|NAME|ETA|HH|LDN|LYNDS|SH2|RCW|PN|NAME-|WESTERLUND|TRUMPLER|PISMIS|COLLINDER|MESSIER|3C|CL|RXJ|SDSS|COSMOS|GOODS|UDF|HUDF|HDF|LMC|SMC|M31|M33|M81|M82|M87|M101)", re.I)


def norm(n):
    return re.sub(r"[^A-Z0-9]", "", n.upper())


def main():
    rows = list(csv.DictReader(open(sys.argv[1])))
    keep = []
    for r in rows:
        r["filters"] = int(r["filters"])
        r["exposure_s"] = int(float(r["exposure_s"] or 0))
        r["ra"] = float(r["ra"]) if r["ra"] != "" else None
        r["dec"] = float(r["dec"]) if r["dec"] != "" else None
        if r["filters"] < 3 or SKIP.match(r["target"].strip()):
            continue
        keep.append(r)
    # Merge groups within 90 arcsec of each other (any program), keeping the richest one.
    cell = 90.0 / 3600
    grid = collections.defaultdict(list)
    parent = list(range(len(keep)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, r in enumerate(keep):
        if r["ra"] is None:
            continue
        cx, cy = int(r["ra"] / cell), int(r["dec"] / cell)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in grid[(cx + dx, cy + dy)]:
                    o = keep[j]
                    d = math.hypot((r["ra"] - o["ra"]) * math.cos(math.radians(r["dec"])), r["dec"] - o["dec"]) * 3600
                    if d < 90:
                        parent[find(i)] = find(j)
        grid[(cx, cy)].append(i)
    clusters = collections.defaultdict(list)
    for i in range(len(keep)):
        clusters[find(i)].append(keep[i])

    def score(r):
        single = "+" not in r["instruments"]
        return min(r["filters"], 8) * 2 + (3 if single else 0) + (1 if r["exposure_s"] > 3000 else 0) + (2 if NAMED.match(r["target"]) else 0)
    master = []
    for members in clusters.values():
        best = max(members, key=score)
        best = dict(best)
        best["score"] = score(best)
        best["groups_merged"] = len(members)
        best["programs"] = " ".join(sorted({m["collection"] + "|" + m["proposal"] for m in members}))
        master.append(best)
    # Mark subjects already made, by program plus target or by name.
    made = set()
    names = set()
    if len(sys.argv) > 3:
        for f in glob.glob(os.path.join(sys.argv[3], "*.json")):
            t = json.load(open(f))
            spec = t.get("spec", "")
            parts = [p.split("+") for p in spec.split("|")]
            if len(parts) >= 3:
                for tg, pr in zip(parts[1], parts[2]):
                    made.add((pr, norm(tg)))
            names.update(norm(x) for x in [t.get("name", "")] + t.get("aliases", []))
    for m in master:
        m["covered"] = int(any((m["proposal"], norm(m["target"])) in made or any(p.split("|")[1] == pr and norm(m["target"]) == nt for p in m["programs"].split() for pr, nt in made) for _ in [0]) or norm(m["target"]) in names)
    master.sort(key=lambda m: (-m["score"], m["collection"], m["target"]))
    fields = ["collection", "proposal", "target", "filters", "filter_names", "instruments", "ra", "dec", "exposure_s", "title", "score", "groups_merged", "programs", "covered"]
    with open(sys.argv[2], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(master)
    for coll in ("HST", "JWST"):
        mine = [m for m in master if m["collection"] == coll]
        print(f"{coll}: {len(mine)} subjects after cleaning ({sum(m['covered'] for m in mine)} already made); with 4+ filters {sum(1 for m in mine if m['filters'] >= 4)}; 5+ {sum(1 for m in mine if m['filters'] >= 5)}")


if __name__ == "__main__":
    main()
