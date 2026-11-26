# Asks JPL Horizons where each planet and moon sat in the sky as seen from Voyager at each frame's time; usage: horizons.py FRAMES.json OUT.json
import json, sys, time, urllib.parse, urllib.request
from collections import defaultdict

API = "https://ssd.jpl.nasa.gov/api/horizons.api"
frames = json.load(open(sys.argv[1]))
jobs = defaultdict(set)
for f in frames.values():
    for bid in f["ids"]:
        jobs[(f["sc"], bid)].add(f["jd"])
out = {}
for (sc, bid), jds in sorted(jobs.items()):
    jds = sorted(jds)
    for i in range(0, len(jds), 40):
        part = jds[i:i + 40]
        q = {"format": "json", "COMMAND": f"'{bid}'", "OBJ_DATA": "NO", "MAKE_EPHEM": "YES", "EPHEM_TYPE": "OBSERVER", "CENTER": f"'500@-3{sc}'",
             "TLIST": "'" + " ".join(f"{j:.7f}" for j in part) + "'", "TIME_TYPE": "UT", "QUANTITIES": "'2,13'", "ANG_FORMAT": "DEG", "CSV_FORMAT": "YES", "EXTRA_PREC": "YES"}
        for attempt in range(4):
            try:
                txt = json.load(urllib.request.urlopen(API + "?" + urllib.parse.urlencode(q), timeout=120))["result"]
                break
            except Exception as e:
                print("retry", sc, bid, e, flush=True)
                time.sleep(5 * (attempt + 1))
        if "$$SOE" not in txt:
            print("no ephemeris", sc, bid, txt[:800], flush=True)
            continue
        rows = txt.split("$$SOE")[1].split("$$EOE")[0].strip().splitlines()
        for jd, row in zip(part, rows):
            c = [x.strip() for x in row.split(",")]
            nums = [x for x in c[1:] if x and x.replace(".", "").replace("-", "").replace("+", "").isdigit()]
            out[f"{sc}|{bid}|{jd:.7f}"] = {"row": row}
        print(sc, bid, len(rows), "of", len(part), flush=True)
        time.sleep(1)
json.dump(out, open(sys.argv[2], "w"), indent=1)
