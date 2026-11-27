# Moon positions for the Webb planet subjects: reads each live manifest, finds the exposure time at MAST and asks JPL Horizons where the moons were, as seen from Webb; usage: webb_moons.py OUT.json
import json, sys, urllib.parse, urllib.request
from astroquery.mast import Observations

BASE = "https://cosmic-darkroom.ponderakaclaude.workers.dev/data"
MOONS = {"uranus-webb": ["799"] + [str(i) for i in range(701, 716)] + ["725", "726", "727"], "neptune-webb": ["899"] + [str(i) for i in range(801, 809)] + ["814"]}


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "mudarkroom-moons"}), timeout=120).read()


out = {}
for tid, bodies in MOONS.items():
    m = json.loads(get(f"{BASE}/{tid}/manifest.json"))
    info = {"grid": m["grid"], "layers": [{"id": l["id"], "label": l["label"], "obs_ids": l["obs_ids"]} for l in m["layers"]]}
    t = Observations.query_criteria(obs_id=m["layers"][0]["obs_ids"][0])
    info["mast"] = [{k: str(r[k]) for k in ("obs_id", "t_min", "t_max", "target_name", "s_ra", "s_dec")} for r in t]
    jd = float(t["t_min"][0]) + 2400000.5 + (float(t["t_max"][0]) - float(t["t_min"][0])) / 2
    info["jd"] = jd
    rows = {}
    for b in bodies:
        q = {"format": "json", "COMMAND": f"'{b}'", "EPHEM_TYPE": "OBSERVER", "CENTER": "'500@-170'", "TLIST": f"'{jd}'", "QUANTITIES": "'2,13,20'", "CSV_FORMAT": "'YES'", "ANG_FORMAT": "'DEG'", "EXTRA_PREC": "'YES'"}
        r = json.loads(get("https://ssd.jpl.nasa.gov/api/horizons.api?" + urllib.parse.urlencode(q)))["result"]
        name = next((l for l in r.splitlines() if l.startswith("Target body name")), "")
        line = r.split("$$SOE")[1].split("$$EOE")[0].strip() if "$$SOE" in r else r[-400:]
        rows[b] = {"name": name, "row": line}
        print(tid, b, name[:60], line[:120], flush=True)
    info["horizons"] = rows
    out[tid] = info
json.dump(out, open(sys.argv[1], "w"), indent=1)
