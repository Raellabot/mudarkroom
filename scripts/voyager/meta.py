# Saves OPUS metadata (geometry included) for the green frame of each Voyager color set in a list; usage: meta.py SETS.csv OUT.jsonl
import csv, json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

API = "https://opus.pds-rings.seti.org/opus/api"


def get(oid):
    for i in range(4):
        try:
            req = urllib.request.Request(f"{API}/metadata/{oid}.json", headers={"User-Agent": "cosmic-darkroom (github.com/Raellabot/mudarkroom)"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except Exception as e:
            err = repr(e)
            time.sleep(3 * (i + 1))
    return {"error": err}


def main(sets, out):
    rows = list(csv.DictReader(open(sets)))
    with ThreadPoolExecutor(6) as ex:
        metas = list(ex.map(lambda r: get(r["green"]), rows))
    with open(out, "w") as f:
        for r, m in zip(rows, metas):
            f.write(json.dumps({"label": r["label"], "green": r["green"], "meta": m}) + "\n")
    print(len(rows), "sets;", sum("error" in m for m in metas), "failed")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
