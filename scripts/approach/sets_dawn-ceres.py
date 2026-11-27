# Writes the Dawn Ceres approach list from the calibrated framing camera index (DWNCAFC2_1B): clear-filter frames from approach through RC3, short exposures only, satellite searches left out.
# usage: sets_dawn-ceres.py OUT.csv
import csv, datetime, re, sys, urllib.request

ROOT = "https://sbnarchive.psi.edu/pds3/dawn/fc/DWNCAFC2_1B"
UA = "mudarkroom (github.com/Raellabot/mudarkroom)"
MAX_EXPOSURE_MS = 300


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=300) as r:
        return r.read()


def main(out):
    lines = get(f"{ROOT}/INDEX/INDEX.TAB").decode("latin1").splitlines()
    head = [h.strip() for h in lines[0].split(",")]
    rows = []
    for line in lines[1:]:
        v = [x.strip().strip('"').strip() for x in next(csv.reader([line]))]
        if len(v) != len(head):
            continue
        r = dict(zip(head, v))
        f = r["FILE_SPECIFICATION_NAME"]
        m = re.search(r"F(\d)[A-Z]\.LBL$", f)
        if not f.startswith("/DATA/FITS/") or not m or m.group(1) != "1" or "_SAT" in f.upper() or float(r["EXPOSURE_DURATION"]) > MAX_EXPOSURE_MS:
            continue
        t = r["START_TIME"]
        y, doy, rest = t[:4], int(t[5:8]), t[9:]
        iso = (datetime.date(int(y), 1, 1) + datetime.timedelta(doy - 1)).isoformat() + "T" + rest
        url = ROOT + f[:-4] + ".FIT"
        rows.append(dict(label=r["PRODUCT_ID"], time=iso, caption="Dawn", kind="fits", red=url, green=url, blue=url))
    rows.sort(key=lambda r: r["time"])
    seen, uniq = set(), []
    for r in rows:
        if r["label"] not in seen:
            seen.add(r["label"])
            uniq.append(r)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(uniq[0]))
        w.writeheader()
        w.writerows(uniq)
    print("dawn frames", len(uniq), uniq[0]["time"], uniq[-1]["time"], flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
