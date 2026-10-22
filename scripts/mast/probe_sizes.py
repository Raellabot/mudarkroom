# Lists each field's filters and mosaic sizes for the given COLLECTION|PROPOSAL programs without downloading; usage: probe_sizes.py OUTDIR PROGRAM [PROGRAM ...]
import collections, os, re, sys, urllib.parse, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import find_rows

DOWNLOAD = "https://mast.stsci.edu/api/v0.1/Download/file?uri="


def size_of(uri):
    url = DOWNLOAD + urllib.parse.quote(uri, safe=":/")
    req = urllib.request.Request(url, headers={"User-Agent": find_rows.UA, "Range": "bytes=0-0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            cr = r.headers.get("Content-Range") or ""
            m = re.search(r"/(\d+)$", cr)
            total = int(m.group(1)) if m else int(r.headers.get("Content-Length") or 0)
            r.close()
            return total
    except Exception as e:  # noqa: BLE001
        return -1


# Sort key by wavelength: the row's band limits when MAST gives them, else the number in the filter name (F160W -> 160).
def wave_key(name, r):
    lo, hi = r.get("em_min"), r.get("em_max")
    if lo is not None and hi is not None:
        return (lo + hi) / 2
    m = re.search(r"(\d+)", name)
    return float(m.group(1)) * 10 if m else 0.0


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    lines = []
    for spec in sys.argv[2:]:
        collection, proposal = spec.split("|")
        print(f"== {spec}", flush=True)
        try:
            rows = find_rows.ask([
                {"paramName": "obs_collection", "values": [collection]},
                {"paramName": "proposal_id", "values": [proposal]},
                {"paramName": "intentType", "values": ["science"]},
                {"paramName": "dataRights", "values": ["PUBLIC"]},
                {"paramName": "calib_level", "values": [3]},
            ])
        except Exception as e:  # noqa: BLE001
            lines.append(f"{spec}: query failed: {e}")
            continue
        rows = [r for r in rows if r.get("dataproduct_type") == "image" and (r.get("dataURL") or "").endswith(".fits") and r.get("filters") and ";" not in r["filters"]]
        if not rows:
            lines.append(f"{spec}: no public level-3 single-filter image mosaics")
            continue
        groups = collections.defaultdict(dict)
        for r in rows:
            key = (r["target_name"], r["instrument_name"])
            f = r["filters"]
            if f not in groups[key] or (r.get("t_exptime") or 0) > (groups[key][f].get("t_exptime") or 0):
                groups[key][f] = r
        for (target, inst), flt in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            if len(flt) < 3:
                continue
            parts, total = [], 0
            for f, r in sorted(flt.items(), key=lambda kv: wave_key(kv[0], kv[1])):
                sz = size_of(r["dataURL"])
                total += max(sz, 0)
                parts.append(f"{f} {sz / 1e9:.2f} GB" if sz >= 0 else f"{f} ?")
            lines.append(f"{spec} | {target} | {inst} | {len(flt)} filters | total {total / 1e9:.1f} GB | {', '.join(parts)}")
            print(lines[-1], flush=True)
            open(os.path.join(out, "report.txt"), "w").write("\n".join(lines) + "\n")
    open(os.path.join(out, "report.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


main()
