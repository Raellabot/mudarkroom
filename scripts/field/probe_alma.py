# Lists what the ALMA archive offers for protoplanetary disks without downloading images; usage: probe_alma.py OUTDIR
import io, os, re, sys, urllib.parse, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from astropy.io.votable import parse_single_table
from chandra_events import UA, get_text

TAPS = ["https://almascience.eso.org/tap", "https://almascience.nrao.edu/tap"]
LP = "https://almascience.eso.org/almadata/lp/"
DISKS = ["HL Tau", "TW Hya", "HD 163296", "IM Lup", "AS 209", "GG Tau", "HD 142527", "Elias 2-27", "V883 Ori", "PDS 70", "HD 169142", "AB Aur", "MWC 480", "Oph IRS 48"]


# One ADQL query in VOTable form, tried on each server in turn; returns a list of dicts.
def tap(adql):
    last = None
    for base in TAPS:
        url = f"{base}/sync?REQUEST=doQuery&LANG=ADQL&FORMAT=votable&QUERY={urllib.parse.quote(adql)}"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=180) as r:
                body = r.read()
            t = parse_single_table(io.BytesIO(body)).to_table()
            return [{c: ("" if str(row[c]) == "--" else str(row[c])) for c in t.colnames} for row in t]
        except Exception as e:  # noqa: BLE001
            last = f"{base}: {str(e)[:200]}"
    raise RuntimeError(last)


def section(lines, title, fn):
    lines.append(f"== {title}")
    try:
        fn(lines)
    except Exception as e:  # noqa: BLE001
        lines.append(f"FAILED: {str(e)[:300]}")
    lines.append("")


# The large programs the archive hosts itself: the folders under almadata/lp and the FITS files in each.
def large_programs(lines):
    html = get_text(LP)
    dirs = sorted({m for m in re.findall(r'href="([^"?/][^"?]*/)"', html) if not m.startswith(("http", "/", ".."))})
    lines.append(f"{len(dirs)} folders under {LP}: {', '.join(d.strip('/') for d in dirs)}")
    for d in dirs:
        try:
            sub = get_text(LP + d)
        except Exception as e:  # noqa: BLE001
            lines.append(f"{d}: cannot list ({str(e)[:100]})")
            continue
        files = re.findall(r'href="([^"?/][^"?]*)"', sub)
        fits = [f for f in files if f.lower().endswith((".fits", ".fits.gz", ".fits.fz"))]
        subdirs = [f for f in files if f.endswith("/") and not f.startswith(("http", "/", ".."))]
        lines.append(f"{d}: {len(fits)} FITS files, {len(subdirs)} subfolders ({', '.join(s.strip('/') for s in subdirs[:8])}); examples: {', '.join(fits[:4])}")
        for s in subdirs[:3]:
            try:
                deeper = re.findall(r'href="([^"?/][^"?]*)"', get_text(LP + d + s))
                df = [f for f in deeper if f.lower().endswith((".fits", ".fits.gz", ".fits.fz"))]
                lines.append(f"    {d}{s}: {len(df)} FITS files; examples: {', '.join(df[:5])}")
            except Exception as e:  # noqa: BLE001
                lines.append(f"    {d}{s}: cannot list ({str(e)[:100]})")


def columns(lines):
    rows = tap("SELECT column_name, datatype FROM TAP_SCHEMA.columns WHERE table_name = 'ivoa.obscore'")
    lines.append(f"ivoa.obscore has {len(rows)} columns: " + ", ".join(r["column_name"] for r in rows))


# Public observations in the disk science category, one line per program and target.
def disk_programs(lines):
    rows = tap("SELECT TOP 20000 proposal_id, target_name, band_list, MIN(release_date) AS first_release, COUNT(*) AS n, MAX(access_estsize) AS max_kb "
               "FROM ivoa.obscore WHERE scientific_category LIKE '%isk%' AND data_rights = 'Public' "
               "GROUP BY proposal_id, target_name, band_list")
    lines.append(f"{len(rows)} program/target/band groups in the disk category with public data")
    by = {}
    for r in rows:
        by.setdefault(r["proposal_id"], []).append(r)
    top = sorted(by.items(), key=lambda kv: -sum(int(x["n"]) for x in kv[1]))[:40]
    for p, rs in top:
        lines.append(f"{p}: {len(rs)} targets, {sum(int(x['n']) for x in rs)} products; e.g. " + "; ".join(f"{x['target_name']} (band {x['band_list']})" for x in rs[:6]))


# The famous disks by name: what each has in the archive.
def named(lines):
    for name in DISKS:
        key = name.replace("'", "''")
        rows = tap("SELECT proposal_id, band_list, dataproduct_type, calib_level, data_rights, COUNT(*) AS n, MIN(s_resolution) AS best_res_arcsec, MAX(access_estsize) AS max_kb "
                   f"FROM ivoa.obscore WHERE target_name LIKE '%{key}%' GROUP BY proposal_id, band_list, dataproduct_type, calib_level, data_rights")
        lines.append(f"{name}: {len(rows)} groups")
        for r in rows[:12]:
            lines.append(f"    {r['proposal_id']} | band {r['band_list']} | {r['dataproduct_type']} | level {r['calib_level']} | {r['data_rights']} | {r['n']} products | best resolution {r['best_res_arcsec']} arcsec | largest {float(r['max_kb'] or 0) / 1e6:.2f} GB")


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    lines = ["ALMA probe"]
    for title, fn in (("Large programs hosted by the archive", large_programs), ("TAP columns", columns), ("Disk programs with public data", disk_programs), ("Famous disks by name", named)):
        section(lines, title, fn)
        open(os.path.join(out, "report.txt"), "w").write("\n".join(lines) + "\n")
        print("\n".join(lines[-3:]))
    print("done")


main()
