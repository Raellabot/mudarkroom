# Lists the Orion Treasury data directory that MAST's HLSP page gives (archive.stsci.edu/pub/hlsp/orion/) to find the ACS mosaic file names.
import re, urllib.parse, urllib.request

ROOT = "https://archive.stsci.edu/pub/hlsp/orion/"


def listing(url):
    req = urllib.request.Request(url, headers={"User-Agent": "cosmic-darkroom builder"})
    html = urllib.request.urlopen(req, timeout=60).read().decode("utf8", "replace")
    return [urllib.parse.urljoin(url, h) for h in re.findall(r'href="([^"?#]+)"', html)]


seen, todo = set(), [ROOT]
while todo and len(seen) < 60:
    d = todo.pop(0)
    if d in seen:
        continue
    seen.add(d)
    try:
        links = [l for l in listing(d) if l.startswith(d) and l != d]
    except Exception as e:
        print("FAIL", d, repr(e)[:200], flush=True)
        continue
    files = [l for l in links if not l.endswith("/")]
    print("DIR", d, len(files), "files", [l[len(d):] for l in files if "acs" in l.lower()][:12] or [l[len(d):] for l in files][:12], flush=True)
    todo += [l for l in links if l.endswith("/")]
