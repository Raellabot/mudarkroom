# Fails unless a target config has everything the page shows for it; usage: check_target.py targets/<id>.json
import json, sys
REQUIRED = {"id": str, "name": str, "blurb": str, "label": str, "spec": str, "find": list, "type": str, "aliases": list, "missions": list, "story": str}
cfg = json.load(open(sys.argv[1]))
# Planet and moon targets (Cassini, JunoCam) name their frames in "source" instead of a MAST "spec" and "find"
if isinstance(cfg.get("source"), dict) and cfg["source"].get("kind") in ("alma", "cassini", "junocam"):
    REQUIRED = {k: v for k, v in REQUIRED.items() if k not in ("spec", "find", "label")}
problems = [f"missing or empty: {k}" for k, t in REQUIRED.items() if not isinstance(cfg.get(k), t) or not cfg.get(k)]
if len(cfg.get("story", "")) < 40:
    problems.append("story should be a sentence or two on what you are looking at")
if problems:
    sys.exit(sys.argv[1] + ": " + "; ".join(problems))
print(sys.argv[1], "has everything the page needs")
