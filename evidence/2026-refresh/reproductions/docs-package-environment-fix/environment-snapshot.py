import hashlib
import json
import os
import sys
from pathlib import Path

phase, label = sys.argv[1:]
root = Path.cwd().resolve()
records = {}
for name in (".venv-docs", ".venv-docs-bootstrap"):
    directory = root / name
    if not directory.exists():
        continue
    tree = {}
    for path in [directory] + sorted(directory.rglob("*")):
        status = path.lstat()
        item = {"mode": status.st_mode, "size": status.st_size,
                "mtime_ns": status.st_mtime_ns}
        if path.is_symlink():
            item["link_target"] = os.readlink(path)
        elif path.is_file():
            item["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        tree[path.relative_to(directory).as_posix()] = item
    records[name] = {"entries": len(tree),
                     "snapshot_sha256": hashlib.sha256(json.dumps(tree, sort_keys=True).encode()).hexdigest(),
                     "python_link": os.readlink(directory / "bin/python"),
                     "sentinels": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in directory.glob("p2-m3-*.json")}}
destination = Path("/tmp/plato-p2-docs-environment-fix")
payload = {"source_commit": "06ba679690f090aacf7e6a95a6dffd74b81ffad3",
           "checkout": str(root), "environments": records}
if phase == "before":
    (destination / (label + "-environment-before.json")).write_text(json.dumps(payload, indent=2) + "\n")
else:
    original = json.loads((destination / (label + "-environment-before.json")).read_text())
    assert original == payload, (label, original, payload)
    payload["untouched_by_package_checker"] = True
    (destination / (label + "-environment-after.json")).write_text(json.dumps(payload, indent=2) + "\n")
