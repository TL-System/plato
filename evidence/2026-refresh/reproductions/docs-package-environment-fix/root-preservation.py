import hashlib
import json
import os
import sys
from pathlib import Path

phase = sys.argv[1]
root = Path("/tmp/plato-refresh-worktrees/deployment-integration")
finding = Path("/tmp/plato-deployment-integration/docs-environment-package-finding.json")
targets = [root / ".venv-docs", root / "ci-artifacts", finding]
targets.extend(Path("/tmp/plato-deployment-integration").glob("publication-*.log"))
records = {}
for target in targets:
    paths = [target] + (sorted(target.rglob("*")) if target.is_dir() else [])
    for path in paths:
        status = path.lstat()
        item = {"mode": status.st_mode, "size": status.st_size,
                "mtime_ns": status.st_mtime_ns}
        if path.is_symlink():
            item["link_target"] = os.readlink(path)
        elif path.is_file():
            item["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        records[str(path)] = item
payload = {"root_source": "0c942d7e81171e01bb715c0ba7b82f08900ea56a",
           "targets": list(map(str, targets)), "snapshot_entries": len(records),
           "snapshot_sha256": hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()}
destination = Path("/tmp/plato-p2-docs-environment-fix")
if phase == "before":
    (destination / "root-preservation-before.json").write_text(json.dumps(payload, indent=2) + "\n")
else:
    original = json.loads((destination / "root-preservation-before.json").read_text())
    assert original == payload, (original, payload)
    payload["unchanged_bytes_symlinks_modes_and_mtimes"] = True
    (destination / "root-preservation-after.json").write_text(json.dumps(payload, indent=2) + "\n")
