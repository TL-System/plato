import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

root = Path("/tmp/plato-p2-generated-fix")
checkout = Path("/tmp/plato-refresh-worktrees/docs-package-fix-negative").resolve()
configuration = checkout / "docs/mkdocs.yml"
original = configuration.read_bytes()
fixture = original + b"\n  - P2 deliberately missing page: p2_missing_page.md\n"
environment = os.environ | {
    "PLATO_DOCS_PYTHON": sys.executable,
    "PLATO_DOCS_ENVIRONMENT": str(root / "negative-docs-environment/.venv-docs"),
}
try:
    configuration.write_bytes(fixture)
    process = subprocess.run(["zsh", "-lc", "./docs/build.sh"], cwd=checkout, env=environment,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (root / "negative/actual-strict-mkdocs-failure.log").write_text(process.stdout)
    assert process.returncode == 1, (process.returncode, process.stdout)
    assert "p2_missing_page.md" in process.stdout and "Aborted with" in process.stdout, process.stdout
finally:
    configuration.write_bytes(original)
receipt_path = root / "negative-checks.json"
receipt = json.loads(receipt_path.read_text())
receipt["results"].append({
    "name": "actual-strict-mkdocs-missing-nav-page",
    "accepted": True,
    "exit_code": process.returncode,
    "command": ["zsh", "-lc", "./docs/build.sh"],
    "cwd": str(checkout),
    "fixture_mkdocs_sha256": hashlib.sha256(fixture).hexdigest(),
    "original_mkdocs_sha256": hashlib.sha256(original).hexdigest(),
    "fixture_restored": True,
    "real_mkdocs_warning_propagated_through_shared_helper": True,
})
receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
assert subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=checkout, text=True) == ""
print(json.dumps({"accepted": True, "negative_checks": len(receipt["results"]),
                  "real_strict_mkdocs_failure_exit_code": process.returncode}))
