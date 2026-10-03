set -eu
python=/tmp/plato-p2-source-binding-fix/controller-env/bin/python
"$python" - <<'PY'
import hashlib
import json
import subprocess
from pathlib import Path
root = Path.cwd().resolve()
external = Path("/tmp/plato-p2-source-binding-fix")
existing = external / "existing-external-output"
existing.mkdir()
(existing / "sentinel.txt").write_text("Existing external output belongs to its caller.\n")
records = []
def snapshot(directory):
    result = {}
    for path in [directory] + sorted(directory.rglob("*")):
        info = path.lstat()
        record = {"mode": info.st_mode, "size": info.st_size, "mtime_ns": info.st_mtime_ns}
        if path.is_file():
            record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        result[str(path.relative_to(directory))] = record
    return result
for label, output in (
    ("existing-unsupported-tracked-directory", root / "docs"),
    ("existing-external-directory", existing),
    ("checkout-root", root),
):
    before = snapshot(output)
    command = ["/tmp/plato-p2-source-binding-fix/controller-env/bin/python",
               ".github/scripts/check_distribution.py", "--output-dir", str(output)]
    process = subprocess.run(command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (external / (label + ".log")).write_text(process.stdout)
    after = snapshot(output)
    assert process.returncode == 1 and before == after, (label, process.stdout)
    assert not (output / "acceptance.json").exists(), label
    records.append({"name": label, "exit_code": process.returncode, "command": command,
                    "directory_unchanged": True, "snapshot_entries": len(before),
                    "snapshot_sha256": hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest()})
(external / "ownership-rejections.json").write_text(json.dumps({
    "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "accepted": True, "results": records,
}, indent=2) + "\n")
PY
mkdir -p ci-artifacts/docs
git rev-parse HEAD > ci-artifacts/docs/commit.txt
shasum -a 256 uv.lock netlify.toml docs/build.sh docs/requirements.txt > ci-artifacts/docs/source-hashes.txt
"$python" --version > ci-artifacts/docs/python.txt
uv --version > ci-artifacts/docs/uv.txt
PLATO_DOCS_PYTHON="$python" PLATO_DOCS_ENVIRONMENT=/tmp/plato-p2-source-binding-fix/docs-environment/.venv-docs PLATO_DOCS_ARTIFACT_DIR="$PWD/ci-artifacts/docs" ./docs/build.sh > ci-artifacts/docs/build.log 2>&1
"$python" - <<'PY'
from pathlib import Path
for filename in ("docs/site/p2-stale-site.json", "ci-artifacts/older/p2-stale-artifact.json"):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"generated": "P2-M2-dirty-default"}\n')
PY
"$python" .github/scripts/check_distribution.py > /tmp/plato-p2-source-binding-fix/workflow-default-checker.log 2>&1
cat /tmp/plato-p2-source-binding-fix/workflow-default-checker.log
