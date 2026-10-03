#!/bin/sh
# Build from the root lockfile without installing the application or dev groups.
set -eu

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_dir"
docs_python=${PLATO_DOCS_PYTHON:-python3}
"$docs_python" -c 'import sys; assert sys.version_info[:2] == (3, 13), sys.version'

if command -v uv >/dev/null 2>&1 && [ "$(uv --version | cut -d ' ' -f 2)" = "0.12.22" ]; then
    docs_uv=$(command -v uv)
else
    bootstrap_dir=${PLATO_DOCS_BOOTSTRAP_ENVIRONMENT:-$repo_dir/.venv-docs-bootstrap}
    "$docs_python" -m venv "$bootstrap_dir"
    "$bootstrap_dir/bin/python" -m pip install --disable-pip-version-check 'uv==0.12.22'
    docs_uv=$bootstrap_dir/bin/uv
fi

export UV_PROJECT_ENVIRONMENT=${PLATO_DOCS_ENVIRONMENT:-$repo_dir/.venv-docs}
"$docs_python" - "$UV_PROJECT_ENVIRONMENT" "$repo_dir" <<'PY'
import sys
from pathlib import Path
environment, repository = map(Path, sys.argv[1:])
assert environment.resolve() != (repository / ".venv").resolve(), environment
PY

export_dir=$(mktemp -d)
trap 'rm -rf "$export_dir"' EXIT HUP INT TERM
"$docs_uv" export --locked --python 3.13 --only-group docs --no-emit-project \
    --no-header --output-file "$export_dir/requirements.txt" >/dev/null
cmp docs/requirements.txt "$export_dir/requirements.txt"
"$docs_uv" sync --locked --python 3.13 --only-group docs
"$docs_uv" pip check --python "$UV_PROJECT_ENVIRONMENT/bin/python"
"$UV_PROJECT_ENVIRONMENT/bin/python" - "$repo_dir" <<'PY'
import importlib.metadata
import json
import os
import sys
import tomllib
from pathlib import Path

assert sys.version_info[:2] == (3, 13), sys.version
repository = Path(sys.argv[1])
locked = tomllib.loads((repository / "uv.lock").read_text())
versions = {package["name"]: package["version"] for package in locked["package"]}
installed = {
    distribution.metadata["Name"].lower().replace("_", "-"): distribution.version
    for distribution in importlib.metadata.distributions()
}
for name in ("torch", "lighteval", "plato-learn"):
    assert name not in installed, (name, installed)
for name in ("mkdocs", "mkdocs-material"):
    assert installed[name] == versions[name], (name, installed[name], versions[name])
receipt = {
    "python": sys.version,
    "executable": sys.executable,
    "prefix": sys.prefix,
    "packages": dict(sorted(installed.items())),
    "requirements_parity": True,
    "runtime_and_project_absent": True,
}
print(json.dumps(receipt, indent=2))
if destination := os.environ.get("PLATO_DOCS_ARTIFACT_DIR"):
    artifact_dir = Path(destination)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "docs-environment.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
PY
"$docs_uv" run --no-sync --python 3.13 mkdocs build --strict -f docs/mkdocs.yml
