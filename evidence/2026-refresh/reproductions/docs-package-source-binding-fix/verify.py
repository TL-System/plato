import ast
import hashlib
import json
import subprocess
import tempfile
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import yaml
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path("/tmp/plato-p2-source-binding-fix")
SOURCE = Path("/tmp/plato-refresh-worktrees/docs-package-refresh").resolve()
COMMIT = "63f0d2000f0370708b9afe1e2e6a2dc8559b72e1"
project = tomllib.loads((SOURCE / "pyproject.toml").read_text())
results = {}
def normalize(text):
    parsed = Requirement(text)
    return (canonicalize_name(parsed.name), tuple(sorted(parsed.extras)),
            str(parsed.specifier), str(parsed.marker) if parsed.marker else None)
for label, path in (
    ("workflow", Path("/tmp/plato-refresh-worktrees/docs-package-binding-workflow")),
    ("release", Path("/tmp/plato-refresh-worktrees/docs-package-binding-release")),
):
    package = path / "ci-artifacts/docs-package"
    accepted = json.loads((package / "acceptance.json").read_text())
    binding = json.loads((package / "sdist-source-binding.json").read_text())
    sdist = json.loads((package / "sdist-manifest.json").read_text())
    wheel = json.loads((package / "wheel-manifest.json").read_text())
    assert accepted["source_commit"] == binding["source_commit"] == COMMIT
    assert accepted["sdist_source_tree_bound"] and accepted["accepted"]
    assert accepted["sdist_source_binding_sha256"] == hashlib.sha256((package / "sdist-source-binding.json").read_bytes()).hexdigest()
    assert len(binding["files"]) == binding["git_tree_bound_members"] == len(sdist["files"]) - 1
    assert set(binding["files"]) == set(sdist["files"]) - {"PKG-INFO"}
    assert all(record["sha256"] == sdist["files"][name] for name, record in binding["files"].items())
    assert all(not name.startswith(("docs/site/", "ci-artifacts/")) for name in sdist["files"])
    with zipfile.ZipFile(package / "distributions" / wheel["filename"]) as archive:
        metadata = BytesParser().parsebytes(archive.read(next(name for name in archive.namelist()
                                                              if name.endswith(".dist-info/METADATA"))))
    declarations = set(map(normalize, project["project"]["dependencies"]))
    for extra, dependencies in project["project"]["optional-dependencies"].items():
        for dependency in dependencies:
            declarations.add(normalize(dependency + '; extra == "' + canonicalize_name(extra) + '"'))
    assert set(map(normalize, metadata.get_all("Requires-Dist"))) == declarations
    assert set(metadata.get_all("Provides-Extra")) == set(map(canonicalize_name, project["project"]["optional-dependencies"]))
    assert metadata["Requires-Python"] == project["project"]["requires-python"]
    installed = json.loads((package / "installed-wheel.json").read_text())
    wheel_path = (package / "distributions" / wheel["filename"]).resolve()
    assert Path(unquote(urlsplit(installed["direct_url"]["url"]).path)).resolve() == wheel_path
    assert installed["installed_payload_matches_wheel"] and installed["cpu_tensor_and_torchvision_operation"]
    assert not Path(installed["cwd"]).resolve().is_relative_to(path.resolve())
    assert all(Path(location).resolve().is_relative_to(Path(installed["prefix"]).resolve())
               for location in installed["imports"].values())
    rebuilt = json.loads((package / "rebuilt-wheel-manifest.json").read_text())
    assert wheel["metadata"] == rebuilt["metadata"] and wheel["files"] == rebuilt["files"]
    for workspace in project["tool"]["uv"]["workspace"]["members"]:
        assert workspace + "/pyproject.toml" in sdist["files"]
    results[label] = {
        "source_commit": COMMIT, "accepted": True, "checkout": str(path),
        "command": ["/tmp/plato-p2-source-binding-fix/controller-env/bin/python", ".github/scripts/check_distribution.py"],
        "sdist_sha256": accepted["sdist_sha256"], "wheel_sha256": accepted["wheel_sha256"],
        "git_tree_bound_members": binding["git_tree_bound_members"],
        "only_generated_exception": binding["generated_metadata"],
        "requires_dist_extras_and_requires_python_parity": True,
        "clean_rebuild_and_installed_wheel_proof": True,
    }
assert results["workflow"]["sdist_sha256"] == results["release"]["sdist_sha256"]
assert results["workflow"]["wheel_sha256"] == results["release"]["wheel_sha256"]
docs = json.loads(Path("/tmp/plato-refresh-worktrees/docs-package-binding-workflow/ci-artifacts/docs/docs-environment.json").read_text())
assert docs["requirements_parity"] and docs["runtime_and_project_absent"]
assert all(name not in docs["packages"] for name in ("torch", "lighteval", "plato-learn"))
exports = ROOT / "requirements-parity.txt"
subprocess.run(["uv", "export", "--locked", "--python", "3.13", "--only-group", "docs",
                "--no-emit-project", "--no-header", "--output-file", str(exports)], cwd=SOURCE,
               stdout=subprocess.DEVNULL, check=True)
assert exports.read_bytes() == (SOURCE / "docs/requirements.txt").read_bytes()
for filename in (".github/workflows/docs_package_checks.yml", ".github/workflows/pypi_publish.yml"):
    workflow = yaml.load((SOURCE / filename).read_text(), Loader=yaml.BaseLoader)
    assert workflow["permissions"] == {"contents": "read"}
    for job in workflow["jobs"].values():
        assert job["defaults"]["run"]["shell"] == 'zsh -lc ". {0}"'
        for step in job["steps"]:
            if "run" in step:
                if step.get("shell") == "python {0}":
                    ast.parse(step["run"])
                else:
                    with tempfile.NamedTemporaryFile(mode="w", suffix=".zsh") as script:
                        script.write(step["run"])
                        script.flush()
                        subprocess.run(["zsh", "-n", script.name], check=True)
    if filename.endswith("pypi_publish.yml"):
        assert workflow["on"] == {"release": {"types": ["created"]}}
configuration = tomllib.loads((SOURCE / "netlify.toml").read_text())
assert configuration["build"] == {"base": ".", "command": "./docs/build.sh", "publish": "docs/site",
                                  "environment": {"PYTHON_VERSION": "3.13"}}
(ROOT / "default-validation.json").write_text(json.dumps({
    "source_commit": COMMIT, "accepted": True, "builds": results,
    "fresh_docs_requirement_parity": True, "docs_dependency_count": len(docs["packages"]),
    "workflow_and_netlify_syntax_and_publication_boundary": True,
}, indent=2) + "\n")
print(json.dumps({"accepted": True, "source_commit": COMMIT,
                  "sdist_sha256": results["workflow"]["sdist_sha256"],
                  "git_tree_bound_members": results["workflow"]["git_tree_bound_members"]}))
