import ast
import hashlib
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import yaml
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path("/tmp/plato-p2-generated-fix")
SOURCE = Path("/tmp/plato-refresh-worktrees/docs-package-refresh").resolve()
WORKFLOW = Path("/tmp/plato-refresh-worktrees/docs-package-fix-workflow").resolve()
RELEASE = Path("/tmp/plato-refresh-worktrees/docs-package-fix-release").resolve()
NEGATIVE = Path("/tmp/plato-refresh-worktrees/docs-package-fix-negative").resolve()
COMMIT = "f1963eb906228c287f9787251f4319339ca7ee78"
PYTHON = sys.executable
UV = shutil.which("uv")
BUILD_PYTHON = subprocess.check_output([UV, "python", "find", "3.13"], text=True).strip()
CHECKER = SOURCE / ".github/scripts/check_distribution.py"
spec = importlib.util.spec_from_file_location("checker", CHECKER)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
project = tomllib.loads((SOURCE / "pyproject.toml").read_text())
sources = set(subprocess.check_output(["git", "ls-files", "plato/**/*.py", "plato/*.py"],
                                     cwd=SOURCE, text=True).splitlines())
scratch = ROOT / "negative"
scratch.mkdir(parents=True, exist_ok=True)
negative_results = []

def command(label, arguments, cwd=SOURCE, extra_environment=None, expected=0, path_prefix=None):
    script = shlex.join(arguments)
    if path_prefix:
        script = "PATH=" + shlex.quote(str(path_prefix) + ":" + os.environ["PATH"]) + " " + script
    environment = os.environ.copy()
    environment.update(extra_environment or {})
    result = subprocess.run(["zsh", "-lc", script], cwd=cwd, env=environment,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log = scratch / (label + ".log")
    log.write_text(result.stdout)
    assert result.returncode == expected, (label, result.returncode, result.stdout)
    return {"name": label, "exit_code": result.returncode, "command": arguments,
            "cwd": str(cwd), "path_prefix": str(path_prefix) if path_prefix else None,
            "log": str(log), "accepted": True}

def reject(label, action, expected):
    try:
        action()
    except (ValueError, RuntimeError) as error:
        assert expected in str(error), (label, error)
        negative_results.append({"name": label, "accepted": True, "rejected": True,
                                 "reason": str(error)})
    else:
        raise AssertionError("Negative accepted: " + label)

def normalize(text):
    value = Requirement(text)
    return (canonicalize_name(value.name), tuple(sorted(value.extras)), str(value.specifier),
            str(value.marker) if value.marker else None)

build_results = {}
for label, checkout in (("workflow_default_after_docs_and_dirty_outputs", WORKFLOW),
                         ("release_default_without_docs", RELEASE)):
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip() == COMMIT
    package = checkout / "ci-artifacts/docs-package"
    receipt = json.loads((package / "acceptance.json").read_text())
    assert receipt["accepted"] and receipt["source_commit"] == COMMIT
    wheel = next((package / "distributions").glob("*.whl"))
    sdist = next((package / "distributions").glob("*.tar.gz"))
    manifest = json.loads((package / "sdist-manifest.json").read_text())
    with tarfile.open(sdist) as archive:
        actual_members = {}
        for member in archive.getmembers():
            if member.isfile():
                relative = "/".join(member.name.split("/")[1:])
                actual_members[relative] = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
    assert actual_members == manifest["files"]
    generated_docs = [name for name in actual_members if name.startswith("docs/site/")]
    transient_ci = [name for name in actual_members if name.startswith("ci-artifacts/")]
    forbidden = [name for name in actual_members if module.forbidden_payload(name)]
    assert generated_docs == transient_ci == forbidden == []
    assert sources <= actual_members.keys()
    for name in project["tool"]["uv"]["workspace"]["members"]:
        assert name + "/pyproject.toml" in actual_members, name
    with zipfile.ZipFile(wheel) as archive:
        metadata = BytesParser().parsebytes(archive.read(next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )))
    expected = set(map(normalize, project["project"]["dependencies"]))
    for extra, dependencies in project["project"]["optional-dependencies"].items():
        for dependency in dependencies:
            expected.add(normalize(dependency + '; extra == "' + canonicalize_name(extra) + '"'))
    assert set(map(normalize, metadata.get_all("Requires-Dist"))) == expected
    assert set(metadata.get_all("Provides-Extra")) == set(map(
        canonicalize_name, project["project"]["optional-dependencies"]
    ))
    assert metadata["Requires-Python"] == project["project"]["requires-python"]
    installed = json.loads((package / "installed-wheel.json").read_text())
    assert Path(unquote(urlsplit(installed["direct_url"]["url"]).path)).resolve() == wheel.resolve()
    assert not Path(installed["cwd"]).resolve().is_relative_to(checkout)
    assert all(Path(location).resolve().is_relative_to(Path(installed["prefix"]).resolve())
               for location in installed["imports"].values())
    assert installed["installed_payload_matches_wheel"]
    assert installed["cpu_tensor_and_torchvision_operation"]
    rebuilt = json.loads((package / "rebuilt-wheel-manifest.json").read_text())
    original = json.loads((package / "wheel-manifest.json").read_text())
    assert original["files"] == rebuilt["files"] and original["metadata"] == rebuilt["metadata"]
    build_results[label] = {
        "source_commit": COMMIT, "checkout": str(checkout),
        "command": [BUILD_PYTHON, ".github/scripts/check_distribution.py"],
        "default_output_path": "ci-artifacts/docs-package",
        "accepted": True, "sdist_sha256": module.sha256(sdist),
        "wheel_sha256": module.sha256(wheel), "sdist_member_count": len(actual_members),
        "generated_docs_members": generated_docs, "transient_ci_members": transient_ci,
        "forbidden_members": forbidden, "tracked_plato_source_count": len(sources),
        "retained_workspace_manifest_count": len(project["tool"]["uv"]["workspace"]["members"]),
        "required_and_optional_metadata_parity": True,
        "installed_direct_url_and_outside_imports": True,
        "clean_from_sdist_rebuild_matching_payload_and_metadata": True,
        "tracked_source_status": subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=checkout, text=True),
        "all_untracked_status": subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"], cwd=checkout, text=True),
    }
assert build_results["workflow_default_after_docs_and_dirty_outputs"]["sdist_sha256"] == build_results["release_default_without_docs"]["sdist_sha256"]
assert (WORKFLOW / "docs/site/p2-stale-output.json").is_file()
assert (WORKFLOW / "ci-artifacts/stale-run/probe.json").is_file()
assert not (RELEASE / "docs/site").exists()
(ROOT / "default-path-regression.json").write_text(json.dumps({
    "source_commit": COMMIT, "accepted": True,
    "dirty_generated_sentinels_remain_in_checkout_and_are_absent_from_archive": True,
    "workflow_and_release_sdist_sha256_identical": True,
    "builds": build_results,
}, indent=2) + "\n")

# Re-export with exactly the generation flags used by the shared docs helper.
requirements_output = ROOT / "requirements-parity.txt"
command("requirements-export", [UV, "export", "--locked", "--python", "3.13",
        "--only-group", "docs", "--no-emit-project", "--no-header",
        "--output-file", str(requirements_output)], WORKFLOW)
assert requirements_output.read_bytes() == (WORKFLOW / "docs/requirements.txt").read_bytes()
docs = json.loads((WORKFLOW / "ci-artifacts/docs/docs-environment.json").read_text())
versions = {package["name"]: package["version"]
            for package in tomllib.loads((SOURCE / "uv.lock").read_text())["package"]}
assert all(name not in docs["packages"] for name in ("torch", "lighteval", "plato-learn"))
assert all(docs["packages"][name] == versions[name] for name in ("mkdocs", "mkdocs-material"))
(ROOT / "docs-parity.json").write_text(json.dumps({
    "source_commit": COMMIT, "requirements_byte_parity": True,
    "requirements_sha256": module.sha256(requirements_output),
    "docs_dependency_count": len(docs["packages"]), "runtime_and_project_absent": True,
    "mkdocs": docs["packages"]["mkdocs"], "mkdocs-material": docs["packages"]["mkdocs-material"],
}, indent=2) + "\n")

wheel = next((WORKFLOW / "ci-artifacts/docs-package/distributions").glob("*.whl"))
with zipfile.ZipFile(wheel) as archive:
    wheel_files = {name: archive.read(name) for name in archive.namelist() if not name.endswith("/")}
metadata_name = next(name for name in wheel_files if name.endswith(".dist-info/METADATA"))
for label, member in (
    ("archive_member", "archives/legacy/module.py"),
    ("reproduction_member", "evidence/2026-refresh/reproductions/probe.json"),
    ("cache_member", "__pycache__/probe.pyc"),
    ("environment_member", ".venv-docs/lib/probe.py"),
    ("generated_docs_site_member", "docs/site/index.html"),
    ("generated_ci_artifacts_member", "ci-artifacts/docs-package/provenance.json"),
):
    folder = scratch / label
    folder.mkdir()
    mutation = folder / wheel.name
    with zipfile.ZipFile(mutation, "w") as archive:
        for name, payload in (wheel_files | {member: b"P2-M1-negative-sentinel"}).items():
            archive.writestr(name, payload)
    reject(label, lambda: module.inspect_distribution(mutation, project["project"], sources),
           "Forbidden distribution payloads")
    negative_results[-1]["artifact_sha256"] = module.sha256(mutation)
for label, removed, replacement, expected in (
    ("missing_plato_source", {"plato/config.py"}, {}, "Missing Plato source files"),
    ("wrong_version_metadata", set(), {"Version: 1.4.3": "Version: 0.0.0"}, "Version:"),
    ("wrong_python_metadata", set(), {"Requires-Python: >=3.13": "Requires-Python: >=3.14"}, "Requires-Python:"),
):
    folder = scratch / label
    folder.mkdir()
    mutation = folder / wheel.name
    with zipfile.ZipFile(mutation, "w") as archive:
        for name, payload in wheel_files.items():
            if name in removed:
                continue
            if name == metadata_name:
                for before, after in replacement.items():
                    assert before.encode() in payload
                    payload = payload.replace(before.encode(), after.encode())
            archive.writestr(name, payload)
    reject(label, lambda: module.inspect_distribution(mutation, project["project"], sources), expected)
for label, red in (
    ("actual_review_default_contamination", Path("/tmp/plato-p2-independent-review/workflow-source/ci-artifacts/docs-package/distributions/plato_learn-1.4.3.tar.gz")),
    ("actual_prior_author_docs_contamination", Path("/tmp/plato-p2-final/package/distributions/plato_learn-1.4.3.tar.gz")),
    ("actual_pre_fix_archive_payload", Path("/tmp/plato-p2-preliminary/package-second/distributions/plato_learn-1.4.3.tar.gz")),
):
    reject(label, lambda: module.inspect_distribution(red, project["project"], sources),
           "Forbidden distribution payloads")
    negative_results[-1]["artifact"] = str(red)
    negative_results[-1]["artifact_sha256"] = module.sha256(red)

existing = scratch / "existing-output"
existing.mkdir()
(existing / "acceptance.json").write_text("preserve-existing-qualification\n")
negative_results.append(command("existing-output-rejected", [PYTHON, str(CHECKER),
    "--output-dir", str(existing)], expected=1))
assert (existing / "acceptance.json").read_text() == "preserve-existing-qualification\n"
unsafe_output = SOURCE / "arbitrary-generated-output"
negative_results.append(command("unexcluded-checkout-output-rejected", [PYTHON, str(CHECKER),
    "--output-dir", str(unsafe_output)], expected=1))
assert not unsafe_output.exists()
negative_results.append(command("repository-cwd-rejected", [PYTHON, "-I", "-c",
    module.IMPORT_PROBE, str(SOURCE),
    str(WORKFLOW / "ci-artifacts/docs-package/wheel-manifest.json"),
    project["project"]["version"], str(scratch / "unexpected-import-receipt.json")],
    expected=1))
assert not (scratch / "unexpected-import-receipt.json").exists()
docs_environment = {"PLATO_DOCS_PYTHON": PYTHON,
                    "PLATO_DOCS_ENVIRONMENT": str(ROOT / "negative-docs-environment/.venv-docs")}
negative_results.append(command("application-environment-rejected", ["./docs/build.sh"], NEGATIVE,
    docs_environment | {"PLATO_DOCS_ENVIRONMENT": str(NEGATIVE / ".venv")}, expected=1))
assert not (NEGATIVE / ".venv").exists()
requirements = NEGATIVE / "docs/requirements.txt"
original_requirements = requirements.read_bytes()
try:
    requirements.write_bytes(original_requirements + b"# Deliberate generated requirements mismatch\n")
    negative_results.append(command("requirements-parity-rejected", ["./docs/build.sh"], NEGATIVE,
                                    docs_environment, expected=1))
finally:
    requirements.write_bytes(original_requirements)
assert not Path(docs_environment["PLATO_DOCS_ENVIRONMENT"]).exists()

stubs = scratch / "failure-stubs"
stubs.mkdir()
wrapper = stubs / "uv"
wrapper.write_text("#!" + PYTHON + "\nimport os, sys\n"
                  "if sys.argv[1:] == ['--version']: print('uv 0.12.22'); sys.exit(0)\n"
                  "if sys.argv[1] == os.environ.get('P2_FAIL_COMMAND'): sys.exit(int(os.environ['P2_FAIL_CODE']))\n"
                  "os.execv(" + repr(UV) + ", [" + repr(UV) + "] + sys.argv[1:])\n")
wrapper.chmod(0o755)
for label, verb, code in (("docs-sync-failure", "sync", 37), ("strict-docs-build-failure", "run", 41)):
    negative_results.append(command(label, ["./docs/build.sh"], NEGATIVE,
        docs_environment | {"P2_FAIL_COMMAND": verb, "P2_FAIL_CODE": str(code)},
        expected=code, path_prefix=stubs))
failed_output = NEGATIVE / "ci-artifacts/failure-propagation"
negative_results.append(command("package-build-failure", [PYTHON, str(NEGATIVE / ".github/scripts/check_distribution.py"),
    "--output-dir", str(failed_output)], NEGATIVE,
    {"P2_FAIL_COMMAND": "build", "P2_FAIL_CODE": "43"}, expected=1, path_prefix=stubs))
failure_receipt = json.loads((failed_output / "acceptance.json").read_text())
assert failure_receipt["accepted"] is False and "43" in failure_receipt["error"]
reject("package-subprocess-failure", lambda: module.run([PYTHON, "-c", "raise SystemExit(29)"],
       ROOT, scratch / "package-subprocess-failure.log"), "29")

wrong_uv = scratch / "wrong-uv"
wrong_uv.mkdir()
(wrong_uv / "uv").write_text("#!/bin/sh\nprintf 'uv 0.12.21\\n'\n")
(wrong_uv / "uv").chmod(0o755)
wrong_output = scratch / "wrong-uv-output"
negative_results.append(command("wrong-uv-version-rejected", [PYTHON, str(CHECKER),
    "--output-dir", str(wrong_output)], expected=1, path_prefix=wrong_uv))
assert not wrong_output.exists()
(ROOT / "negative-checks.json").write_text(json.dumps({
    "source_commit": COMMIT, "checker_sha256": module.sha256(CHECKER),
    "accepted": True, "results": negative_results,
}, indent=2) + "\n")

for filename in (".github/workflows/docs_package_checks.yml", ".github/workflows/pypi_publish.yml"):
    document = yaml.load((SOURCE / filename).read_text(), Loader=yaml.BaseLoader)
    assert document["permissions"] == {"contents": "read"}
    for job in document["jobs"].values():
        assert job["defaults"]["run"]["shell"] == 'zsh -lc ". {0}"'
        for step in job["steps"]:
            if "uses" in step:
                assert len(step["uses"].split("@")[1]) == 40
            if "run" in step:
                if step.get("shell") == "python {0}":
                    ast.parse(step["run"])
                else:
                    with tempfile.NamedTemporaryFile(mode="w", suffix=".zsh") as handle:
                        handle.write(step["run"])
                        handle.flush()
                        subprocess.run(["zsh", "-n", handle.name], check=True)
    if filename.endswith("pypi_publish.yml"):
        assert document["on"] == {"release": {"types": ["created"]}}
configuration = tomllib.loads((SOURCE / "netlify.toml").read_text())
assert configuration["build"] == {"base": ".", "command": "./docs/build.sh", "publish": "docs/site",
                                  "environment": {"PYTHON_VERSION": "3.13"}}
(ROOT / "syntax-checks.json").write_text(json.dumps({
    "source_commit": COMMIT, "accepted": True,
    "workflow_yaml_and_embedded_shell_python_commands": True, "netlify_toml": True,
}, indent=2) + "\n")
print(json.dumps({"accepted": True, "negative_checks": len(negative_results),
                  "sdist_sha256": build_results["release_default_without_docs"]["sdist_sha256"],
                  "source_commit": COMMIT}))
