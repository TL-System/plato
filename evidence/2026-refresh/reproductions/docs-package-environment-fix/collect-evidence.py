import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/tmp/plato-p2-docs-environment-fix")
SOURCE = Path("/tmp/plato-refresh-worktrees/docs-package-refresh")
COMMIT = "06ba679690f090aacf7e6a95a6dffd74b81ffad3"
EVIDENCE = SOURCE / "evidence/2026-refresh/docs-package-environment-fix"
REPRODUCTION = SOURCE / "evidence/2026-refresh/reproductions/docs-package-environment-fix"
FINDING = Path("/tmp/plato-deployment-integration/docs-environment-package-finding.json")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True).strip() == COMMIT
assert not EVIDENCE.exists() and not REPRODUCTION.exists()
EVIDENCE.mkdir(parents=True)
REPRODUCTION.mkdir(parents=True)
for name in ("default-validation.json", "independent-source-bytes.json", "negative-checks.json",
             "bootstrap-inventory.json", "default-environment-before.json", "default-environment-after.json",
             "bootstrap-environment-before.json", "bootstrap-environment-after.json",
             "root-preservation-before.json", "root-preservation-after.json", "requirements-parity.txt",
             "default-docs.log", "bootstrap-docs.log", "default-package.log", "bootstrap-package.log",
             "release-package.log", "negative-execution.log", "verification.log"):
    copy(ROOT / name, EVIDENCE / name)
for name in ("default-command.zsh", "bootstrap-command.zsh", "release-command.zsh",
             "environment-snapshot.py", "root-preservation.py", "negative-checks.py", "verify.py",
             "collect-evidence.py"):
    copy(ROOT / name, REPRODUCTION / name)
for label in ("default", "bootstrap", "release"):
    checkout = Path("/tmp/plato-refresh-worktrees/docs-package-env-" + label)
    package = checkout / "ci-artifacts/docs-package"
    for path in sorted(package.iterdir()):
        if path.is_file():
            copy(path, EVIDENCE / label / "package" / path.name)
    if label != "release":
        for path in sorted((checkout / "ci-artifacts/docs").iterdir()):
            if path.is_file():
                copy(path, EVIDENCE / label / "docs" / path.name)
negative_checkout = Path("/tmp/plato-refresh-worktrees/docs-package-env-negative")
for folder in sorted((negative_checkout / "ci-artifacts").iterdir()):
    for path in sorted(folder.iterdir()):
        if path.is_file():
            copy(path, EVIDENCE / "negative" / folder.name / path.name)
for path in sorted((ROOT / "negative").glob("*.log")):
    copy(path, EVIDENCE / "negative" / "commands" / path.name)
finding = json.loads(FINDING.read_text())
copy(FINDING, EVIDENCE / "root-red" / FINDING.name)
for record in finding["artifacts"]:
    path = Path(record["path"])
    assert path.stat().st_size == record["bytes"] and digest(path) == record["sha256"]
    if path.suffixes[-2:] == [".tar", ".gz"]:
        copy(path, REPRODUCTION / "red" / path.name)
    elif "ci-artifacts" in path.parts:
        copy(path, EVIDENCE / "root-red" / "package" / path.name)
    else:
        copy(path, EVIDENCE / "root-red" / path.name)
container_log = Path("/tmp/plato-deployment-integration/publication-container-static.log")
copy(container_log, EVIDENCE / "root-red" / container_log.name)
default = json.loads((ROOT / "default-validation.json").read_text())
negative = json.loads((ROOT / "negative-checks.json").read_text())
assert all(result["accepted"] for result in negative["results"])
red = next(result for result in negative["results"] if result["name"] == "real-root-default-environment-red")
files = {
    path.relative_to(SOURCE).as_posix(): {"bytes": path.stat().st_size, "sha256": digest(path)}
    for directory in (EVIDENCE, REPRODUCTION)
    for path in sorted(directory.rglob("*")) if path.is_file()
}
manifest = EVIDENCE / "artifact-manifest.json"
manifest.write_text(json.dumps({"frozen_source_commit": COMMIT, "files": files}, indent=2) + "\n")
source_files = ("pyproject.toml", "uv.lock", ".python-version", "netlify.toml", "docs/build.sh",
                "docs/requirements.txt", "docs/mkdocs.yml", ".github/scripts/check_distribution.py",
                ".github/workflows/docs_package_checks.yml", ".github/workflows/pypi_publish.yml")
hashes = {}
for filename in source_files:
    frozen = subprocess.check_output(["git", "show", COMMIT + ":" + filename], cwd=SOURCE)
    assert frozen == (SOURCE / filename).read_bytes()
    hashes[filename] = hashlib.sha256(frozen).hexdigest()
qualification = {
    "schema_version": 1,
    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    "scope": "ROOT-P2-M3 original-author repair execution; a different fresh independent reviewer is required before acceptance.",
    "finding": "ROOT-P2-M3",
    "finding_sha256": digest(FINDING),
    "root_integrated_red_source": "0c942d7e81171e01bb715c0ba7b82f08900ea56a",
    "previous_author_source_commit": "63f0d2000f0370708b9afe1e2e6a2dc8559b72e1",
    "previous_author_evidence_commit": "cd40f3f862782ccf22fd95d5c7ba1822328842f9",
    "frozen_source_commit": COMMIT,
    "source_files_sha256": hashes,
    "authorized_source_delta": {"pyproject.toml": ["Added .venv-docs/** to Hatch build exclusions.",
                                                   "Added .venv-docs-bootstrap/** to Hatch build exclusions."],
                                "dependencies_versions_markers_lock_or_existing_p2_code_changed": False},
    "host": {"platform": platform.platform(), "machine": platform.machine(), "python": sys.version},
    "actual_default_paths": default,
    "independent_git_archive_byte_verification": json.loads((ROOT / "independent-source-bytes.json").read_text()),
    "environment_snapshots": {label: json.loads((ROOT / (label + "-environment-after.json")).read_text())
                              for label in ("default", "bootstrap")},
    "bootstrap_inventory": json.loads((ROOT / "bootstrap-inventory.json").read_text()),
    "root_preservation": json.loads((ROOT / "root-preservation-after.json").read_text()),
    "focused_negative_and_boundary_checks": len(negative["results"]),
    "executed_checks": [
        "Strict docs using the exact documented Python3.13 command and the default in-checkout .venv-docs, then the exact uv run --no-project checker command, passed.",
        "With uv absent from PATH, the helper created its default .venv-docs-bootstrap with uv0.12.22 and default .venv-docs; strict docs and the same subsequent package command passed.",
        "Both environments, their actual Python symlinks, generated docs/site, dirty CI output and added sentinel payloads remained present during package builds and were absent from wheel/sdist manifests.",
        "Package checking left docs/bootstrap environment bytes, paths, modes and mtimes untouched.",
        "A separate pristine source checkout passed the actual release-default checker command; all three green builds produced identical wheel and sdist hashes.",
        "All three builds retained locked constraints, clean extracted-sdist rebuild, matching metadata/payload, uv pip check, outside-checkout installed-wheel imports and tensor/NMS operations.",
        "Each actual sdist had 1298 source-file payloads equal to an independently read git archive of the frozen source, plus PKG-INFO exactly equal to wheel METADATA.",
        "Required/optional dependencies, extras, Requires-Python and retained workspace manifests matched pyproject; generated docs requirements matched a fresh locked export.",
        "Twenty-four focused checks retained untouched existing output rejection, real unrelated untracked and altered tracked build rejection, generated metadata/source/member rejection, forbidden generated/environment/cache/archive paths and owned-output failure propagation.",
        "The real root red sdist was preserved and rejected for .venv-docs/bin/python; all root environment and failure evidence bytes, symlinks, modes and mtimes remained untouched.",
        "uv lock --check, unchanged-source byte comparison, shell/TOML/workflow/embedded syntax, Ruff import/format and whitespace checks passed."
    ],
    "red_artifact": {"sha256": red["actual_sdist_sha256"],
                     "default_environment_members": red["default_environment_members"],
                     "absolute_environment_symlinks": red["absolute_environment_symlinks"],
                     "preserved_in_commit": (REPRODUCTION / "red/plato_learn-1.4.3.tar.gz").relative_to(SOURCE).as_posix(),
                     "current_checker_rejects": True},
    "retained_green_distributions": {label: "/tmp/plato-refresh-worktrees/docs-package-env-" + label + "/ci-artifacts/docs-package/distributions"
                                      for label in ("default", "bootstrap", "release")},
    "limits": [
        "Local macOS arm64 Python3.13.16 and uv0.12.22 execution only; no GitHub Linux or hosted Netlify execution was dispatched.",
        "These receipts bind source06ba6796 only. Root's final integrated source and any evidence-bearing final tree require their own binding and qualification.",
        "No push, PR, release, package publication or hosted deployment occurred; release.created/token publishing intent is unchanged.",
        "All docs/default/bootstrap/controller/runtime environments used for execution were created in dedicated author-owned temporary paths. Root's protected integration environments were read only for preservation verification.",
        "Existing docs helper and distribution checker were not changed; the earlier external-environment success claims did not cover the helper's default environment path.",
        "Checker temporary extracted-source and installed-wheel environments are removed after proof; actual raw green distributions and negative archives remain in author-local temporary paths.",
        "Reproduction helpers identify author-local test paths. A new reviewer must use a fresh independent source checkout."
    ],
    "artifact_manifest_sha256": digest(manifest),
    "reproduction_commands_in_commit": [(REPRODUCTION / (label + "-command.zsh")).relative_to(SOURCE).as_posix()
                                        for label in ("default", "bootstrap", "release")],
}
receipt = EVIDENCE / "qualification.json"
receipt.write_text(json.dumps(qualification, indent=2) + "\n")
print(json.dumps({"source_commit": COMMIT, "qualification_sha256": digest(receipt),
                  "artifact_count": len(files), "artifact_manifest_sha256": digest(manifest)}))
