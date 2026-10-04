import ast
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

import yaml

ROOT = Path("/tmp/plato-p2-workflow-context-fix")
SOURCE = Path("/tmp/plato-refresh-worktrees/docs-package-refresh")
COMMIT = "8dd6a604712973c6db050082ccd35f154d59ce56"
BASE = "24b5e154bd43795ea6b2a47b6c9458fdf3a587ea"
PUBLISHED = "0791c01afcaea96223a925a32ee22882dc437ee2"
DOCS = ".github/workflows/docs_package_checks.yml"
PUBLISH = ".github/workflows/pypi_publish.yml"
TOOL = ROOT / "tools/actionlint"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen(filename, revision=COMMIT):
    return subprocess.check_output(["git", "show", revision + ":" + filename], cwd=SOURCE)


assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True).strip() == COMMIT
assert (ROOT / "green-actionlint-exit.txt").read_text().strip() == "0"
assert (ROOT / "green-actionlint.log").read_bytes() == b""
version = subprocess.check_output([str(TOOL), "-version"], text=True)
assert version.splitlines()[0] == "1.7.12"
(ROOT / "actionlint-version.log").write_text(version)
old = yaml.load(frozen(DOCS, BASE), Loader=yaml.BaseLoader)
current = yaml.load(frozen(DOCS), Loader=yaml.BaseLoader)
old_job = old["jobs"]["docs-package"]
old_value = old_job["env"].pop("PLATO_DOCS_ENVIRONMENT")
old_step = next(step for step in old_job["steps"] if step["name"] == "Build strict docs and verify locked requirements parity")
old_step["env"] = {"PLATO_DOCS_ENVIRONMENT": old_value}
assert current == old
assert frozen(PUBLISH) == frozen(PUBLISH, BASE) == frozen(PUBLISH, PUBLISHED)
assert "runner." not in frozen(PUBLISH).decode()
assert subprocess.check_output(["git", "diff", "--name-only", BASE, COMMIT], cwd=SOURCE, text=True).splitlines() == [DOCS]
hashes = {}
for filename in (DOCS, PUBLISH, "docs/build.sh", "docs/requirements.txt", "netlify.toml",
                 "pyproject.toml", "uv.lock", ".github/scripts/check_distribution.py"):
    payload = frozen(filename)
    assert payload == (SOURCE / filename).read_bytes(), filename
    hashes[filename] = hashlib.sha256(payload).hexdigest()
    if filename != DOCS:
        assert payload == frozen(filename, BASE), filename
for filename in (DOCS, PUBLISH):
    workflow = yaml.load(frozen(filename), Loader=yaml.BaseLoader)
    assert workflow["permissions"] == {"contents": "read"}
    for job in workflow["jobs"].values():
        assert job["runs-on"] == "ubuntu-24.04" and job["timeout-minutes"] == "30"
        assert job["defaults"]["run"]["shell"] == 'zsh -lc ". {0}"'
        for step in job["steps"]:
            if "run" not in step:
                continue
            if step.get("shell") == "python {0}":
                ast.parse(step["run"])
            else:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".zsh") as script:
                    script.write(step["run"])
                    script.flush()
                    subprocess.run(["zsh", "-n", script.name], check=True)
    if filename == PUBLISH:
        assert workflow["on"] == {"release": {"types": ["created"]}}
run = json.loads((ROOT / "remote-run.json").read_text())
jobs = json.loads((ROOT / "remote-jobs.json").read_text())
assert run["id"] == 37167176060 and run["head_sha"] == PUBLISHED and run["conclusion"] == "failure"
assert jobs["total_count"] == 0 and jobs["jobs"] == []


class TextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, text):
        self.parts.append(text)


parser = TextParser()
parser.feed((ROOT / "remote-run.html").read_text())
text = " ".join(" ".join(parser.parts).split())
annotation = "(Line: 20, Col: 31): Unrecognized named-value: 'runner'. Located at position 1 within expression: runner.temp"
assert annotation in text, annotation
remote = {"run_id": run["id"], "head_sha": run["head_sha"], "event": run["event"],
          "conclusion": run["conclusion"], "created_jobs": 0, "annotation": annotation,
          "url": run["html_url"], "raw_api_sha256": digest(ROOT / "remote-run.json"),
          "raw_jobs_sha256": digest(ROOT / "remote-jobs.json"), "raw_html_sha256": digest(ROOT / "remote-run.html")}
(ROOT / "remote-failure.json").write_text(json.dumps(remote, indent=2) + "\n")
evidence = SOURCE / "evidence/2026-refresh/docs-package-context-fix"
reproduction = SOURCE / "evidence/2026-refresh/reproductions/docs-package-context-fix"
assert not evidence.exists() and not reproduction.exists()
evidence.mkdir(parents=True)
reproduction.mkdir(parents=True)
for name in ("red-validation.json", "red-docs_package_checks.log", "red-pypi_publish.log",
             "green-actionlint.log", "green-actionlint-exit.txt", "actionlint-version.log",
             "remote-run.json", "remote-jobs.json", "remote-failure.json"):
    shutil.copy2(ROOT / name, evidence / name)
for name in ("red-docs_package_checks.yml", "red-pypi_publish.yml", "remote-run.html",
             "verify-and-record.py"):
    shutil.copy2(ROOT / name, reproduction / name)
shutil.copy2(ROOT / "tools/actionlint_1.7.12_checksums.txt", reproduction / "actionlint_1.7.12_checksums.txt")
previous_gate = Path("/tmp/plato-deployment-integration/final-publication-acceptance.json")
shutil.copy2(previous_gate, evidence / "root-previous-final-publication-acceptance.json")
files = {path.relative_to(SOURCE).as_posix(): {"bytes": path.stat().st_size, "sha256": digest(path)}
         for folder in (evidence, reproduction) for path in sorted(folder.rglob("*")) if path.is_file()}
manifest = evidence / "artifact-manifest.json"
manifest.write_text(json.dumps({"frozen_source_commit": COMMIT, "files": files}, indent=2) + "\n")
receipt = {
    "schema_version": 1, "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    "finding": "ROOT-P2-M4", "scope": "Original-author correction; fresh different independent review and root integrated gate required before pushing.",
    "previous_author_candidate": BASE, "published_remote_rejected_source": PUBLISHED,
    "frozen_source_commit": COMMIT, "source_files_sha256": hashes,
    "authorized_source_delta": {DOCS: "Moved PLATO_DOCS_ENVIRONMENT from job env to the sole consuming strict-docs step env, preserving its value."},
    "unchanged_related_publish_workflow": {"file": PUBLISH, "sha256": hashes[PUBLISH],
                                           "original_and_current_actionlint_pass": True,
                                           "same_runner_context_defect_present": False},
    "remote_actual_failure": remote,
    "validator": {"name": "actionlint", "version": "1.7.12", "version_output": version,
                  "official_release": "https://github.com/rhysd/actionlint/releases/tag/v1.7.12",
                  "archive_url": "https://github.com/rhysd/actionlint/releases/download/v1.7.12/actionlint_1.7.12_darwin_arm64.tar.gz",
                  "archive_sha256": digest(ROOT / "tools/actionlint_1.7.12_darwin_arm64.tar.gz"),
                  "official_checksum_verified": True, "binary_sha256": digest(TOOL),
                  "invocation": [str(TOOL), "-no-color", DOCS, PUBLISH],
                  "corrected_frozen_source_exit_code": 0, "optional_checks_explicitly_disabled": False},
    "executed_checks": [
        "Actionlint reproduced the exact runner-context rejection on the original docs workflow; its bytes also match published0791c01.",
        "Actionlint passed both corrected frozen docs and unchanged release workflows without ignores or explicit optional-check disable flags.",
        "Published remote REST metadata reports failure on0791c01 and zero created jobs; the raw public page contains the reported annotation.",
        "Parsed workflow comparison confirms the sole semantic delta is moving the same docs environment value to its consuming step.",
        "All other source files, release.created trigger, read-only permissions, secrets placement, action/version pins, runner, timeouts and build/publish commands are unchanged.",
        "Embedded Python AST and zsh syntax validation passed for both P2 workflows; source whitespace check passed."
    ],
    "primary_references": [
        {"url": "https://docs.github.com/en/actions/reference/workflows-and-actions/contexts#context-availability",
         "supports": "runner is absent from jobs.<job_id>.env contexts and available in jobs.<job_id>.steps.env."},
        {"url": "https://github.com/rhysd/actionlint/blob/v1.7.12/docs/checks.md#expression-check",
         "supports": "Actionlint validates expression context availability at workflow locations."},
        {"url": run["html_url"], "supports": "The actual published workflow was rejected before jobs for runner.temp at job-level env."}
    ],
    "artifact_manifest_sha256": digest(manifest),
    "limits": [
        "Local macOS arm64 actionlint validation only. No corrected workflow was pushed, remotely dispatched or run on GitHub.",
        "No docs/package/training runtime tests were repeated for this context-only change. Root's previous0791c01 runtime proof is retained as historical evidence and is not rebound to this source.",
        "Final integrated source requires root's new gate addendum and real remote jobs before merge; local actionlint does not claim remote runtime success.",
        "Tool and existing author-owned controller environment only; no root environment or integration checkout was modified.",
        "Prior candidates and receipts remain preserved; no push, PR update, release or publication occurred."
    ],
}
qualification = evidence / "qualification.json"
qualification.write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({"source_commit": COMMIT, "qualification_sha256": digest(qualification),
                  "artifact_count": len(files), "actionlint_green_exit": 0}))
