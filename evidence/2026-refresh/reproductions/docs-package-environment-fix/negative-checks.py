import hashlib
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path("/tmp/plato-p2-docs-environment-fix")
SOURCE = Path("/tmp/plato-refresh-worktrees/docs-package-refresh").resolve()
WORKFLOW = Path("/tmp/plato-refresh-worktrees/docs-package-env-default").resolve()
RELEASE = Path("/tmp/plato-refresh-worktrees/docs-package-env-release").resolve()
NEGATIVE = Path("/tmp/plato-refresh-worktrees/docs-package-env-negative").resolve()
COMMIT = "06ba679690f090aacf7e6a95a6dffd74b81ffad3"
PYTHON = sys.executable
UV = shutil.which("uv")
spec = importlib.util.spec_from_file_location("checker", SOURCE / ".github/scripts/check_distribution.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)
scratch = ROOT / "negative"
scratch.mkdir()
results = []

def snapshot(directory):
    data = {}
    for path in [directory] + sorted(directory.rglob("*")):
        status = path.lstat()
        item = {"mode": status.st_mode, "size": status.st_size, "mtime_ns": status.st_mtime_ns}
        if path.is_file():
            item["sha256"] = checker.sha256(path)
        data[path.relative_to(directory).as_posix()] = item
    return data

def command(label, arguments, cwd=NEGATIVE, environment=None, path_prefix=None, expected=1):
    script = shlex.join(arguments)
    if path_prefix:
        script = "PATH=" + shlex.quote(str(path_prefix) + ":" + os.environ["PATH"]) + " " + script
    process = subprocess.run(["zsh", "-lc", script], cwd=cwd, env=os.environ | (environment or {}),
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (scratch / (label + ".log")).write_text(process.stdout)
    assert process.returncode == expected, (label, process.returncode, process.stdout)
    record = {"name": label, "accepted": True, "exit_code": process.returncode,
              "command": arguments, "cwd": str(cwd),
              "path_prefix": str(path_prefix) if path_prefix else None}
    results.append(record)
    return process, record

def reject(label, action, expected):
    try:
        action()
    except ValueError as error:
        assert expected in str(error), (label, error)
        record = {"name": label, "accepted": True, "reason": str(error)}
        results.append(record)
        return record
    else:
        raise AssertionError("Negative input accepted: " + label)

existing = ROOT / "existing-external-output"
existing.mkdir()
(existing / "sentinel.txt").write_text("Original caller-owned external output.\n")
for label, output in (("existing-unsupported-tracked-directory", NEGATIVE / "docs"),
                      ("existing-external-directory", existing),
                      ("checkout-root", NEGATIVE)):
    before = snapshot(output)
    _, record = command(label, [PYTHON, ".github/scripts/check_distribution.py", "--output-dir", str(output)])
    assert before == snapshot(output)
    assert not (output / "acceptance.json").exists()
    record.update({"directory_unchanged": True, "snapshot_entries": len(before),
                   "snapshot_sha256": hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest()})

# An arbitrary caller-owned unsupported output must be left entirely unchanged.
unsupported = NEGATIVE / "arbitrary-generated-output"
unsupported.mkdir()
sentinel = unsupported / "existing-source.txt"
sentinel.write_text("Original caller-owned content.\n")
before = snapshot(unsupported)
process, record = command("existing-unsupported-untracked-directory",
    [PYTHON, ".github/scripts/check_distribution.py", "--output-dir", str(unsupported)])
assert before == snapshot(unsupported)
assert not (unsupported / "acceptance.json").exists()
record.update({"directory_unchanged": True, "snapshot": before})
# Restore only this test's own fixture before probing unrelated untracked input.
sentinel.unlink()
unsupported.rmdir()

existing_ci = WORKFLOW / "ci-artifacts/docs-package"
before = snapshot(existing_ci)
_, record = command("pre-existing-valid-ci-output", [PYTHON, ".github/scripts/check_distribution.py"],
                    cwd=WORKFLOW)
assert before == snapshot(existing_ci)
record.update({"directory_unchanged": True, "snapshot_entries": len(before),
               "snapshot_sha256": hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest()})

untracked = NEGATIVE / "unrelated-research-output.json"
untracked.write_text('{"unrelated_untracked_payload": true}\n')
try:
    process, record = command("actual-unrelated-untracked-default-build",
        [PYTHON, ".github/scripts/check_distribution.py"])
    output = NEGATIVE / "ci-artifacts/docs-package"
    receipt = json.loads((output / "acceptance.json").read_text())
    assert receipt["accepted"] is False
    assert "absent from frozen Git tree" in receipt["error"]
    assert "unrelated-research-output.json" in receipt["error"]
    assert checker.sha256(untracked) == hashlib.sha256(b'{"unrelated_untracked_payload": true}\n').hexdigest()
    record.update({"owned_failure_receipt": receipt, "source_commit": COMMIT,
                   "input_unchanged": True, "actual_sdist": str(next((output / "distributions").glob("*.tar.gz")))})
finally:
    untracked.unlink()

tracked = NEGATIVE / "plato/config.py"
original = tracked.read_bytes()
try:
    tracked.write_bytes(original + b"\n# Deliberately altered tracked source for P2-M2.\n")
    bad_output = NEGATIVE / "ci-artifacts/altered-tracked-source"
    _, record = command("actual-altered-tracked-source-build",
        [PYTHON, ".github/scripts/check_distribution.py", "--output-dir", str(bad_output)])
    receipt = json.loads((bad_output / "acceptance.json").read_text())
    assert receipt["accepted"] is False
    assert "Sdist bytes differ from frozen Git tree" in receipt["error"]
    assert "plato/config.py" in receipt["error"]
    record.update({"owned_failure_receipt": receipt, "source_commit": COMMIT,
                   "fixture_sha256": checker.sha256(tracked),
                   "frozen_sha256": hashlib.sha256(original).hexdigest(),
                   "actual_sdist": str(next((bad_output / "distributions").glob("*.tar.gz")))})
finally:
    tracked.write_bytes(original)
assert subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=NEGATIVE, text=True) == ""

package = WORKFLOW / "ci-artifacts/docs-package"
wheel = json.loads((package / "wheel-manifest.json").read_text())
sdist = json.loads((package / "sdist-manifest.json").read_text())
with (SOURCE / "pyproject.toml").open("rb") as incoming:
    import tomllib
    project = tomllib.load(incoming)["project"]
sources = set(subprocess.check_output(["git", "ls-files", "plato/**/*.py", "plato/*.py"],
                                     cwd=SOURCE, text=True).splitlines())
accepted_binding = checker.bind_sdist_source(SOURCE, COMMIT, sdist, wheel)
assert accepted_binding["generated_metadata_matches_wheel"]
assert accepted_binding["git_tree_bound_members"] == len(sdist["files"]) - 1
results.append({"name": "exact-pkg-info-generated-exception", "accepted": True,
                "git_tree_bound_members": accepted_binding["git_tree_bound_members"],
                "metadata_sha256": accepted_binding["generated_metadata"]["PKG-INFO"]})

red = Path("/tmp/plato-refresh-worktrees/deployment-integration/ci-artifacts/docs-package/distributions/plato_learn-1.4.3.tar.gz")
assert checker.sha256(red) == "206a0cbf2910daaf58cb7d44776ef478d6c967728e00c8e129622fee72f0b36a"
record = reject("real-root-default-environment-red",
                lambda: checker.inspect_distribution(red, project, sources), "Non-regular sdist source member")
with tarfile.open(red) as archive:
    environment_members = [member for member in archive.getmembers()
                           if ".venv-docs" in Path(member.name).parts]
    absolute_links = {member.name: member.linkname for member in environment_members
                      if member.issym() and member.linkname.startswith("/")}
assert environment_members and absolute_links
record.update({"actual_sdist_sha256": checker.sha256(red),
               "default_environment_members": len(environment_members),
               "absolute_environment_symlinks": absolute_links})

raw_sdist = next((package / "distributions").glob("*.tar.gz"))
with tarfile.open(raw_sdist) as incoming:
    root_name = incoming.getmembers()[0].name.split("/")[0]
    original_members = {member.name: (member, incoming.extractfile(member).read())
                        for member in incoming.getmembers() if member.isfile()}
for label, relative, action, expected in (
    ("injected-untracked-sdist", "unrelated-generated.txt", "add", "absent from frozen Git tree"),
    ("altered-tracked-sdist", "README.md", "change", "Sdist bytes differ"),
    ("unverified-pkg-info", "PKG-INFO", "change", "PKG-INFO must match"),
    ("other-generated-metadata", "backend-generated.json", "add", "absent from frozen Git tree"),
):
    folder = scratch / label
    folder.mkdir()
    mutated = folder / raw_sdist.name
    with tarfile.open(mutated, "w:gz") as outgoing:
        for name, (member, payload) in original_members.items():
            if action == "change" and name == root_name + "/" + relative:
                payload += b"\nP2-M2 altered bytes.\n"
            copy = tarfile.TarInfo(name)
            copy.size = len(payload)
            outgoing.addfile(copy, __import__("io").BytesIO(payload))
        if action == "add":
            payload = b"P2-M2 untracked bytes.\n"
            member = tarfile.TarInfo(root_name + "/" + relative)
            member.size = len(payload)
            outgoing.addfile(member, __import__("io").BytesIO(payload))
    inspected = checker.inspect_distribution(mutated, project, sources)
    record = reject(label, lambda: checker.bind_sdist_source(SOURCE, COMMIT, inspected, wheel), expected)
    record["actual_sdist_sha256"] = checker.sha256(mutated)

for label, member_path, kind in (
    ("duplicate-sdist-member", "README.md", "duplicate"),
    ("nonregular-sdist-member", "untracked-link", "symlink"),
):
    folder = scratch / label
    folder.mkdir()
    mutated = folder / raw_sdist.name
    with tarfile.open(mutated, "w:gz") as outgoing:
        for name, (_, payload) in original_members.items():
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            outgoing.addfile(member, __import__("io").BytesIO(payload))
        member = tarfile.TarInfo(root_name + "/" + member_path)
        if kind == "duplicate":
            payload = b"Duplicate source bytes.\n"
            member.size = len(payload)
            outgoing.addfile(member, __import__("io").BytesIO(payload))
        else:
            member.type = tarfile.SYMTYPE
            member.linkname = "README.md"
            outgoing.addfile(member)
    record = reject(label, lambda: checker.inspect_distribution(mutated, project, sources),
                    "Duplicate" if kind == "duplicate" else "Non-regular")
    record["actual_sdist_sha256"] = checker.sha256(mutated)

stubs = scratch / "failure-stub"
stubs.mkdir()
wrapper = stubs / "uv"
wrapper.write_text("#!" + PYTHON + "\nimport os, sys\n"
    "if sys.argv[1:] == ['--version']: print('uv 0.12.22'); sys.exit(0)\n"
    "if sys.argv[1] == 'build': sys.exit(43)\n"
    "os.execv(" + repr(UV) + ", [" + repr(UV) + "] + sys.argv[1:])\n")
wrapper.chmod(0o755)
failed_output = NEGATIVE / "ci-artifacts/owned-build-failure"
assert not failed_output.exists()
_, record = command("owned-output-build-failure-receipt",
    [PYTHON, ".github/scripts/check_distribution.py", "--output-dir", str(failed_output)],
    path_prefix=stubs)
failed_receipt = json.loads((failed_output / "acceptance.json").read_text())
assert failed_receipt["accepted"] is False and "43" in failed_receipt["error"]
record.update({"output_created_by_invocation": True, "failure_receipt": failed_receipt})

# Preserve generated-path and old archive gates with the exact frozen checker.
for label, member_path in (
    ("generated-docs-site", "docs/site/index.html"),
    ("generated-ci-artifact", "ci-artifacts/provenance.json"),
    ("archived-source", "archives/retired/probe.py"),
    ("environment", ".venv/lib/probe.py"),
    ("default-docs-environment", ".venv-docs/lib/probe.py"),
    ("default-bootstrap-environment", ".venv-docs-bootstrap/lib/probe.py"),
    ("cache", "__pycache__/probe.pyc"),
    ("reproduction", "evidence/2026-refresh/reproductions/probe.json"),
):
    mutation = dict(sdist)
    mutation["files"] = sdist["files"] | {member_path: hashlib.sha256(b"sentinel").hexdigest()}
    # Exercise the actual archive manifest reader rather than its path predicate.
    folder = scratch / label
    folder.mkdir()
    mutated = folder / raw_sdist.name
    with tarfile.open(mutated, "w:gz") as outgoing:
        for name, (_, payload) in original_members.items():
            entry = tarfile.TarInfo(name)
            entry.size = len(payload)
            outgoing.addfile(entry, __import__("io").BytesIO(payload))
        payload = b"sentinel"
        entry = tarfile.TarInfo(root_name + "/" + member_path)
        entry.size = len(payload)
        outgoing.addfile(entry, __import__("io").BytesIO(payload))
    reject(label, lambda: checker.inspect_distribution(mutated, project, sources),
           "Forbidden distribution payloads")
(ROOT / "negative-checks.json").write_text(json.dumps({
    "source_commit": COMMIT, "checker_sha256": checker.sha256(SOURCE / ".github/scripts/check_distribution.py"),
    "accepted": True, "results": results,
}, indent=2) + "\n")
print(json.dumps({"accepted": True, "checks": len(results), "source_commit": COMMIT,
                  "git_tree_bound_members": accepted_binding["git_tree_bound_members"]}))
