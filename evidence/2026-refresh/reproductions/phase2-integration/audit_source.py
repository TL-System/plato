"""Record committed source identity, pinned submodules and untracked source files."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
records = []
problems = []

def git(root, *argv):
    return subprocess.check_output(["git", *argv], cwd=root)

def inspect(root, prefix=""):
    for line in git(root, "ls-files", "--stage", "-z").split(b"\0"):
        if not line:
            continue
        meta, path_bytes = line.split(b"\t", 1)
        mode, expected, stage = meta.decode().split()
        rel = path_bytes.decode()
        path = root / rel
        if stage != "0":
            problems.append(prefix + rel + ": unmerged")
            continue
        if mode == "160000":
            actual = git(path, "rev-parse", "HEAD").decode().strip()
            records.append(dict(path=prefix+rel, kind="submodule",
                                expected=expected, actual=actual))
            if actual != expected:
                problems.append(prefix + rel + ": submodule mismatch")
            inspect(path, prefix + rel + "/")
            continue
        data = path.readlink().as_posix().encode() if path.is_symlink() else path.read_bytes()
        actual = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        records.append(dict(path=prefix+rel, kind="blob", expected=expected,
                            actual=actual, sha256=hashlib.sha256(data).hexdigest()))
        if actual != expected:
            problems.append(prefix + rel + ": source differs from index")
    untracked = git(root, "ls-files", "--others", "--exclude-standard", "-z")
    for name in untracked.decode().split("\0"):
        if name and Path(name).suffix in (".py", ".pyc", ".so", ".pth"):
            problems.append(prefix + name + ": untracked importable source")
    if git(root, "diff", "--cached", "--name-only").strip():
        problems.append(prefix + ": index differs from HEAD")

inspect(args.root)
result = dict(root=str(args.root), commit=git(args.root, "rev-parse", "HEAD").decode().strip(),
              records=records, problems=problems, status="PASS" if not problems else "FAIL")
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(dict(status=result["status"], files=len(records), problems=problems)))
raise SystemExit(bool(problems))
