"""Prove the Phase4 boundary and helper mechanics, never family qualification."""

import ast
import copy
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.examples_phase4 import common
from tests.test_mlx_profile_contract import (
    policy,
    profile_source,
    project,
)
from tests.test_mlx_profile_contract import (
    run_pytest as _run_pytest,
)

REPO = Path(__file__).resolve().parents[1]
MODULE = "tests/examples_phase4/test_core.py"
LEDGER = json.loads((REPO / "tests/examples_phase4/cases.json").read_text())


def run_pytest(project, *arguments, **options):
    result = _run_pytest(project, *arguments, **options)
    number = len(list(project.root.glob("phase4-policy-*.json")))
    prefix = project.root / f"phase4-policy-{number}"
    prefix.with_suffix(".log").write_text(result.output)
    prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "arguments": list(map(str, arguments)),
                "options": {key: str(value) for key, value in options.items()},
                "status": result.returncode,
            },
            indent=2,
        )
        + "\n"
    )
    return result


def _spec(**fields):
    return {
        "case_id": "contract",
        "family": "basic",
        "config": "examples/customized_client_training/fedprox/fedprox_MNIST_lenet5.toml",
        "entrypoint": "examples/customized_client_training/fedprox/fedprox.py",
        "cwd": ".",
        "seed": 7,
        "overlays": {},
        "allowed_overlay_paths": {},
        "required_distributions": [],
        "source_paths": [],
        "timeout_seconds": 20,
        "transport": "in-process",
        **fields,
    }


def _case(node):
    return {
        "nodeid": node,
        "case_id": "contract",
        "family": "basic",
        "config_paths": [],
        "include_consumers": [],
        "entrypoint_paths": [],
        "branch_labels": ["synthetic-policy-probe"],
        "proof_kind": "real-execution",
        "required_distributions": [],
        "overlay_allowed_paths_with_rationale": {},
        "timeout_seconds": 20,
        "explicit_source_paths": [],
        "authored_spec_sha256": common._digest(_spec()),
    }


@pytest.fixture
def phase_project(project):
    boundary = project.tests / "examples_phase4"
    boundary.mkdir()
    (boundary / "__init__.py").write_text('"""Inert contract namespace."""\n')
    marker = "from pathlib import Path\nPath(__file__).parents[2].joinpath('phase-import').touch()\n"
    (boundary / "test_core.py").write_text(marker + "def test_case():\n    pass\n")
    ledger = copy.deepcopy(LEDGER)
    row = next(r for r in ledger["tasks"] if r["test_module"] == MODULE)
    row.update(state="frozen", cases=[_case(MODULE + "::test_case")])
    (boundary / "cases.json").write_text(json.dumps(ledger))
    project.phase4, project.ledger = boundary, ledger
    return project


def _write_ledger(project):
    (project.phase4 / "cases.json").write_text(json.dumps(project.ledger))


def test_inert_import_and_standard_library_helpers():
    init = ast.parse((REPO / "tests/examples_phase4/__init__.py").read_text())
    assert all(
        isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) for n in init.body
    )
    tree = ast.parse((REPO / "tests/examples_phase4/common.py").read_text())
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [a.name.split(".")[0] for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module.split(".")[0]]
            )
            assert all(
                name in sys.stdlib_module_names or name == "__future__"
                for name in names
            )
    script = "import tests.examples_phase4; import tests.examples_phase4.common; import sys; assert not any(n.split('.')[0] in {'torch', 'numpy', 'psutil', 'plato'} for n in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=REPO,
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


def test_draft_ownership_matches_frozen_authority(policy):
    assert len(LEDGER["tasks"]) == 12
    assert all(r["state"] == "draft" and r["cases"] == [] for r in LEDGER["tasks"])
    inventory = LEDGER["source_inventory"]
    assert len(inventory) == len({r["path"] for r in inventory}) == 399
    assert common._digest(inventory) == LEDGER["path_map_sha256"]
    paths = [p for row in LEDGER["tasks"] for p in row["owned_paths"]]
    assert len(paths) == len(set(paths))
    bindings = [b for row in LEDGER["tasks"] for b in row["config_bindings"]]
    assert len(bindings) == len({b["config"] for b in bindings}) == 136
    assert all(r["families"] for r in LEDGER["tasks"])
    for row in LEDGER["tasks"]:
        with pytest.raises(pytest.UsageError, match="draft or empty"):
            policy._load_phase4_ledger(
                REPO / "tests/examples_phase4/cases.json", row["test_module"]
            )


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["."],
        ["tests"],
        ["tests", "--test-profile=base"],
        ["tests", "--test-profile=mandatory"],
    ],
)
def test_ordinary_collection_excludes_before_import(phase_project, arguments):
    result = run_pytest(phase_project, *arguments, "--collect-only")
    assert result.returncode == 0, result.output
    assert not (phase_project.root / "phase-import").exists()
    assert MODULE not in result.output


@pytest.mark.parametrize(
    "arguments",
    [
        ["tests", "--test-profile=examples-phase4"],
        [MODULE, "--test-profile=mandatory"],
        [MODULE, "tests/test_core.py"],
        [MODULE, MODULE, "--test-profile=examples-phase4"],
        [MODULE + "::test_case", "--test-profile=examples-phase4"],
        ["tests/examples_phase4", "--test-profile=examples-phase4"],
        [MODULE, "--test-profile=examples-phase4", "-k", "case"],
        [MODULE, "--test-profile=examples-phase4", "-m", "examples_phase4"],
        [
            MODULE,
            "--test-profile=examples-phase4",
            "--deselect=" + MODULE + "::test_case",
        ],
        [MODULE, "--test-profile=examples-phase4", "--ignore=other.py"],
        [MODULE, "--test-profile=examples-phase4", "--ignore-glob=*other*"],
        ["tests", "-m", "examples_phase4"],
    ],
)
def test_bad_scope_or_filters_reject_before_import(phase_project, arguments):
    result = run_pytest(phase_project, *arguments)
    assert result.returncode != 0, result.output
    assert not (phase_project.root / "phase-import").exists()


@pytest.mark.parametrize(
    "selector,directory",
    [
        ("tests.examples_phase4.test_core", ""),
        ("examples_phase4.test_core", "tests"),
        ("test_core", "tests/examples_phase4"),
    ],
)
def test_pyargs_aliases_reject_before_import(phase_project, selector, directory):
    result = run_pytest(
        phase_project, "--pyargs", selector, cwd=phase_project.root / directory
    )
    assert result.returncode != 0, result.output
    assert "Phase4 selector unsupported" in result.output
    assert not (phase_project.root / "phase-import").exists()


@pytest.mark.parametrize("configured", [False, True])
def test_environment_and_configured_filters_reject(phase_project, configured):
    if configured:
        p = phase_project.root / "pytest.ini"
        p.write_text(
            p.read_text().replace("--strict-markers", "--strict-markers -k case")
        )
    result = run_pytest(
        phase_project,
        MODULE,
        "--test-profile=examples-phase4",
        addopts=None if configured else "-k case",
    )
    assert result.returncode != 0
    assert not (phase_project.root / "phase-import").exists()


@pytest.mark.parametrize("collect_only", [False, True])
def test_complete_and_collection_only_labels(phase_project, collect_only):
    args = ["--collect-only"] if collect_only else []
    result = run_pytest(phase_project, MODULE, "--test-profile=examples-phase4", *args)
    assert result.returncode == 0, result.output
    expected = (
        "collection-only validation (no execution qualification)"
        if collect_only
        else "complete task qualification"
    )
    assert "Phase4 P4-core " + expected in result.output


@pytest.mark.parametrize(
    "damage", ["draft", "empty", "missing", "extra", "duplicate", "substituted"]
)
def test_ledger_damage_cannot_execute_or_qualify(phase_project, damage):
    row = next(r for r in phase_project.ledger["tasks"] if r["test_module"] == MODULE)
    if damage == "draft":
        row["state"] = "draft"
    elif damage == "empty":
        row["cases"] = []
    elif damage == "missing":
        row["cases"].append(
            {
                **row["cases"][0],
                "case_id": "missing",
                "nodeid": MODULE + "::test_missing",
            }
        )
    elif damage == "duplicate":
        row["cases"].append(row["cases"][0])
    elif damage == "substituted":
        row["cases"][0]["nodeid"] = MODULE + "::test_other"
    else:
        with (phase_project.phase4 / "test_core.py").open("a") as f:
            f.write('def test_extra():\n    raise AssertionError("executed-extra")\n')
    _write_ledger(phase_project)
    result = run_pytest(phase_project, MODULE, "--test-profile=examples-phase4")
    assert result.returncode != 0, result.output
    assert "Phase4 P4-core complete task qualification" not in result.output
    assert "executed-extra" not in result.output


@pytest.mark.parametrize(
    "body",
    [
        'import pytest\ndef test_case():\n    pytest.skip("call skip")\n',
        'import pytest\n@pytest.fixture(autouse=True)\ndef f():\n    pytest.skip("setup skip")\ndef test_case():\n    pass\n',
        'import pytest\n@pytest.fixture(autouse=True)\ndef f():\n    yield\n    pytest.skip("teardown skip")\ndef test_case():\n    pass\n',
        'import pytest\npytest.skip("collection skip", allow_module_level=True)\n',
        "import pytest\n@pytest.mark.xfail\ndef test_case():\n    assert False\n",
        "import pytest\n@pytest.mark.xfail\ndef test_case():\n    pass\n",
        'import pytest\n@pytest.fixture(autouse=True)\ndef f():\n    raise ValueError("setup failure")\ndef test_case():\n    pass\n',
        'import pytest\n@pytest.fixture(autouse=True)\ndef f():\n    yield\n    raise ValueError("teardown failure")\ndef test_case():\n    pass\n',
    ],
)
@pytest.mark.parametrize("strict", [False, True])
def test_skip_xfail_and_errors_fail_even_focused(phase_project, body, strict):
    (phase_project.phase4 / "test_core.py").write_text(body)
    args = ["--test-profile=examples-phase4"] if strict else []
    result = run_pytest(phase_project, MODULE, *args)
    assert result.returncode != 0, result.output
    assert "Phase4 P4-core complete task qualification" not in result.output


def test_focused_and_spoofed_marker(phase_project):
    result = run_pytest(phase_project, MODULE, "-k", "case")
    assert result.returncode == 0, result.output
    assert "Phase4 focused execution (not complete task qualification)" in result.output
    (phase_project.tests / "test_spoof.py").write_text(
        "import pytest\n@pytest.mark.examples_phase4\ndef test_spoof():\n    pass\n"
    )
    result = run_pytest(phase_project, "tests/test_spoof.py")
    assert result.returncode != 0
    assert "marker outside boundary" in result.output


def test_canonical_symlink_boundary(phase_project):
    alias = phase_project.root / "alias.py"
    alias.symlink_to(phase_project.phase4 / "test_core.py")
    result = run_pytest(phase_project, "alias.py", "--test-profile=mandatory")
    assert result.returncode != 0
    assert not (phase_project.root / "phase-import").exists()


def test_config_provenance_overlay_and_restoration(tmp_path):
    from plato.config import Config, TomlConfigLoader

    spec = _spec(
        overlays={
            "trainer": {"epochs": 2, "contract_list": [3], "contract_null": None}
        },
        allowed_overlay_paths={
            "trainer.epochs": "bounded execution",
            "trainer.contract_list": "contract fixture",
            "trainer.contract_null": "contract fixture",
        },
    )
    previous = Config._instance, sys.argv, os.environ.copy(), Path.cwd(), sys.path[:]
    root = tmp_path / "config"
    with pytest.raises(RuntimeError, match="deliberate"):
        with common.configured_case(spec, root):
            assert Config.trainer.epochs == 2
            assert Config.trainer.contract_list == [3]
            assert Config.trainer.contract_null is None
            assert Config.device() == "cpu"
            raise RuntimeError("deliberate")
    assert (
        Config._instance,
        sys.argv,
        os.environ.copy(),
        Path.cwd(),
        sys.path[:],
    ) == previous
    record = json.loads((root / "configuration.json").read_text())
    assert record["resolved"] == TomlConfigLoader(REPO / spec["config"]).load()
    assert len(record["includes"]) > 1
    assert all(
        hashlib.sha256(Path(v["path"]).read_bytes()).hexdigest() == v["sha256"]
        for v in record["includes"]
    )
    assert {d["path"]: d["after"] for d in record["authored_overlay_diff"]}[
        "trainer.contract_list"
    ] == [3]
    assert record["runtime_overlay_diff"]
    events = [
        json.loads(s)
        for s in next(root.glob("events-*.jsonl")).read_text().splitlines()
    ]
    assert [e["event"] for e in events] == ["configured", "failure", "provenance"]
    with pytest.raises(ValueError, match="unreviewed overlay"):
        with common.configured_case(
            _spec(overlays={"algorithm": {"type": "fake"}}), tmp_path / "forbidden"
        ):
            pytest.fail("forbidden overlay executed")


def test_real_nested_list_null_and_include_errors(tmp_path):
    from plato.config import TomlConfigLoader

    (tmp_path / "base.toml").write_text(
        "[trainer]\nitems = [1]\nnone = {null = true}\n"
    )
    (tmp_path / "middle.toml").write_text(
        'include = "base.toml"\n[trainer]\nitems = [2]\n'
    )
    (tmp_path / "leaf.toml").write_text('include = ["middle.toml", "base.toml"]\n')
    assert TomlConfigLoader(tmp_path / "leaf.toml").load()["trainer"] == {
        "items": [1, 2, 1],
        "none": None,
    }
    (tmp_path / "leaf.toml").write_text('include = "missing.toml"\n')
    with pytest.raises(FileNotFoundError):
        TomlConfigLoader(tmp_path / "leaf.toml").load()
    (tmp_path / "base.toml").write_text('include = "leaf.toml"\n')
    (tmp_path / "leaf.toml").write_text('include = "base.toml"\n')
    with pytest.raises(ValueError, match="Circular include"):
        TomlConfigLoader(tmp_path / "leaf.toml").load()


@pytest.fixture
def worker_repo(tmp_path, monkeypatch):
    """Copy the real helpers into a task-owned miniature checkout for child probes."""
    root = tmp_path / "repo"
    for relative in [
        "tests/__init__.py",
        "tests/examples_phase4/__init__.py",
        "tests/examples_phase4/common.py",
        "tests/integration/utils.py",
        "plato/__init__.py",
        "plato/config.py",
        "plato/utils/__init__.py",
        "plato/utils/toml_writer.py",
    ]:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPO / relative).read_bytes())
    (root / "leaf.toml").write_text((REPO / "tests/config.toml").read_text())
    (root / "entry.py").write_text('"""Contract entrypoint."""\n')
    monkeypatch.setattr(common, "REPO", root)
    return root


def _worker(root, body):
    path = root / "tests/examples_phase4/contract_worker.py"
    path.write_text(
        'import json, os, sys\nfrom pathlib import Path\nfrom tests.examples_phase4.common import configured_case, emit, source_snapshot\nspec = json.loads(Path(os.environ["PLATO_PHASE4_SPEC"]).read_text())\ndirectory = Path(spec["runtime"]["directory"])\nemit(directory, "worker_started")\n'
        + body
    )
    return "tests.examples_phase4.contract_worker"


@pytest.mark.parametrize("cwd", [".", "example"])
def test_real_worker_natural_exit_identity_and_numeric_events(
    worker_repo, tmp_path, cwd
):
    worker = _worker(
        worker_repo,
        'with configured_case(spec, directory):\n    import entry\n    emit(directory, "numeric", loss=0.25)\nemit(directory, "success")\n',
    )
    (worker_repo / "example").mkdir()
    (worker_repo / "example/entry.py").write_text(
        '"""Contract example entrypoint."""\n'
    )
    spec = _spec(config="leaf.toml", entrypoint="example/entry.py", cwd=cwd)
    result = common.run_case(tmp_path / "success", worker=worker, spec=spec)
    assert (
        result["returncode"] == 0 and not result["forced"] and not result["survivors"]
    )
    assert any(e.get("loss") == 0.25 for e in result["events"])
    assert result["environment"]["PYTHONPATH"].split(os.pathsep) == (
        [str(worker_repo), str(worker_repo / "example")]
        if cwd == "example"
        else [str(worker_repo)]
    )
    assert (tmp_path / "success/stdout.log").exists()


@pytest.mark.parametrize(
    "damage",
    [
        "assertion",
        "malformed",
        "timeout",
        "missing_dependency",
        "wrong_origin",
        "bad_success",
        "invalid_provenance",
    ],
)
def test_real_failed_workers_retain_evidence_and_contain(worker_repo, tmp_path, damage):
    body = "with configured_case(spec, directory):\n    import entry\n"
    if damage == "assertion":
        body += '    assert False, "real assertion"\n'
    elif damage == "missing_dependency":
        body += "    import phase4_intentionally_absent_dependency\n"
    elif damage == "wrong_origin":
        (tmp_path / "entry.py").write_text('"""Wrong origin."""\n')
        body += f'    sys.path.insert(0, {str(tmp_path)!r})\n    sys.modules.pop("entry", None)\n    import entry\n'
    elif damage == "timeout":
        body += '    import subprocess, psutil, time\n    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])\n    emit(directory, "child_started", child_pid=child.pid, child_created=psutil.Process(child.pid).create_time())\n    time.sleep(30)\n'
    elif damage == "malformed":
        body += (
            '    directory.joinpath("events-bad.jsonl").write_text("invalid json\\n")\n'
        )
    if damage == "invalid_provenance":
        body += 'emit(directory, "provenance", snapshot={})\n'
    body += 'emit(directory, "success")\n'
    if damage == "bad_success":
        body += 'emit(directory, "success")\n'
    worker = _worker(worker_repo, body)
    spec = _spec(
        config="leaf.toml",
        entrypoint="entry.py",
        timeout_seconds=3 if damage == "timeout" else 20,
    )
    with pytest.raises(AssertionError, match="Phase4 case failed"):
        common.run_case(tmp_path / "failure", worker=worker, spec=spec)
    result = json.loads((tmp_path / "failure/result.json").read_text())
    assert result["errors"] and result["survivors"] == []
    assert (tmp_path / "failure/stderr.log").exists()
    if damage == "timeout":
        assert result["timed_out"] and result["forced"]
    if damage == "missing_dependency":
        assert "ModuleNotFoundError" in (tmp_path / "failure/stderr.log").read_text()
    if damage == "wrong_origin":
        assert (
            any(
                "wrong entrypoint origin" in e.get("message", "")
                for e in result["events"]
            )
            or "wrong entrypoint origin"
            in (tmp_path / "failure/stderr.log").read_text()
        )


def test_frozen_spec_and_worker_binding_precedes_launch(
    worker_repo, tmp_path, monkeypatch
):
    ledger = copy.deepcopy(LEDGER)
    row = next(r for r in ledger["tasks"] if r["task_id"] == "P4-core")
    row.update(state="frozen", cases=[_case(MODULE + "::test_case")])
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps(ledger))
    monkeypatch.setenv("PLATO_PHASE4_STRICT_TASK", "P4-core")
    monkeypatch.setenv("PLATO_PHASE4_LEDGER", str(path))
    with pytest.raises(AssertionError, match="frozen ledger"):
        common.run_case(
            tmp_path / "never",
            worker="tests.examples_phase4.wrong_worker",
            spec=_spec(),
        )
    with pytest.raises(AssertionError, match="frozen ledger"):
        common.run_case(
            tmp_path / "never", worker=row["worker_module"], spec=_spec(seed=8)
        )
    assert not (tmp_path / "never").exists()


def test_unrelated_process_identity_is_never_signaled(worker_repo, tmp_path):
    import psutil

    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        body = (
            f"emit(directory, 'child_started', child_pid={unrelated.pid}, "
            f"child_created={psutil.Process(unrelated.pid).create_time() + 1})\n"
            "with configured_case(spec, directory):\n    import entry\n"
            "emit(directory, 'success')\n"
        )
        worker = _worker(worker_repo, body)
        result = common.run_case(
            tmp_path / "unrelated",
            worker=worker,
            spec=_spec(config="leaf.toml", entrypoint="entry.py"),
        )
        assert unrelated.poll() is None
        assert str(unrelated.pid) not in result["processes"]
        assert unrelated.pid not in result["processes"]
    finally:
        unrelated.terminate()
        unrelated.communicate(timeout=5)


def test_finally_cleanup_is_recorded_and_fails(worker_repo, tmp_path, monkeypatch):
    worker = _worker(worker_repo, "import time; time.sleep(30)\n")
    original = subprocess.Popen

    class BrokenCommunication(original):
        def communicate(self, *args, **kwargs):
            if self.args[-1] == worker and not getattr(self, "injected", False):
                self.injected = True
                raise RuntimeError("deliberate communicate failure")
            return super().communicate(*args, **kwargs)

    monkeypatch.setattr(common.subprocess, "Popen", BrokenCommunication)
    with pytest.raises(AssertionError, match="Phase4 case failed"):
        common.run_case(
            tmp_path / "fallback",
            worker=worker,
            spec=_spec(config="leaf.toml", entrypoint="entry.py"),
        )
    record = json.loads((tmp_path / "fallback/result.json").read_text())
    assert any(action.startswith("finally-KILL") for action in record["forced"])
    assert record["survivors"] == []
    assert any("deliberate communicate failure" in error for error in record["errors"])


def test_original_nonzero_exit_and_serial_execution_guard(policy):
    options = {"test_profile": "examples-phase4", "numprocesses": 2}
    config = SimpleNamespace(
        rootpath=REPO,
        args=[MODULE],
        invocation_params=SimpleNamespace(dir=REPO),
        getoption=lambda name, default=None: options.get(name, default),
    )
    with pytest.raises(pytest.UsageError, match="serial execution"):
        policy._ProfileChecks(config)
    options.pop("numprocesses")
    checks = policy._ProfileChecks(config)
    checks.phase4_expected = Counter({MODULE + "::test_case": 1})
    checks.phase4_passed = checks.phase4_expected.copy()
    session = SimpleNamespace(exitstatus=pytest.ExitCode.INTERRUPTED, items=[])
    checks.pytest_sessionfinish(session, pytest.ExitCode.INTERRUPTED)
    assert session.exitstatus == pytest.ExitCode.INTERRUPTED
    assert any("original exit was nonzero" in error for error in checks.violations)
